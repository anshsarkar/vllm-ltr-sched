#!/usr/bin/env python3
"""
Per-bucket prediction error analysis for LTR-Ranking vs LTR-Classification.

Reads the .pt result files (which contain aux_model_scores and actual_output_lens)
and computes per-bucket metrics comparing both models side by side.

Output per QPS:
  1. LTR-Ranking bucket table (True#, Pred#, Correct#, Kendall tau)
  2. LTR-Classification bucket table A — same buckets as ranking (+ MAE)
  3. LTR-Classification bucket table B — native 820-token class buckets (+ P/R/F1)

Also exports a CSV with per-request actual vs predicted lengths for both models.

Usage:
  python experiments/analyze_per_bucket_errors.py experiments/results/sharegpt_8b_h100_metrics
  python experiments/analyze_per_bucket_errors.py experiments/results/lmsys_8b_h100_metrics
  python experiments/analyze_per_bucket_errors.py experiments/results/sharegpt_8b_h100_metrics experiments/results/lmsys_8b_h100_metrics
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy import stats

# .pt file indices (from benchmark_serving_real_with_metrics.py)
IDX_OUTPUT_LENS = 4
IDX_INPUT_LENS = 5
IDX_AUX_MODEL_SCORES = 8
IDX_PRED_SCORES = 9

# Predefined linear buckets (no log scale)
BUCKET_EDGES = np.array([0, 256, 512, 768, 1024, np.inf])
BUCKET_LABELS = ["1-256", "257-512", "513-768", "769-1024", "1025+"]

# Training parameters (from train.sh)
LABEL_MAX_LENGTH = 8192


def parse_pt_filename(filename):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        filename,
    )
    if not m:
        return None
    return {
        "scheduler": m.group(1),
        "request_rate": float(m.group(4)),
    }


def load_pt_data(folder):
    """Load all .pt files, return list of dicts with scores and output lens."""
    results = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith(".pt"):
            continue
        info = parse_pt_filename(fname)
        if info is None:
            continue
        sched = info["scheduler"]
        if not (sched.startswith("opt") or sched.startswith("tpt")):
            continue

        path = os.path.join(folder, fname)
        data = torch.load(path, map_location="cpu", weights_only=False)

        output_lens = np.array(data[IDX_OUTPUT_LENS], dtype=float)
        scores = np.array(data[IDX_AUX_MODEL_SCORES], dtype=float)
        input_lens = np.array(data[IDX_INPUT_LENS], dtype=float)

        if len(scores) == 0 or np.all(np.isnan(scores)):
            continue

        results.append({
            "scheduler": sched,
            "rate": info["request_rate"],
            "scores": scores,
            "output_lens": output_lens,
            "input_lens": input_lens,
        })
    return results


# ── Score → Predicted Length Conversion ───────────────────────────────────────

def classification_score_to_length(scores, schedule_type,
                                   label_max_length=LABEL_MAX_LENGTH):
    """Convert classification scores (class indices) to predicted output lengths.

    Uses midpoint of each class's length range.
    """
    match = re.search(r'class(\d+)', schedule_type)
    if not match:
        return None
    num_labels = int(match.group(1))
    group_size = label_max_length // num_labels
    max_label_val = label_max_length // group_size

    pred_classes = np.clip(np.round(scores).astype(int), 0, num_labels - 1)
    # Midpoint of each class's length range
    pred_lens = ((max_label_val - pred_classes) - 0.5) * group_size
    pred_lens = np.clip(pred_lens, 0, label_max_length)
    return pred_lens


# ── Bucketing ────────────────────────────────────────────────────────────────

def assign_buckets(lengths, edges=BUCKET_EDGES, labels=BUCKET_LABELS):
    """Assign lengths to predefined linear buckets. Returns indices (0..N-1)."""
    bucket_idx = np.digitize(lengths, edges[1:-1])
    bucket_idx = np.clip(bucket_idx, 0, len(labels) - 1)
    return bucket_idx


# ── Per-Bucket Metrics ───────────────────────────────────────────────────────

def compute_ranking_bucket_table(scores, output_lens, edges=BUCKET_EDGES,
                                 labels=BUCKET_LABELS):
    """Compute per-bucket Kendall tau for ranking (no length conversion needed)."""
    true_bucket = assign_buckets(output_lens, edges, labels)

    rows = []
    for b in range(len(labels)):
        true_mask = true_bucket == b
        true_count = int(true_mask.sum())

        tau = float("nan")
        if true_count >= 2:
            tau, _ = stats.kendalltau(scores[true_mask], output_lens[true_mask])

        rows.append({
            "label": labels[b],
            "true_count": true_count,
            "kendall_tau": float(tau),
        })

    overall_tau, _ = stats.kendalltau(scores, output_lens)
    return rows, float(overall_tau)


def compute_classification_bucket_table(scores, output_lens, schedule_type,
                                        edges=BUCKET_EDGES, labels=BUCKET_LABELS):
    """Compute per-bucket metrics for classification: True#, Pred#, Correct#, MAE, Kendall tau."""
    true_bucket = assign_buckets(output_lens, edges, labels)
    pred_lens = classification_score_to_length(scores, schedule_type)
    pred_bucket = assign_buckets(pred_lens, edges, labels)

    rows = []
    for b in range(len(labels)):
        true_mask = true_bucket == b
        pred_mask = pred_bucket == b
        true_count = int(true_mask.sum())
        pred_count = int(pred_mask.sum())
        correct = int((true_mask & pred_mask).sum())  # TP

        tau = float("nan")
        if true_count >= 2:
            tau, _ = stats.kendalltau(scores[true_mask], output_lens[true_mask])

        mae = float("nan")
        if true_count > 0:
            mae = float(np.mean(np.abs(pred_lens[true_mask] - output_lens[true_mask])))

        rows.append({
            "label": labels[b],
            "true_count": true_count,
            "pred_count": pred_count,
            "correct": correct,
            "kendall_tau": float(tau),
            "mae_tokens": mae,
        })

    overall_tau, _ = stats.kendalltau(scores, output_lens)
    overall_mae = float(np.mean(np.abs(pred_lens - output_lens)))
    return rows, float(overall_tau), overall_mae


def len2label(length, label_max_length, label_group_size):
    """Replicate trainer.py RankingDataset.__len2label__."""
    return label_max_length // label_group_size - min(label_max_length, length) // label_group_size


def compute_native_class_table(scores, output_lens, schedule_type,
                               label_max_length=LABEL_MAX_LENGTH):
    """Compute per-class metrics using the classification model's native buckets.

    Returns per-class True#, Pred#, TP, Precision, Recall, F1, MAE.
    """
    match = re.search(r'class(\d+)', schedule_type)
    if not match:
        return None
    num_labels = int(match.group(1))
    group_size = label_max_length // num_labels

    pred_labels = np.clip(np.round(scores).astype(int), 0, num_labels - 1)
    true_labels = np.array([
        len2label(int(l), label_max_length, group_size)
        for l in output_lens
    ], dtype=int)
    true_labels = np.clip(true_labels, 0, num_labels - 1)

    pred_lens = classification_score_to_length(scores, schedule_type, label_max_length)
    max_label_val = label_max_length // group_size

    rows = []
    for c in range(num_labels):
        true_mask = true_labels == c
        pred_mask = pred_labels == c

        tp = int((true_mask & pred_mask).sum())
        fp = int((~true_mask & pred_mask).sum())
        fn = int((true_mask & ~pred_mask).sum())
        true_count = int(true_mask.sum())
        pred_count = int(pred_mask.sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Length range for this class
        upper = (max_label_val - c) * group_size
        lower = max(0, upper - group_size + 1)
        if c == 0:
            upper = label_max_length

        # MAE for requests truly in this class
        mae = float("nan")
        if true_count > 0:
            mae = float(np.mean(np.abs(pred_lens[true_mask] - output_lens[true_mask])))

        rows.append({
            "class": c,
            "lower": lower, "upper": upper,
            "true_count": true_count, "pred_count": pred_count,
            "correct": tp, "precision": precision, "recall": recall, "f1": f1,
            "mae_tokens": mae,
        })

    # Summary stats
    n = len(scores)
    micro_f1 = float(np.sum(pred_labels == true_labels)) / n
    occupied_f1s = [r["f1"] for r in rows if r["true_count"] > 0]
    macro_f1 = float(np.mean(occupied_f1s)) if occupied_f1s else 0.0
    majority_class = int(np.bincount(true_labels, minlength=num_labels).argmax())
    majority_acc = float(np.sum(true_labels == majority_class)) / n
    n_distinct = int(np.sum(np.bincount(pred_labels, minlength=num_labels) > 0))

    return {
        "rows": rows,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "majority_class": majority_class,
        "majority_acc": majority_acc,
        "n_distinct": n_distinct,
        "num_labels": num_labels,
    }


# ── Printing ─────────────────────────────────────────────────────────────────

def print_ranking_table(rows, overall_tau, rate, n):
    """Print the LTR-Ranking bucket table (Kendall tau only)."""
    print(f"\n  ┌{'─' * 68}┐")
    print(f"  │  LTR-RANKING  |  {rate} req/s  |  N = {n:<26}│")
    print(f"  └{'─' * 68}┘")
    print(f"  Overall Kendall Tau: {overall_tau:+.4f}")
    print(f"\n  {'Bucket':<12} {'True#':>7} {'Kendall τ':>10}")
    print(f"  {'─' * 32}")
    for r in rows:
        tau_str = f"{r['kendall_tau']:>+10.4f}" if not np.isnan(r["kendall_tau"]) else f"{'n/a':>10}"
        print(f"  {r['label']:<12} {r['true_count']:>7} {tau_str}")


def print_classification_table_a(rows, overall_tau, overall_mae, rate, n):
    """Print the LTR-Classification bucket table (same buckets as ranking)."""
    print(f"\n  ┌{'─' * 68}┐")
    print(f"  │  LTR-CLASSIFICATION (common buckets)  |  {rate} req/s  |  N = {n:<5}│")
    print(f"  └{'─' * 68}┘")
    print(f"  Overall Kendall Tau: {overall_tau:+.4f}")
    print(f"  Overall MAE: {overall_mae:.1f} tokens")
    print(f"\n  {'Bucket':<12} {'True#':>7} {'Pred#':>7} {'Correct#':>9} {'MAE(tok)':>10} {'Kendall τ':>10}")
    print(f"  {'─' * 58}")
    for r in rows:
        tau_str = f"{r['kendall_tau']:>+10.4f}" if not np.isnan(r["kendall_tau"]) else f"{'n/a':>10}"
        mae_str = f"{r['mae_tokens']:>10.1f}" if not np.isnan(r["mae_tokens"]) else f"{'n/a':>10}"
        print(f"  {r['label']:<12} {r['true_count']:>7} {r['pred_count']:>7} "
              f"{r['correct']:>9} {mae_str} {tau_str}")


def print_classification_table_b(info, rate, n):
    """Print the LTR-Classification native class table."""
    print(f"\n  ┌{'─' * 68}┐")
    print(f"  │  LTR-CLASSIFICATION (native {info['num_labels']}-class buckets)             "
          f"         │")
    print(f"  └{'─' * 68}┘")
    print(f"  Micro F1 (accuracy):     {info['micro_f1']:.4f}")
    print(f"  Macro F1:                {info['macro_f1']:.4f}")
    print(f"  Majority-class baseline: {info['majority_acc']:.4f}  (always predict class {info['majority_class']})")
    gap = info['micro_f1'] - info['majority_acc']
    print(f"  Model lift over baseline: {gap:+.4f}  ({'better' if gap > 0 else 'WORSE or equal'})")
    print(f"  Distinct classes predicted: {info['n_distinct']}/{info['num_labels']}")

    print(f"\n  {'Class':>6} {'Len Range':>14} {'True#':>7} {'Pred#':>7} {'Correct#':>9}"
          f" {'Prec':>7} {'Recall':>7} {'F1':>7} {'MAE(tok)':>10}")
    print(f"  {'─' * 82}")
    for r in info["rows"]:
        mae_str = f"{r['mae_tokens']:>10.1f}" if not np.isnan(r["mae_tokens"]) else f"{'n/a':>10}"
        print(f"  {r['class']:>6} {r['lower']:>6}-{r['upper']:<6}  {r['true_count']:>7} {r['pred_count']:>7}"
              f" {r['correct']:>9} {r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f} {mae_str}")


# ── CSV Export ───────────────────────────────────────────────────────────────

def export_csv(folder, results, out_dir):
    """Export per-request CSV with actual output length, raw scores, and
    classification predicted length.

    Matches ranking and classification runs at the same QPS rate.
    """
    dataset_name = os.path.basename(folder)

    by_sched = defaultdict(list)
    for r in results:
        by_sched[r["scheduler"]].append(r)

    ranking_scheds = [s for s in by_sched if s.startswith("opt")]
    class_scheds = [s for s in by_sched if s.startswith("tpt")]

    if not class_scheds:
        print(f"  Skipping CSV export: no classification data found")
        return

    cls_sched = class_scheds[0]
    cls_by_rate = {r["rate"]: r for r in by_sched[cls_sched]}

    # If ranking data exists, include raw scores (but not predicted lengths)
    rank_by_rate = {}
    if ranking_scheds:
        rank_by_rate = {r["rate"]: r for r in by_sched[ranking_scheds[0]]}

    os.makedirs(out_dir, exist_ok=True)

    for rate in sorted(cls_by_rate.keys()):
        cls_run = cls_by_rate[rate]
        rank_run = rank_by_rate.get(rate)

        n = len(cls_run["output_lens"])
        if rank_run is not None:
            n = min(n, len(rank_run["output_lens"]))

        cls_pred = classification_score_to_length(cls_run["scores"][:n], cls_sched)

        csv_path = os.path.join(out_dir, f"{dataset_name}_predictions_r{rate:.0f}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "request_idx", "actual_output_len",
                "classification_pred_len", "classification_score",
                "ranking_score",
            ])
            for i in range(n):
                rank_score = f"{rank_run['scores'][i]:.4f}" if rank_run is not None else ""
                writer.writerow([
                    i,
                    int(cls_run["output_lens"][i]),
                    f"{cls_pred[i]:.1f}",
                    f"{cls_run['scores'][i]:.1f}",
                    rank_score,
                ])
        print(f"  Saved CSV: {csv_path}")


# ── Main Analysis ────────────────────────────────────────────────────────────

def print_analysis(folder, results, out_dir):
    """Print side-by-side analysis: ranking then classification per QPS."""
    dataset_name = os.path.basename(folder)
    print(f"\n{'=' * 80}")
    print(f"  PREDICTION ANALYSIS: {dataset_name}")
    print(f"{'=' * 80}")

    by_sched = defaultdict(list)
    for r in results:
        by_sched[r["scheduler"]].append(r)

    ranking_scheds = [s for s in by_sched if s.startswith("opt")]
    class_scheds = [s for s in by_sched if s.startswith("tpt")]

    # Collect all rates
    all_rates = sorted(set(r["rate"] for r in results))

    for rate in all_rates:
        print(f"\n{'━' * 80}")
        print(f"  QPS = {rate} req/s")
        print(f"{'━' * 80}")

        # ── LTR-Ranking ──
        for sched in ranking_scheds:
            runs_at_rate = [r for r in by_sched[sched] if abs(r["rate"] - rate) < 0.1]
            for run in runs_at_rate:
                rows, overall_tau = compute_ranking_bucket_table(
                    run["scores"], run["output_lens"])
                print_ranking_table(rows, overall_tau, rate, len(run["scores"]))

        # ── LTR-Classification ──
        for sched in class_scheds:
            runs_at_rate = [r for r in by_sched[sched] if abs(r["rate"] - rate) < 0.1]
            for run in runs_at_rate:
                # Table A: same buckets as ranking
                rows, overall_tau, overall_mae = compute_classification_bucket_table(
                    run["scores"], run["output_lens"], sched)
                print_classification_table_a(rows, overall_tau, overall_mae,
                                             rate, len(run["scores"]))

                # Table B: native class buckets
                native = compute_native_class_table(
                    run["scores"], run["output_lens"], sched)
                if native:
                    print_classification_table_b(native, rate, len(run["scores"]))

    # Export CSV
    print(f"\n{'─' * 80}")
    export_csv(folder, results, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Per-bucket prediction error analysis for LTR models"
    )
    parser.add_argument(
        "input_folders", nargs="+",
        help="Folder(s) containing .pt result files"
    )
    parser.add_argument(
        "--rate", type=float, default=None,
        help="Only analyze this request rate (e.g. 8.0)"
    )
    parser.add_argument(
        "--out-dir", type=str, default="experiments/analysis/predictions",
        help="Output directory for CSV files"
    )
    args = parser.parse_args()

    for folder in args.input_folders:
        if not os.path.isdir(folder):
            print(f"Warning: '{folder}' is not a directory, skipping")
            continue

        results = load_pt_data(folder)
        if args.rate is not None:
            results = [r for r in results if abs(r["rate"] - args.rate) < 0.1]

        if not results:
            print(f"No valid .pt files found in {folder}")
            continue

        print_analysis(folder, results, args.out_dir)


if __name__ == "__main__":
    main()
