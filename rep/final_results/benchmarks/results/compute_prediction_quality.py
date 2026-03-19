#!/usr/bin/env python3
"""Compute Kendall's Tau and accuracy from benchmark .pt files."""

import csv
import os
import re

import numpy as np
import torch
from scipy.stats import kendalltau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IDX_OUTPUT_LENS = 4
IDX_AUX_SCORES = 8

LABEL_MAX_LENGTH = 8192

# Schedulers to include (no PO — it's an oracle, not a predictor)
PRED_SCHEDULERS = {"opt-xxx", "tpt-class10-xxx", "tpt-class82-xxx",
                   "tpt-width10-xxx", "tpt-pctl10-xxx", "tpt-pctl10-mse-xxx"}

SCHED_LABELS = {
    "opt-xxx":           "Ranking",
    "tpt-class10-xxx":   "Classification (#Buckets=10)",
    "tpt-class82-xxx":   "Classification (Bucket Size=100)",
    "tpt-width10-xxx":   "Classification (Bucket Size=10)",
    "tpt-pctl10-xxx":    "Classification (Percentile, CE)",
    "tpt-pctl10-mse-xxx":"Classification (Percentile, MSE)",
}

# Schedulers where accuracy is meaningful (uniform-bucket classifiers only).
# Ranking uses continuous scores (not class labels), and percentile classifiers
# predict percentile-bucket IDs that don't match uniform true labels.
ACCURACY_VALID = {"tpt-class10-xxx", "tpt-class82-xxx", "tpt-width10-xxx"}


def parse_filename(fname):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        fname,
    )
    if not m:
        return None
    return {"scheduler": m.group(1), "request_rate": float(m.group(4))}


def get_num_labels(schedule_type):
    match = re.search(r'class(\d+)', schedule_type)
    if match:
        return int(match.group(1))
    if "width10" in schedule_type:
        return LABEL_MAX_LENGTH // 10  # 820
    if "pctl10" in schedule_type:
        return 10
    return 10  # default


def len2label(length, num_labels, group_size):
    """Replicate the authors' __len2label__ from trainer.py."""
    return min(num_labels - 1, max(0,
        LABEL_MAX_LENGTH // group_size
        - min(LABEL_MAX_LENGTH, int(length)) // group_size))


def compute_metrics(scores, lengths, schedule_type):
    scores = np.array(scores, dtype=float)
    lengths = np.array(lengths, dtype=float)

    # Filter out any None/NaN
    valid = np.isfinite(scores) & np.isfinite(lengths)
    scores, lengths = scores[valid], lengths[valid]
    if len(scores) < 2:
        return None

    # Kendall's Tau (valid for all methods)
    tau, p_tau = kendalltau(scores, lengths)

    # Accuracy — only meaningful for uniform-bucket classifiers
    accuracy = None
    num_labels = get_num_labels(schedule_type)

    if schedule_type in ACCURACY_VALID:
        group_size = LABEL_MAX_LENGTH // num_labels
        true_labels = np.array([len2label(l, num_labels, group_size) for l in lengths])
        pred_labels = np.clip(np.round(scores).astype(int), 0, num_labels - 1)
        accuracy = float(np.mean(pred_labels == true_labels))

    return {
        "kendall_tau": float(tau),
        "kendall_tau_p": float(p_tau),
        "accuracy": accuracy,
        "num_labels": num_labels,
        "n_requests": len(scores),
    }


def process_dataset(dataset_dir, dataset_name):
    rows = []
    for fname in sorted(os.listdir(dataset_dir)):
        if not fname.endswith(".pt"):
            continue
        info = parse_filename(fname)
        if info is None:
            continue
        sched = info["scheduler"]
        if sched not in PRED_SCHEDULERS:
            continue

        path = os.path.join(dataset_dir, fname)
        data = torch.load(path, map_location="cpu", weights_only=False)
        scores = data[IDX_AUX_SCORES]
        lengths = data[IDX_OUTPUT_LENS]

        if scores is None or all(s is None for s in scores):
            continue

        metrics = compute_metrics(scores, lengths, sched)
        if metrics is None:
            continue

        rows.append({
            "dataset": dataset_name,
            "scheduler": sched,
            "label": SCHED_LABELS.get(sched, sched),
            "request_rate": info["request_rate"],
            **metrics,
        })

    return rows


def main():
    output_dir = os.path.join(BASE_DIR, "plots_for_paper")
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    for dataset in ["lmsys", "sharegpt"]:
        dataset_dir = os.path.join(BASE_DIR, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"Skipping {dataset} (not found)")
            continue
        print(f"Processing {dataset}...")
        rows = process_dataset(dataset_dir, dataset)
        all_rows.extend(rows)
        print(f"  {len(rows)} entries")

    if not all_rows:
        print("No data found.")
        return

    # Write per-rate CSV
    per_rate_path = os.path.join(output_dir, "prediction_quality_per_rate.csv")
    fieldnames = ["dataset", "scheduler", "label", "request_rate",
                  "kendall_tau", "accuracy", "num_labels", "n_requests"]
    with open(per_rate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda r: (r["dataset"], r["scheduler"], r["request_rate"])))
    print(f"\nPer-rate CSV: {per_rate_path}")

    # Summary uses only request_rate=64 results
    rate64 = [r for r in all_rows if r["request_rate"] == 64.0]

    summary_rows = []
    for r in sorted(rate64, key=lambda x: (x["dataset"], x["scheduler"])):
        acc = r["accuracy"]
        summary_rows.append({
            "dataset": r["dataset"],
            "scheduler": r["scheduler"],
            "label": r["label"],
            "num_labels": r["num_labels"],
            "abs_tau": round(abs(r["kendall_tau"]), 4),
            "accuracy_pct": round(acc * 100, 1) if acc is not None else "N/A",
            "n_requests": r["n_requests"],
        })

    agg_path = os.path.join(output_dir, "prediction_quality_summary.csv")
    agg_fields = ["dataset", "scheduler", "label", "num_labels",
                  "abs_tau", "accuracy_pct", "n_requests"]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {agg_path}")

    # Print table
    print(f"\n{'='*80}")
    for dataset in ["lmsys", "sharegpt"]:
        print(f"\n  {dataset.upper()} (request_rate=64, n=3840)")
        print(f"  {'Method':<40} {'Acc (%)':>8} {'|Tau|':>8} {'#Labels':>8}")
        print(f"  {'-'*66}")
        for r in summary_rows:
            if r["dataset"] == dataset:
                acc_str = f"{r['accuracy_pct']:>7.1f}%" if r['accuracy_pct'] != "N/A" else "     N/A"
                print(f"  {r['label']:<40} {acc_str} {r['abs_tau']:>8.4f} {r['num_labels']:>8}")
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
