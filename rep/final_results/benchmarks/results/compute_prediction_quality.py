#!/usr/bin/env python3
"""Compute prediction quality summary from benchmark prediction_metrics.jsonl files.

Reads pre-computed metrics (Kendall's Tau, accuracy, Spearman's Rho, MAE, etc.)
logged during v2 benchmark runs and produces summary CSVs for the paper.
"""

import csv
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Display order and labels
SCHED_LABELS = {
    "srtf-PO-X":         "Oracle (Pre-run)",
    "opt-xxx":            "Ranking",
    "tpt-class10-xxx":    "Classification (#Buckets=10)",
    "tpt-class82-xxx":    "Classification (Bucket Size=100)",
    "tpt-width10-xxx":    "Classification (Bucket Size=10)",
    "tpt-pctl10-xxx":     "Classification (Percentile, CE)",
    "tpt-pctl10-mse-xxx": "Classification (Percentile, MSE)",
}

SCHED_ORDER = list(SCHED_LABELS.keys())


def load_metrics(dataset_dir):
    """Load prediction_metrics.jsonl from a dataset directory."""
    path = os.path.join(dataset_dir, "prediction_metrics.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def process_dataset(dataset_dir, dataset_name):
    """Read metrics and normalize into rows."""
    raw = load_metrics(dataset_dir)
    rows = []
    for r in raw:
        sched = r["schedule_type"]
        if sched not in SCHED_LABELS:
            continue
        rows.append({
            "dataset":       dataset_name,
            "scheduler":     sched,
            "label":         SCHED_LABELS[sched],
            "request_rate":  r["request_rate"],
            "kendall_tau":   r["kendall_tau"],
            "spearman_rho":  r.get("spearman_rho"),
            "accuracy":      r.get("accuracy"),
            "mae":           r.get("mae"),
            "num_labels":    r.get("num_labels"),
            "n_requests":    r["n_requests"],
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

    # ── Per-rate CSV ──────────────────────────────────────────────────────
    per_rate_path = os.path.join(output_dir, "prediction_quality_per_rate.csv")
    per_rate_fields = ["dataset", "scheduler", "label", "request_rate",
                       "kendall_tau", "spearman_rho", "accuracy", "mae",
                       "num_labels", "n_requests"]
    with open(per_rate_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_rate_fields,
                                extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(all_rows,
                                key=lambda r: (r["dataset"],
                                               SCHED_ORDER.index(r["scheduler"])
                                               if r["scheduler"] in SCHED_ORDER else 99,
                                               r["request_rate"])))
    print(f"\nPer-rate CSV: {per_rate_path}")

    # ── Summary CSV (rate=64 only) ────────────────────────────────────────
    rate64 = [r for r in all_rows if r["request_rate"] == 64.0]

    summary_rows = []
    for r in sorted(rate64, key=lambda x: (x["dataset"],
                                            SCHED_ORDER.index(x["scheduler"])
                                            if x["scheduler"] in SCHED_ORDER else 99)):
        acc = r["accuracy"]
        rho = r["spearman_rho"]
        mae = r["mae"]
        summary_rows.append({
            "dataset":      r["dataset"],
            "scheduler":    r["scheduler"],
            "label":        r["label"],
            "num_labels":   r["num_labels"] if r["num_labels"] else "N/A",
            "abs_tau":      round(abs(r["kendall_tau"]), 4),
            "abs_rho":      round(abs(rho), 4) if rho is not None else "N/A",
            "accuracy_pct": round(acc * 100, 2) if acc is not None else "N/A",
            "mae":          round(mae, 4) if mae is not None else "N/A",
            "n_requests":   r["n_requests"],
        })

    agg_path = os.path.join(output_dir, "prediction_quality_summary.csv")
    agg_fields = ["dataset", "scheduler", "label", "num_labels",
                  "abs_tau", "abs_rho", "accuracy_pct", "mae", "n_requests"]
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {agg_path}")

    # ── Print table ───────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    for dataset in ["lmsys", "sharegpt"]:
        ds_rows = [r for r in summary_rows if r["dataset"] == dataset]
        if not ds_rows:
            continue
        print(f"\n  {dataset.upper()} (request_rate=64)")
        print(f"  {'Method':<40} {'|Tau|':>7} {'|Rho|':>7} {'Acc%':>8} {'MAE':>7} {'#Lbl':>5}")
        print(f"  {'-'*76}")
        for r in ds_rows:
            acc = f"{r['accuracy_pct']:>7.2f}%" if r['accuracy_pct'] != "N/A" else "     N/A"
            rho = f"{r['abs_rho']:>7.4f}" if r['abs_rho'] != "N/A" else "    N/A"
            mae = f"{r['mae']:>7.4f}" if r['mae'] != "N/A" else "    N/A"
            nlbl = f"{r['num_labels']:>5}" if r['num_labels'] != "N/A" else "  N/A"
            print(f"  {r['label']:<40} {r['abs_tau']:>7.4f} {rho} {acc} {mae} {nlbl}")
    print(f"\n{'='*90}")


if __name__ == "__main__":
    main()
