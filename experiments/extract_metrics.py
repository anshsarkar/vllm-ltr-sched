#!/usr/bin/env python3

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# [ttfts, tpots, latencies, nlatencies, output_lens, input_lens, est_lens, texts, aux_model_scores, pred_scores]
IDX_TTFTS = 0
IDX_TPOTS = 1
IDX_LATENCIES = 2
IDX_NLATENCIES = 3
IDX_OUTPUT_LENS = 4
IDX_INPUT_LENS = 5


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
        "model": m.group(2),
        "request_rate": float(m.group(4)),
    }


def get_latest_pt_files(folder):
    """Get the latest .pt file for each unique scheduler-rate configuration.

    Returns dict: (scheduler, rate) -> full_path
    """
    configs = defaultdict(list)  # (scheduler, rate) -> [(mtime, filepath)]

    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue

        info = parse_pt_filename(fname)
        if info is None:
            print(f"Warning: could not parse '{fname}', skipping")
            continue

        path = os.path.join(folder, fname)
        mtime = os.path.getmtime(path)
        key = (info["scheduler"], info["request_rate"])
        configs[key].append((mtime, path, fname))

    # For each config, keep only the latest file
    latest = {}
    for key, files in configs.items():
        files.sort(reverse=True)  # sort by mtime descending
        latest[key] = files[0][1]  # path of newest file

    return latest


def plot_metrics(csv_path, output_dir):
    """Generate normalized latency comparison plot from CSV."""
    os.makedirs(output_dir, exist_ok=True)

    # Read CSV
    df = pd.read_csv(csv_path)

    # Friendly scheduler names for plots
    sched_names = {
        "fcfs": "FCFS",
        "srtf-PO-X": "Oracle (SRTF-PO)",
        "opt-xxx": "LTR-Ranking",
        "tpt-class10-xxx": "LTR-Classification",
        "mlfq-base0.03-thres10": "MLFQ",
        "mlfq": "MLFQ",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each scheduler that exists in the CSV
    for sched in sorted(df["scheduler"].unique()):
        sched_data = df[df["scheduler"] == sched].sort_values("request_rate")
        label = sched_names.get(sched, sched)
        ax.plot(sched_data["request_rate"], sched_data["mean_nlatency_ms"],
               marker="o", label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("Mean Normalized Latency (ms/token)", fontsize=12)
    ax.set_title("Scheduler Comparison: Mean Normalized Latency", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, "mean_nlatency_comparison_lmsys.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract benchmark metrics from .pt files to CSV and generate plots"
    )
    parser.add_argument("input_folder", help="Folder containing .pt result files")
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output CSV path (default: <input_folder>/metrics.csv)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison plots (saved to <input_folder>/analysis/)",
    )
    args = parser.parse_args()

    folder = args.input_folder
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory")
        sys.exit(1)

    output = args.output or os.path.join(folder, "metrics_sharegpt.csv")

    # Get latest .pt file for each config
    latest_files = get_latest_pt_files(folder)
    print(f"Found {len(latest_files)} unique configurations")

    rows = []
    for (scheduler, rate), path in sorted(latest_files.items()):
        data = torch.load(path, map_location="cpu", weights_only=False)

        ttfts_s = data[IDX_TTFTS]
        tpots_s = data[IDX_TPOTS]
        latencies_s = data[IDX_LATENCIES]
        nlatencies_s = data[IDX_NLATENCIES]

        row = {
            "scheduler": scheduler,
            "request_rate": rate,
            "num_requests": len(latencies_s),
            "mean_ttft_ms": round(np.mean(ttfts_s) * 1000, 2),
            "median_ttft_ms": round(np.median(ttfts_s) * 1000, 2),
            "p99_ttft_ms": round(np.percentile(ttfts_s, 99) * 1000, 2),
            "mean_tpot_ms": round(np.mean(tpots_s) * 1000, 2),
            "mean_nlatency_ms": round(np.mean(nlatencies_s) * 1000, 2),
            "median_nlatency_ms": round(np.median(nlatencies_s) * 1000, 2),
            "p90_nlatency_ms": round(np.percentile(nlatencies_s, 90) * 1000, 2),
            "p99_nlatency_ms": round(np.percentile(nlatencies_s, 99) * 1000, 2),
        }
        rows.append(row)

    if not rows:
        print("No .pt files found or none could be parsed.")
        sys.exit(1)

    rows.sort(key=lambda r: (r["scheduler"], r["request_rate"]))

    fieldnames = [
        "scheduler", "request_rate", "num_requests",
        "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
        "mean_tpot_ms",
        "mean_nlatency_ms", "median_nlatency_ms", "p90_nlatency_ms", "p99_nlatency_ms",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output}")

    if args.plot:
        analysis_dir = "experiments/analysis"
        print(f"\nGenerating plots in {analysis_dir}/")
        plot_metrics(output, analysis_dir)
        print("Done!")


if __name__ == "__main__":
    main()
