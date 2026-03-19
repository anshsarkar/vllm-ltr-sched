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

IDX_TTFTS = 0
IDX_TPOTS = 1
IDX_LATENCIES = 2
IDX_NLATENCIES = 3

SCHED_NAMES = {
    "fcfs": "FCFS",
    "sjf": "Oracle SJF",
    "srtf-PO-X": "Oracle (SRTF-PO)",
    "opt-xxx": "LTR-Ranking",
    "tpt-class10-xxx": "Classification (10-class, w=820)",
    "tpt-class82-xxx": "Classification (82-class, w=100)",
    "tpt-pctl10-xxx": "Classification (10-class, percentile)",
    "tpt-width10-xxx": "Classification (~820-class, w=10)",
    "mlfq-base0.03-thres10": "MLFQ",
    "mlfq": "MLFQ",
}


def parse_pt_filename(filename):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        filename,
    )
    if not m:
        return None
    return {"scheduler": m.group(1), "request_rate": float(m.group(4))}


def get_latest_pt_files(folder):
    configs = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        info = parse_pt_filename(fname)
        if info is None:
            print(f"Warning: could not parse '{fname}', skipping")
            continue
        path = os.path.join(folder, fname)
        key = (info["scheduler"], info["request_rate"])
        configs[key].append((os.path.getmtime(path), path))

    latest = {}
    for key, files in configs.items():
        files.sort(reverse=True)
        latest[key] = files[0][1]
    return latest


def extract_metrics(folder):
    latest_files = get_latest_pt_files(folder)
    print(f"Found {len(latest_files)} unique configurations")

    rows = []
    for (scheduler, rate), path in sorted(latest_files.items()):
        data = torch.load(path, map_location="cpu", weights_only=False)
        ttfts_s = data[IDX_TTFTS]
        tpots_s = data[IDX_TPOTS]
        latencies_s = data[IDX_LATENCIES]
        nlatencies_s = data[IDX_NLATENCIES]

        rows.append({
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
        })

    rows.sort(key=lambda r: (r["scheduler"], r["request_rate"]))
    return rows


def plot_metrics(csv_path, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for sched in sorted(df["scheduler"].unique()):
        sched_data = df[df["scheduler"] == sched].sort_values("request_rate")
        ax.plot(
            sched_data["request_rate"],
            sched_data["mean_nlatency_ms"],
            marker="o", label=SCHED_NAMES.get(sched, sched), linewidth=2, markersize=6,
        )

    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("Mean Normalized Latency (ms/token)", fontsize=12)
    ax.set_title(f"Scheduler Comparison: Mean Normalized Latency ({tag})", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, f"mean_nlatency_comparison_{tag}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder", help="Folder containing .pt result files")
    parser.add_argument("-o", "--output", default=None, help="Output CSV path")
    parser.add_argument("--no-plot", action="store_true", help="Skip plot generation")
    args = parser.parse_args()

    folder = args.input_folder
    if not os.path.isdir(folder):
        print(f"Error: '{folder}' is not a directory")
        sys.exit(1)

    tag = os.path.basename(os.path.normpath(folder))
    results_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)
    output = args.output or os.path.join(results_dir, f"metrics_{tag}.csv")

    rows = extract_metrics(folder)
    if not rows:
        print("No .pt files found or none could be parsed.")
        sys.exit(1)

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

    if not args.no_plot:
        print("\nGenerating plot...")
        plot_metrics(output, results_dir, tag=tag)
        print("Done!")


if __name__ == "__main__":
    main()