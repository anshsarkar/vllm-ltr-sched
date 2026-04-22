#!/usr/bin/env python3
"""Static vs DSRTF comparison plots — one subplot per predictor.

Each subplot shows 5 lines: FCFS (baseline), Oracle SJF (ceiling),
Oracle SRTF (dynamic ceiling), the static predictor, and its DSRTF variant.
Layout: 5 rows (predictors) x 2 columns (datasets).

Usage (from repo root):
    python extension/analysis/plot_dsrtf_comparison.py

Outputs:
    extension/benchmarks/results/dsrtf_comparison.pdf
"""

import os
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "extension", "benchmarks", "results")

IDX_NLATENCIES = 3

DATASETS = ["lmsys", "sharegpt"]
DATASET_TITLES = {"lmsys": "LMSYS-Chat-1M", "sharegpt": "ShareGPT"}

# The 5 predictors that have both static and DSRTF variants
PREDICTORS = [
    ("tpt-class10-xxx",    "dsrtf-tpt-class10-xxx",    "Class-10 (w=820)"),
    ("tpt-class82-xxx",    "dsrtf-tpt-class82-xxx",    "Class-82 (w=100)"),
    ("tpt-width10-xxx",    "dsrtf-tpt-width10-xxx",    "Class-820 (w=10)"),
    ("tpt-pctl10-xxx",     "dsrtf-tpt-pctl10-xxx",     "Pctl-10 (CE)"),
    ("tpt-pctl10-mse-xxx", "dsrtf-tpt-pctl10-mse-xxx", "Pctl-10 (MSE)"),
]

# Shared style for baselines and static/dsrtf
STYLE_FCFS   = {"color": "#888888", "marker": "s", "ls": "--", "lw": 1.4, "ms": 4}
STYLE_SJF    = {"color": "#2ca02c", "marker": "D", "ls": "--", "lw": 1.4, "ms": 4}
STYLE_SRTF   = {"color": "#ff7f0e", "marker": "d", "ls": "--", "lw": 1.4, "ms": 4}
STYLE_STATIC = {"color": "#1f77b4", "marker": "o", "ls": "-",  "lw": 1.8, "ms": 5}
STYLE_DSRTF  = {"color": "#d62728", "marker": "^", "ls": "-",  "lw": 1.8, "ms": 5}


def parse_filename(fname):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        fname,
    )
    if not m:
        return None
    return {"scheduler": m.group(1), "rate": float(m.group(4))}


def load_pt_files(folder):
    """Return dict: (scheduler, rate) -> mean_nlatency_s."""
    records = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        info = parse_filename(fname)
        if info is None:
            continue
        path = os.path.join(folder, fname)
        records[(info["scheduler"], info["rate"])].append(
            (os.path.getmtime(path), path)
        )

    results = {}
    for key, files in records.items():
        files.sort(reverse=True)
        path = files[0][1]
        data = torch.load(path, map_location="cpu", weights_only=False)
        nlatencies = np.array(data[IDX_NLATENCIES])
        results[key] = float(np.mean(nlatencies))
    return results


def main():
    # Load all data: static results from {dataset}/, dsrtf from dsrtf_{dataset}/
    all_data = {}
    for dataset in DATASETS:
        static_folder = os.path.join(RESULTS_ROOT, dataset)
        dsrtf_folder = os.path.join(RESULTS_ROOT, f"dsrtf_{dataset}")
        static = load_pt_files(static_folder)
        dsrtf = load_pt_files(dsrtf_folder)
        # Merge into one dict keyed by (scheduler, rate)
        merged = {}
        merged.update(static)
        merged.update(dsrtf)
        all_data[dataset] = merged

    fig, axes = plt.subplots(5, 2, figsize=(11, 18), sharex=True)

    for row, (static_key, dsrtf_key, pred_name) in enumerate(PREDICTORS):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            data = all_data[dataset]

            rates = sorted({r for (_, r) in data})

            # Plot 5 lines: baselines + static + dsrtf
            for sched_key, label, style in [
                ("fcfs",         "FCFS",         STYLE_FCFS),
                ("sjf",          "Oracle SJF",   STYLE_SJF),
                ("srtf-oracle",  "Oracle SRTF",  STYLE_SRTF),
                (static_key,     "Static",       STYLE_STATIC),
                (dsrtf_key,      "DSRTF",        STYLE_DSRTF),
            ]:
                ys = [data.get((sched_key, r), np.nan) for r in rates]
                ax.plot(rates, ys, label=label,
                        color=style["color"], marker=style["marker"],
                        linestyle=style["ls"], linewidth=style["lw"],
                        markersize=style["ms"])

            # Subplot title: predictor name on every panel, dataset on top row
            title = pred_name
            if row == 0:
                title = f"{DATASET_TITLES[dataset]}\n{pred_name}"
            ax.set_title(title, fontsize=10, fontweight="bold")

            if col == 0:
                ax.set_ylabel("Mean norm. latency (s/token)", fontsize=9)

            if row == len(PREDICTORS) - 1:
                ax.set_xlabel("Request rate (req/s)", fontsize=10)

            ax.set_xticks(rates)
            ax.grid(True, linestyle=":", alpha=0.5)

    # Single shared legend at the bottom — labels are role-based, same for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Static vs DSRTF Scheduling: Per-Predictor Comparison",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = os.path.join(RESULTS_ROOT, "dsrtf_comparison.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {out_path}")

    # Also print a summary table
    print("\n=== Mean nlatency summary (static vs DSRTF) ===\n")
    for dataset in DATASETS:
        data = all_data[dataset]
        rates = sorted({r for (_, r) in data})
        print(f"--- {DATASET_TITLES[dataset]} ---")
        print(f"{'Predictor':<22} {'Rate':>6} {'Static':>10} {'DSRTF':>10} {'Delta':>10} {'%Change':>10}")
        print("-" * 72)
        for static_key, dsrtf_key, pred_name in PREDICTORS:
            for rate in rates:
                s = data.get((static_key, rate), np.nan)
                d = data.get((dsrtf_key, rate), np.nan)
                delta = d - s
                pct = 100 * delta / s if s != 0 else np.nan
                print(f"{pred_name:<22} {rate:>6.0f} {s:>10.4f} {d:>10.4f} {delta:>+10.4f} {pct:>+9.1f}%")
        print()


if __name__ == "__main__":
    main()
