#!/usr/bin/env python3

import os
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "extension", "benchmarks", "results")
OUT_DIR = os.path.join(REPO_ROOT, "extension", "analysis", "fixregion_pctl10mse_analysis")

IDX_NLATENCIES = 3

DATASETS = ["lmsys", "sharegpt"]
DATASET_TITLES = {"lmsys": "LMSYS-Chat-1M", "sharegpt": "ShareGPT"}
RATES = [32.0, 64.0]

# Fixregion variants we ran
FIXREGION_VARIANTS = [
    "fixshort3-tpt-pctl10-mse-xxx",
    "fixshort5-tpt-pctl10-mse-xxx",
    "fixlong3-tpt-pctl10-mse-xxx",
    "fixlong5-tpt-pctl10-mse-xxx",
]

# Short display names for plots
VARIANT_LABELS = {
    "fixshort3-tpt-pctl10-mse-xxx": "Fix Short 3",
    "fixshort5-tpt-pctl10-mse-xxx": "Fix Short 5",
    "fixlong3-tpt-pctl10-mse-xxx":  "Fix Long 3",
    "fixlong5-tpt-pctl10-mse-xxx":  "Fix Long 5",
}

# Baselines to load from the static results
BASELINES = ["fcfs", "sjf", "opt-xxx", "tpt-pctl10-mse-xxx"]
BASELINE_LABELS = {
    "fcfs": "FCFS",
    "sjf": "Oracle SJF",
    "opt-xxx": "Ranking (opt-xxx)",
    "tpt-pctl10-mse-xxx": "Pctl-10 MSE",
}

# Color scheme
COLORS_FIX = {
    "fixshort3-tpt-pctl10-mse-xxx": "#e377c2",
    "fixshort5-tpt-pctl10-mse-xxx": "#d62728",
    "fixlong3-tpt-pctl10-mse-xxx":  "#2ca02c",
    "fixlong5-tpt-pctl10-mse-xxx":  "#1f77b4",
}
COLORS_BASELINE = {
    "fcfs": "#888888",
    "sjf": "#2ca02c",
    "opt-xxx": "#ff7f0e",
    "tpt-pctl10-mse-xxx": "#1f77b4",
}


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
    records = defaultdict(list)
    if not os.path.isdir(folder):
        return {}
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


def load_all_data():
    all_data = {}
    for dataset in DATASETS:
        merged = {}
        static_folder = os.path.join(RESULTS_ROOT, dataset)
        merged.update(load_pt_files(static_folder))
        fix_folder = os.path.join(RESULTS_ROOT, f"fixregion_pctl10mse_{dataset}")
        merged.update(load_pt_files(fix_folder))
        all_data[dataset] = merged
    return all_data


def compute_gap_recovery(data, variant, rate):
    sjf = data.get(("sjf", rate))
    classifier = data.get(("tpt-pctl10-mse-xxx", rate))
    fixregion = data.get((variant, rate))
    if sjf is None or classifier is None or fixregion is None:
        return np.nan
    gap = classifier - sjf
    if gap <= 0:
        return np.nan
    return (classifier - fixregion) / gap * 100


def compute_remaining_gap(data, variant, rate):
    sjf = data.get(("sjf", rate))
    fixregion = data.get((variant, rate))
    if sjf is None or fixregion is None or sjf <= 0:
        return np.nan
    return fixregion / sjf


def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")


def print_and_save_summary(all_data):
    print("=== Fix-Region Phase 1 Summary (pctl10-mse) ===\n")
    csv_rows = []
    for dataset in DATASETS:
        data = all_data[dataset]
        print(f"--- {DATASET_TITLES[dataset]} ---")
        print(f"{'Scheduler':<35} {'Rate':>6} {'MeanNLat':>10} {'GapRecov%':>10} {'Ratio/SJF':>10}")
        print("-" * 75)
        # Baselines
        for sched in BASELINES:
            for rate in RATES:
                val = data.get((sched, rate), np.nan)
                sjf_val = data.get(("sjf", rate), np.nan)
                ratio = val / sjf_val if sjf_val and sjf_val > 0 else np.nan
                print(f"{BASELINE_LABELS.get(sched, sched):<35} {rate:>6.0f} {val:>10.4f} {'---':>10} {ratio:>10.2f}x")
                csv_rows.append({"dataset": dataset, "scheduler": sched, "rate": rate,
                                 "mean_nlatency": val, "gap_recovery_pct": "", "ratio_to_sjf": ratio})
        # Fixregion variants
        for variant in FIXREGION_VARIANTS:
            for rate in RATES:
                val = data.get((variant, rate), np.nan)
                gr = compute_gap_recovery(data, variant, rate)
                rg = compute_remaining_gap(data, variant, rate)
                print(f"{VARIANT_LABELS[variant]:<35} {rate:>6.0f} {val:>10.4f} {gr:>9.1f}% {rg:>10.2f}x")
                csv_rows.append({"dataset": dataset, "scheduler": variant, "rate": rate,
                                 "mean_nlatency": val, "gap_recovery_pct": gr, "ratio_to_sjf": rg})
        print()

    csv_path = os.path.join(OUT_DIR, "fixregion_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "scheduler", "rate", "mean_nlatency", "gap_recovery_pct", "ratio_to_sjf"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"  Summary CSV saved to {csv_path}\n")


def plot_gap_recovery(all_data):
    """Plot 1: Gap recovery % bar chart, one subplot per rate."""
    fig, axes = plt.subplots(1, len(RATES), figsize=(12, 5))
    for idx, rate in enumerate(RATES):
        ax = axes[idx]
        for dataset in DATASETS:
            data = all_data[dataset]
            recoveries = [compute_gap_recovery(data, v, rate) for v in FIXREGION_VARIANTS]
            x = np.arange(len(FIXREGION_VARIANTS))
            width = 0.35
            offset = -width / 2 if dataset == "lmsys" else width / 2
            color = "#1f77b4" if dataset == "lmsys" else "#ff7f0e"
            bars = ax.bar(x + offset, recoveries, width, label=DATASET_TITLES[dataset], color=color, alpha=0.8)
            for bar, val in zip(bars, recoveries):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(np.arange(len(FIXREGION_VARIANTS)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in FIXREGION_VARIANTS], fontsize=9)
        ax.set_ylabel("Gap Recovery %")
        ax.set_title(f"Rate = {rate:.0f} req/s", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Gap Recovery: Fraction of Classifier-to-SJF Gap Closed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "01_gap_recovery.pdf")


def plot_asymmetry(all_data):
    """Plot 2: Asymmetry — fixlongN vs fixshortN at same N."""
    pairs = [
        ("fixlong3-tpt-pctl10-mse-xxx", "fixshort3-tpt-pctl10-mse-xxx", "N=3"),
        ("fixlong5-tpt-pctl10-mse-xxx", "fixshort5-tpt-pctl10-mse-xxx", "N=5"),
    ]
    fig, axes = plt.subplots(1, len(RATES), figsize=(12, 5))
    for idx, rate in enumerate(RATES):
        ax = axes[idx]
        x = np.arange(len(DATASETS))
        width = 0.15
        for pi, (long_v, short_v, pair_label) in enumerate(pairs):
            long_vals = [compute_gap_recovery(all_data[d], long_v, rate) for d in DATASETS]
            short_vals = [compute_gap_recovery(all_data[d], short_v, rate) for d in DATASETS]
            offset_l = -width * (len(pairs) - pi) + width / 2
            offset_s = offset_l + width * len(pairs)
            ax.bar(x + offset_l, long_vals, width, label=f"Fix Long {pair_label}",
                   color="#2ca02c" if pi == 0 else "#1f77b4", alpha=0.8)
            ax.bar(x + offset_s, short_vals, width, label=f"Fix Short {pair_label}",
                   color="#e377c2" if pi == 0 else "#d62728", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([DATASET_TITLES[d] for d in DATASETS], fontsize=10)
        ax.set_ylabel("Gap Recovery %")
        ax.set_title(f"Rate = {rate:.0f} req/s", fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Asymmetry: Fix Long vs Fix Short at Same N", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "02_asymmetry.pdf")


def plot_marginal_recovery(all_data):
    """Plot 3: Marginal recovery curve — gap recovery % vs bins fixed from each end."""
    fig, axes = plt.subplots(1, len(RATES), figsize=(12, 5))
    for idx, rate in enumerate(RATES):
        ax = axes[idx]
        for dataset in DATASETS:
            data = all_data[dataset]
            long_bins = [3, 5]
            long_recoveries = [compute_gap_recovery(data, f"fixlong{n}-tpt-pctl10-mse-xxx", rate) for n in long_bins]
            short_bins = [3, 5]
            short_recoveries = [compute_gap_recovery(data, f"fixshort{n}-tpt-pctl10-mse-xxx", rate) for n in short_bins]

            color_l = "#2ca02c" if dataset == "lmsys" else "#1f77b4"
            color_s = "#d62728" if dataset == "lmsys" else "#ff7f0e"
            lstyle = "-" if dataset == "lmsys" else "--"

            ax.plot([0] + long_bins, [0] + long_recoveries, marker="o", ls=lstyle, color=color_l,
                    label=f"{DATASET_TITLES[dataset]} Fix Long", lw=2, ms=7)
            ax.plot([0] + short_bins, [0] + short_recoveries, marker="s", ls=lstyle, color=color_s,
                    label=f"{DATASET_TITLES[dataset]} Fix Short", lw=2, ms=7)

        ax.set_xlabel("Number of bins fixed", fontsize=10)
        ax.set_ylabel("Gap Recovery %")
        ax.set_title(f"Rate = {rate:.0f} req/s", fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_xticks([0, 1, 2, 3, 4, 5])
        ax.grid(True, linestyle=":", alpha=0.5)
    fig.suptitle("Marginal Recovery Curve: Gap Recovery vs Bins Fixed", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "03_marginal_recovery_curve.pdf")


def plot_remaining_gap(all_data):
    """Plot 4: Remaining gap to SJF — ratio of fixregion nlatency to SJF."""
    fig, axes = plt.subplots(1, len(RATES), figsize=(12, 5))
    for idx, rate in enumerate(RATES):
        ax = axes[idx]
        for dataset in DATASETS:
            data = all_data[dataset]
            classifier_ratio = compute_remaining_gap(data, "tpt-pctl10-mse-xxx", rate)
            ratios = [compute_remaining_gap(data, v, rate) for v in FIXREGION_VARIANTS]

            color = "#1f77b4" if dataset == "lmsys" else "#ff7f0e"
            x = np.arange(len(FIXREGION_VARIANTS))
            offset = -0.175 if dataset == "lmsys" else 0.175
            bars = ax.bar(x + offset, ratios, 0.35, label=DATASET_TITLES[dataset], color=color, alpha=0.8)
            for bar, val in zip(bars, ratios):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{val:.2f}x", ha="center", va="bottom", fontsize=7)

            lstyle = "-" if dataset == "lmsys" else "--"
            ax.axhline(y=classifier_ratio, color=color, ls=lstyle, lw=1.5, alpha=0.5,
                        label=f"Pctl-10 MSE ({DATASET_TITLES[dataset]})")

        ax.axhline(y=1.0, color="green", ls=":", lw=1.5, alpha=0.7)
        ax.text(len(FIXREGION_VARIANTS) - 0.5, 1.02, "Oracle SJF = 1.0x", fontsize=8, color="green", ha="right")
        ax.set_xticks(np.arange(len(FIXREGION_VARIANTS)))
        ax.set_xticklabels([VARIANT_LABELS[v] for v in FIXREGION_VARIANTS], fontsize=9)
        ax.set_ylabel("Ratio to Oracle SJF")
        ax.set_title(f"Rate = {rate:.0f} req/s", fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.suptitle("Remaining Gap to SJF: How Far Each Variant Still Is from Oracle", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "04_remaining_gap_to_sjf.pdf")


def plot_nlatency_comparison(all_data):
    """Plot 5: Mean nlatency bar chart — baselines + fixregion variants, 2x2 grid."""
    all_schedulers = BASELINES + FIXREGION_VARIANTS
    all_labels = {**BASELINE_LABELS, **VARIANT_LABELS}
    all_colors = {**COLORS_BASELINE, **COLORS_FIX}

    fig, axes = plt.subplots(len(DATASETS), len(RATES), figsize=(14, 9), sharey="row")

    for row, dataset in enumerate(DATASETS):
        data = all_data[dataset]
        for col, rate in enumerate(RATES):
            ax = axes[row, col]
            vals = []
            labels = []
            colors = []
            for sched in all_schedulers:
                val = data.get((sched, rate), np.nan)
                vals.append(val)
                labels.append(all_labels.get(sched, sched))
                colors.append(all_colors.get(sched, "#333333"))

            bars = ax.bar(np.arange(len(vals)), vals, color=colors, alpha=0.85)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

            ax.set_xticks(np.arange(len(vals)))
            ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
            ax.set_title(f"{DATASET_TITLES[dataset]} — rate={rate:.0f}", fontweight="bold", fontsize=10)
            if col == 0:
                ax.set_ylabel("Mean Normalized Latency")
            ax.grid(axis="y", linestyle=":", alpha=0.5)

    fig.suptitle("Mean Normalized Latency: Baselines vs Fix-Region Variants",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "05_nlatency_comparison.pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_data = load_all_data()

    print_and_save_summary(all_data)

    print("Generating plots...")
    plot_gap_recovery(all_data)
    plot_asymmetry(all_data)
    plot_marginal_recovery(all_data)
    plot_remaining_gap(all_data)
    plot_nlatency_comparison(all_data)

    print(f"\nAll outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()