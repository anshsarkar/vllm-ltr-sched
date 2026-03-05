#!/usr/bin/env python3
"""Idea 2 Post-hoc: Prove prompt length independently affects latency
but LTR-Ranking ignores it.

Thesis: Two requests with the same output length but different prompt lengths
will have different latencies because prompt length adds prefill cost.
LTR-Ranking sorts only by predicted output length, so it benefits short-prompt
requests more than long-prompt ones.

Data source: opt-xxx and fcfs .pt files from existing benchmark results.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATASETS, OUT_BASE,
    load_scheduler_data, make_subplots, hide_unused, savefig,
)

OUT_DIR = os.path.join(OUT_BASE, "idea2")

# Number of prompt-length quartiles
N_QUARTILES = 4
QUARTILE_LABELS = ["Q1 (short)", "Q2", "Q3", "Q4 (long)"]
QUARTILE_COLORS = ["#4CAF50", "#8BC34A", "#FF9800", "#E53935"]


def build_dataframe(fcfs_data, ltr_data, rate):
    """Build a DataFrame with matched requests from FCFS and LTR-Ranking."""
    fc = fcfs_data[rate]
    lr = ltr_data[rate]

    n = min(len(fc["input_lens"]), len(lr["input_lens"]))
    df = pd.DataFrame({
        "input_len": lr["input_lens"][:n],
        "output_len": lr["output_lens"][:n],
        "nlat_ltr": lr["nlatencies"][:n],
        "nlat_fcfs": fc["nlatencies"][:n],
    })

    # Assign prompt-length quartile
    df["prompt_q"] = pd.qcut(df["input_len"], N_QUARTILES, labels=False, duplicates="drop")
    # Assign output-length quartile (for partial-effect analysis)
    df["output_q"] = pd.qcut(df["output_len"], N_QUARTILES, labels=False, duplicates="drop")
    return df


def plot_latency_by_prompt_quartile(all_dfs, out_dir, dataset_label):
    """Plot 1: Grouped bars — mean nLatency by prompt quartile, FCFS vs LTR."""
    rates = sorted(all_dfs.keys())
    fig, axes = make_subplots(len(rates))

    bar_w = 0.35
    for ax, rate in zip(axes, rates):
        df = all_dfs[rate]
        n_q = df["prompt_q"].nunique()
        x = np.arange(n_q)

        means_fcfs = df.groupby("prompt_q")["nlat_fcfs"].mean().values * 1000
        means_ltr = df.groupby("prompt_q")["nlat_ltr"].mean().values * 1000

        ax.bar(x - bar_w / 2, means_fcfs, bar_w, label="FCFS", color="#2196F3", alpha=0.85)
        ax.bar(x + bar_w / 2, means_ltr, bar_w, label="LTR-Ranking", color="#FF5722", alpha=0.85)

        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(QUARTILE_LABELS[:n_q], fontsize=8)
        ax.set_ylabel("Mean nLatency (ms/tok)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — Mean nLatency by Prompt-Length Quartile\n"
        "(Q1 = shortest prompts, Q4 = longest prompts)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_latency_by_prompt_quartile.png"))


def plot_improvement_ratio(all_dfs, out_dir, dataset_label):
    """Plot 2: Line plot — LTR improvement ratio per prompt quartile vs QPS."""
    rates = sorted(all_dfs.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    # Collect per-quartile improvement across rates
    n_q = all_dfs[rates[0]]["prompt_q"].nunique()
    for q in range(n_q):
        ratios = []
        for rate in rates:
            df = all_dfs[rate]
            grp = df[df["prompt_q"] == q]
            fcfs_mean = grp["nlat_fcfs"].mean()
            ltr_mean = grp["nlat_ltr"].mean()
            ratio = (fcfs_mean - ltr_mean) / fcfs_mean if fcfs_mean > 0 else 0
            ratios.append(ratio)
        ax.plot(rates, ratios, marker="o", label=QUARTILE_LABELS[q],
                color=QUARTILE_COLORS[q], linewidth=2)

    ax.set_xlabel("Request Rate (req/s)", fontsize=11)
    ax.set_ylabel("LTR Improvement Ratio\n(FCFS - LTR) / FCFS", fontsize=11)
    ax.set_title(
        f"{dataset_label} — LTR-Ranking Improvement by Prompt-Length Quartile\n"
        "(Higher = LTR helps more; expect Q1 > Q4)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_improvement_ratio.png"))


def plot_prompt_vs_latency_within_output_bins(all_dfs, out_dir, dataset_label):
    """Plot 3: Within output-length quartiles, trend of prompt_len vs nLatency."""
    # Use the highest non-trivial rate for clearest signal
    rates = sorted(all_dfs.keys())
    # Pick a high-load rate (second highest or highest)
    rate = rates[-2] if len(rates) > 1 else rates[0]
    df = all_dfs[rate]

    n_oq = df["output_q"].nunique()
    fig, axes = make_subplots(n_oq, max_cols=4, cell_w=5, cell_h=4)

    oq_labels = [f"Output Q{q+1}" for q in range(n_oq)]

    for ax, oq in zip(axes, range(n_oq)):
        grp = df[df["output_q"] == oq]
        x = grp["input_len"].values
        y = grp["nlat_ltr"].values * 1000

        ax.scatter(x, y, alpha=0.15, s=6, color="#666666")

        # Trend line
        if len(x) > 20:
            order = np.argsort(x)
            xs, ys = x[order], y[order]
            edges = np.linspace(xs.min(), xs.max(), 16)
            bx, by = [], []
            for i in range(len(edges) - 1):
                mask = (xs >= edges[i]) & (xs < edges[i + 1])
                if mask.sum() > 0:
                    bx.append((edges[i] + edges[i + 1]) / 2)
                    by.append(np.mean(ys[mask]))
            if bx:
                ax.plot(bx, by, color="#E53935", linewidth=2.5, zorder=5, label="Trend")

        ax.set_title(f"{oq_labels[oq]} (n={len(grp)})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Prompt Length (tokens)", fontsize=9)
        ax.set_ylabel("nLatency (ms/tok)", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    hide_unused(axes, n_oq)
    fig.suptitle(
        f"{dataset_label} — Prompt Length vs nLatency within Output-Length Quartiles\n"
        f"(LTR-Ranking, {rate} req/s — positive slope proves prompt length is independent predictor)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_prompt_vs_latency_within_output_bins.png"))


def plot_heatmap(all_dfs, out_dir, dataset_label):
    """Plot 4: 2D heatmap — prompt_len vs output_len, color = mean nLatency."""
    rates = sorted(all_dfs.keys())
    rate = rates[-2] if len(rates) > 1 else rates[0]
    df = all_dfs[rate]

    n_bins = 8
    df = df.copy()
    df["pbin"] = pd.cut(df["input_len"], bins=n_bins, labels=False)
    df["obin"] = pd.cut(df["output_len"], bins=n_bins, labels=False)

    pivot = df.groupby(["obin", "pbin"])["nlat_ltr"].mean().unstack() * 1000

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="YlOrRd")
    ax.set_xlabel("Prompt Length Bin (low → high)", fontsize=11)
    ax.set_ylabel("Output Length Bin (low → high)", fontsize=11)
    ax.set_title(
        f"{dataset_label} — Mean nLatency (ms/tok) Heatmap\n"
        f"(LTR-Ranking, {rate} req/s)",
        fontsize=12, fontweight="bold",
    )
    fig.colorbar(im, ax=ax, label="Mean nLatency (ms/tok)")
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_heatmap.png"))


def print_summary(all_dfs, dataset_label):
    print(f"\n{'=' * 90}")
    print(f"  IDEA 2: PROMPT LENGTH ANALYSIS — {dataset_label}")
    print(f"{'=' * 90}")

    fmt = "{:<8} {:<14} {:>8} {:>16} {:>16} {:>14}"
    print(fmt.format("QPS", "Prompt Q", "Count", "FCFS nLat(ms)", "LTR nLat(ms)", "Improvement"))
    print("-" * 90)

    for rate in sorted(all_dfs.keys()):
        df = all_dfs[rate]
        n_q = df["prompt_q"].nunique()
        for q in range(n_q):
            grp = df[df["prompt_q"] == q]
            fcfs_mean = grp["nlat_fcfs"].mean() * 1000
            ltr_mean = grp["nlat_ltr"].mean() * 1000
            improvement = (fcfs_mean - ltr_mean) / fcfs_mean if fcfs_mean > 0 else 0

            print(fmt.format(
                f"{rate}",
                QUARTILE_LABELS[q],
                f"{len(grp)}",
                f"{fcfs_mean:.1f}",
                f"{ltr_mean:.1f}",
                f"{improvement:.1%}",
            ))
        print()

    print("=" * 90)


def analyze_dataset(ds_info, out_dir):
    label = ds_info["label"]
    pt_dir = ds_info["pt_dir"]

    print(f"\n--- Analyzing {label} from {pt_dir} ---")

    fcfs_data = load_scheduler_data(pt_dir, "fcfs")
    ltr_data = load_scheduler_data(pt_dir, "opt-xxx")

    if not fcfs_data or not ltr_data:
        print(f"  Missing fcfs or opt-xxx data in {pt_dir}, skipping")
        return

    # Only use rates present in both
    common_rates = sorted(set(fcfs_data.keys()) & set(ltr_data.keys()))
    if not common_rates:
        print(f"  No common rates between fcfs and opt-xxx, skipping")
        return

    all_dfs = {}
    for rate in common_rates:
        df = build_dataframe(fcfs_data, ltr_data, rate)
        all_dfs[rate] = df
        print(f"  Rate {rate}: {len(df)} requests, "
              f"prompt range [{df['input_len'].min()}, {df['input_len'].max()}], "
              f"output range [{df['output_len'].min()}, {df['output_len'].max()}]")

    if not all_dfs:
        return

    plot_latency_by_prompt_quartile(all_dfs, out_dir, label)
    plot_improvement_ratio(all_dfs, out_dir, label)
    plot_prompt_vs_latency_within_output_bins(all_dfs, out_dir, label)
    plot_heatmap(all_dfs, out_dir, label)
    print_summary(all_dfs, label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Idea 2: Proving prompt length independently affects latency but LTR-Ranking ignores it")

    for _, ds_info in DATASETS.items():
        analyze_dataset(ds_info, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
