#!/usr/bin/env python3

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

# Allow running from project root
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATASETS, OUT_BASE, SCHED_COLORS,
    load_scheduler_data, make_subplots, hide_unused, add_trend, savefig,
)

OUT_DIR = os.path.join(OUT_BASE, "idea1")

# Displacement threshold: fraction of N to consider "significant"
DISP_THRESHOLD_FRAC = 0.10  # top/bottom 10% rank displacement


def compute_rank_displacement(aux_scores, output_lens):
    # Filter out None scores
    valid = np.array([s is not None for s in aux_scores])
    if valid.sum() == 0:
        return None, None, None

    scores = np.array([float(s) for s in aux_scores])[valid]
    olens = np.array(output_lens)[valid]
    valid_indices = np.where(valid)[0]

    # Predicted rank: higher score = predicted shorter = rank 1
    predicted_rank = rankdata(-scores, method="average")
    # True rank: shorter output = rank 1
    true_rank = rankdata(olens, method="average")

    displacement = predicted_rank - true_rank
    return displacement, valid_indices, olens


def classify_errors(displacement, n, threshold_frac=DISP_THRESHOLD_FRAC):
    threshold = n * threshold_frac
    return {
        "long_as_short": displacement < -threshold,  # ranked too early, actually long
        "short_as_long": displacement > threshold,    # ranked too late, actually short
        "correct": np.abs(displacement) <= threshold,
    }


def plot_latency_by_error_direction(all_data, out_dir, dataset_label):
    rates = sorted(all_data.keys())
    fig, axes = make_subplots(len(rates))

    colors = {"long_as_short": "#E53935", "correct": "#757575", "short_as_long": "#1E88E5"}
    labels = {"long_as_short": "Long-as-Short", "correct": "Correct", "short_as_long": "Short-as-Long"}
    cats = ["long_as_short", "correct", "short_as_long"]

    for ax, rate in zip(axes, rates):
        info = all_data[rate]
        x = np.arange(len(cats))
        means = []
        counts = []
        for cat in cats:
            mask = info["categories"][cat]
            nlats = info["nlatencies"][mask]
            means.append(np.mean(nlats) * 1000 if len(nlats) > 0 else 0)
            counts.append(mask.sum())

        bars = ax.bar(x, means, color=[colors[c] for c in cats], alpha=0.85)
        for bar, v, cnt in zip(bars, means, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"n={cnt}\n{v:.1f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[c] for c in cats], fontsize=9)
        ax.set_ylabel("Mean nLatency (ms/tok)", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — Mean nLatency by Prediction Error Direction\n"
        "(Long-as-Short = long request predicted short; Short-as-Long = short predicted long)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_latency_by_error_direction.png"))


def plot_displacement_vs_latency(all_data, out_dir, dataset_label):
    rates = sorted(all_data.keys())
    fig, axes = make_subplots(len(rates))

    for ax, rate in zip(axes, rates):
        info = all_data[rate]
        disp = info["displacement"]
        nlat = info["nlatencies"] * 1000

        ax.scatter(disp, nlat, alpha=0.15, s=6, color="#666666")
        add_trend(ax, disp, nlat, color="#E53935", bins=25, label="Trend")

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xlabel("Rank Displacement (− = long-as-short, + = short-as-long)", fontsize=9)
        ax.set_ylabel("nLatency (ms/tok)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — Rank Displacement vs Normalized Latency\n"
        "(negative displacement = long request ranked as short → should cause higher latency)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_displacement_vs_latency.png"))


def plot_excess_latency_contribution(all_data, out_dir, dataset_label):
    rates = sorted(all_data.keys())
    fig, axes = make_subplots(len(rates))

    colors = {"long_as_short": "#E53935", "correct": "#757575", "short_as_long": "#1E88E5"}
    labels = {"long_as_short": "Long-as-Short", "correct": "Correct", "short_as_long": "Short-as-Long"}
    cats = ["long_as_short", "correct", "short_as_long"]

    for ax, rate in zip(axes, rates):
        info = all_data[rate]
        nlats = info["nlatencies"]
        median_nlat = np.median(nlats)

        # Excess latency = max(0, nlat - median) for each request
        excess = np.maximum(0, nlats - median_nlat)
        total_excess = excess.sum()

        fracs = []
        for cat in cats:
            mask = info["categories"][cat]
            cat_excess = excess[mask].sum()
            fracs.append(cat_excess / total_excess if total_excess > 0 else 0)

        bars = ax.bar(range(len(cats)), fracs, color=[colors[c] for c in cats], alpha=0.85)
        for bar, v in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=8)

        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels([labels[c] for c in cats], fontsize=9)
        ax.set_ylabel("Fraction of Excess Latency", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — Share of Total Excess Latency by Error Direction\n"
        "(excess = per-request latency above median; higher = more damage from that error type)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_excess_latency_contribution.png"))


def print_summary(all_data, dataset_label):
    print(f"\n{'=' * 80}")
    print(f"  IDEA 1: ASYMMETRIC ERROR ANALYSIS — {dataset_label}")
    print(f"{'=' * 80}")

    cats = ["long_as_short", "correct", "short_as_long"]
    cat_labels = {"long_as_short": "Long-as-Short", "correct": "Correct", "short_as_long": "Short-as-Long"}

    fmt = "{:<8} {:<16} {:>8} {:>14} {:>14} {:>14}"
    print(fmt.format("QPS", "Error Type", "Count", "Mean nLat(ms)", "P99 nLat(ms)", "Excess %"))
    print("-" * 80)

    for rate in sorted(all_data.keys()):
        info = all_data[rate]
        nlats = info["nlatencies"]
        median_nlat = np.median(nlats)
        excess = np.maximum(0, nlats - median_nlat)
        total_excess = excess.sum()

        for cat in cats:
            mask = info["categories"][cat]
            cat_nlats = nlats[mask]
            cat_excess = excess[mask].sum()

            print(fmt.format(
                f"{rate}",
                cat_labels[cat],
                f"{mask.sum()}",
                f"{np.mean(cat_nlats) * 1000:.1f}" if len(cat_nlats) > 0 else "—",
                f"{np.percentile(cat_nlats, 99) * 1000:.1f}" if len(cat_nlats) > 0 else "—",
                f"{cat_excess / total_excess:.1%}" if total_excess > 0 else "—",
            ))
        print()

    print("=" * 80)


def analyze_dataset(ds_info, out_dir):
    label = ds_info["label"]
    pt_dir = ds_info["pt_dir"]

    print(f"\n--- Analyzing {label} (opt-xxx) from {pt_dir} ---")

    rate_data = load_scheduler_data(pt_dir, "opt-xxx")
    if not rate_data:
        print(f"  No opt-xxx data found in {pt_dir}, skipping")
        return

    all_data = {}
    for rate in sorted(rate_data.keys()):
        d = rate_data[rate]
        displacement, valid_idx, olens = compute_rank_displacement(
            d["aux_model_scores"], d["output_lens"]
        )
        if displacement is None:
            print(f"  Rate {rate}: no valid aux_model_scores, skipping")
            continue

        n = len(displacement)
        categories = classify_errors(displacement, n)

        # nlatencies for valid requests only
        nlats = d["nlatencies"][valid_idx]

        all_data[rate] = {
            "displacement": displacement,
            "nlatencies": nlats,
            "output_lens": olens,
            "categories": categories,
            "n": n,
        }
        print(f"  Rate {rate}: {n} requests, "
              f"LAS={categories['long_as_short'].sum()}, "
              f"correct={categories['correct'].sum()}, "
              f"SAL={categories['short_as_long'].sum()}")

    if not all_data:
        return

    plot_latency_by_error_direction(all_data, out_dir, label)
    plot_displacement_vs_latency(all_data, out_dir, label)
    plot_excess_latency_contribution(all_data, out_dir, label)
    print_summary(all_data, label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Idea 1: Proving asymmetric prediction errors cause asymmetric latency damage")

    for _, ds_info in DATASETS.items():
        analyze_dataset(ds_info, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
