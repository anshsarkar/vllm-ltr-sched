#!/usr/bin/env python3
"""Idea 3 Post-hoc: Prove prediction error grows with output length;
longest requests are most mispredicted and cause disproportionate tail latency.

Thesis: The LTR-Ranking aux model makes a one-shot prediction at admission.
For longer requests, the prediction is more likely to be wrong because the
model has less signal about how long the generation will actually run.
This means static ordering (SJF) leaves performance on the table — a dynamic
approach (SRTF) that re-evaluates remaining time would help most for the
longest requests.

Data source: opt-xxx .pt files from existing benchmark results.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    DATASETS, OUT_BASE,
    load_scheduler_data, make_subplots, hide_unused, add_trend, savefig,
)

OUT_DIR = os.path.join(OUT_BASE, "idea3")

# Output length buckets
BUCKET_EDGES = [0, 64, 256, 1024, float("inf")]
BUCKET_LABELS = ["1-64", "65-256", "257-1024", "1025+"]
BUCKET_COLORS = ["#4CAF50", "#8BC34A", "#FF9800", "#E53935"]


def assign_buckets(output_lens):
    """Assign each request to an output-length bucket."""
    buckets = np.zeros(len(output_lens), dtype=int)
    for i, (lo, hi) in enumerate(zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:])):
        mask = (output_lens > lo) & (output_lens <= hi)
        buckets[mask] = i
    return buckets


def compute_per_bucket_tau(aux_scores, output_lens, buckets):
    """Compute Kendall tau between aux_model_scores and output_lens per bucket."""
    results = {}
    for b in range(len(BUCKET_LABELS)):
        mask = buckets == b
        if mask.sum() < 10:
            results[b] = {"tau": None, "n": mask.sum()}
            continue

        scores_b = np.array([float(s) for s in np.array(aux_scores)[mask]])
        olens_b = output_lens[mask]

        # Higher score = predicted shorter, so negate for correlation with output_len
        tau, p = kendalltau(-scores_b, olens_b)
        results[b] = {"tau": tau, "p": p, "n": int(mask.sum())}

    return results


def compute_rank_displacement(aux_scores, output_lens):
    """Compute displacement = predicted_rank - true_rank for each request."""
    valid = np.array([s is not None for s in aux_scores])
    if valid.sum() == 0:
        return None, None

    scores = np.array([float(s) for s in aux_scores])[valid]
    olens = output_lens[valid]

    predicted_rank = rankdata(-scores, method="average")
    true_rank = rankdata(olens, method="average")

    displacement = predicted_rank - true_rank
    return displacement, olens


def plot_tau_by_bucket(all_data, out_dir, dataset_label):
    """Plot 1: Bar chart — Kendall tau by output length bucket per QPS."""
    rates = sorted(all_data.keys())
    fig, axes = make_subplots(len(rates))

    for ax, rate in zip(axes, rates):
        info = all_data[rate]
        taus = []
        counts = []
        for b in range(len(BUCKET_LABELS)):
            t = info["bucket_taus"][b]
            taus.append(t["tau"] if t["tau"] is not None else 0)
            counts.append(t["n"])

        bars = ax.bar(range(len(BUCKET_LABELS)), taus,
                      color=BUCKET_COLORS, alpha=0.85)
        for bar, v, n in zip(bars, taus, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.01,
                    f"n={n}\nτ={v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(BUCKET_LABELS)))
        ax.set_xticklabels(BUCKET_LABELS, fontsize=9)
        ax.set_ylabel("Kendall τ", fontsize=10)
        ax.set_ylim(-0.1, 1.0)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — Prediction Accuracy (Kendall τ) by Output Length Bucket\n"
        "(Lower τ = worse prediction; expect degradation for longer requests)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_tau_by_bucket.png"))


def plot_displacement_distribution(all_data, out_dir, dataset_label):
    """Plot 2: Box plots — rank displacement distribution by output bucket."""
    rates = sorted(all_data.keys())
    # Pick a high-load rate for clearest signal
    rate = rates[-2] if len(rates) > 1 else rates[0]
    info = all_data[rate]

    fig, ax = plt.subplots(figsize=(10, 6))

    disp = info["displacement"]
    olens = info["displacement_olens"]
    buckets = assign_buckets(olens)

    box_data = []
    labels = []
    for b in range(len(BUCKET_LABELS)):
        mask = buckets == b
        if mask.sum() > 0:
            box_data.append(disp[mask])
            labels.append(f"{BUCKET_LABELS[b]}\n(n={mask.sum()})")

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True,
                    showfliers=False, whis=[5, 95])
    for patch, color in zip(bp["boxes"], BUCKET_COLORS[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Output Length Bucket (tokens)", fontsize=11)
    ax.set_ylabel("Rank Displacement (predicted - true)", fontsize=11)
    ax.set_title(
        f"{dataset_label} — Rank Displacement Distribution by Output Length\n"
        f"(LTR-Ranking, {rate} req/s — wider spread = less predictable)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_displacement_distribution.png"))


def plot_output_len_vs_displacement(all_data, out_dir, dataset_label):
    """Plot 3: Scatter — actual output len vs rank displacement."""
    rates = sorted(all_data.keys())
    rate = rates[-2] if len(rates) > 1 else rates[0]
    info = all_data[rate]

    fig, ax = plt.subplots(figsize=(10, 6))

    disp = info["displacement"]
    olens = info["displacement_olens"]

    ax.scatter(olens, np.abs(disp), alpha=0.15, s=6, color="#666666")
    add_trend(ax, olens, np.abs(disp), color="#E53935", bins=25, label="Trend (|displacement|)")

    ax.set_xlabel("Actual Output Length (tokens)", fontsize=11)
    ax.set_ylabel("|Rank Displacement|", fontsize=11)
    ax.set_title(
        f"{dataset_label} — Prediction Error vs Output Length\n"
        f"(LTR-Ranking, {rate} req/s — upward trend proves longer = harder to predict)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_output_len_vs_displacement.png"))


def plot_tail_latency_cdf(all_data, out_dir, dataset_label):
    """Plot 4: CDF — nLatency for top-10% longest requests vs rest."""
    rates = sorted(all_data.keys())
    fig, axes = make_subplots(len(rates))

    for ax, rate in zip(axes, rates):
        info = all_data[rate]
        olens = info["output_lens"]
        nlats = info["nlatencies"]

        p90 = np.percentile(olens, 90)
        long_mask = olens >= p90
        short_mask = ~long_mask

        for mask, label, color in [
            (short_mask, f"Bottom 90% (n={short_mask.sum()})", "#2196F3"),
            (long_mask, f"Top 10% longest (n={long_mask.sum()})", "#E53935"),
        ]:
            if mask.sum() == 0:
                continue
            vals = np.sort(nlats[mask]) * 1000
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=label, color=color, linewidth=2)

        ax.set_title(f"{rate} req/s", fontsize=11, fontweight="bold")
        ax.set_xlabel("nLatency (ms/tok)", fontsize=9)
        ax.set_ylabel("CDF", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    hide_unused(axes, len(rates))
    fig.suptitle(
        f"{dataset_label} — nLatency CDF: Top-10% Longest vs Rest\n"
        "(Longest requests have worst tail latency under LTR-Ranking)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    savefig(fig, os.path.join(out_dir, f"{dataset_label.lower()}_tail_latency_cdf.png"))


def print_summary(all_data, dataset_label):
    print(f"\n{'=' * 90}")
    print(f"  IDEA 3: PREDICTION STALENESS ANALYSIS — {dataset_label}")
    print(f"{'=' * 90}")

    fmt = "{:<8} {:<12} {:>8} {:>12} {:>14} {:>16} {:>14}"
    print(fmt.format("QPS", "Bucket", "Count", "Kendall τ",
                     "Mean |Disp|", "Mean nLat(ms)", "P99 nLat(ms)"))
    print("-" * 90)

    for rate in sorted(all_data.keys()):
        info = all_data[rate]
        olens = info["output_lens"]
        nlats = info["nlatencies"]
        buckets = assign_buckets(olens)

        disp = info["displacement"]
        disp_olens = info["displacement_olens"]
        disp_buckets = assign_buckets(disp_olens)

        for b in range(len(BUCKET_LABELS)):
            # Tau from bucket analysis
            t = info["bucket_taus"][b]
            tau_str = f"{t['tau']:.3f}" if t["tau"] is not None else "—"

            # Mean |displacement| for this bucket
            disp_mask = disp_buckets == b
            mean_disp = np.mean(np.abs(disp[disp_mask])) if disp_mask.sum() > 0 else 0

            # nLatency for this bucket
            nlat_mask = buckets == b
            nlat_b = nlats[nlat_mask]

            print(fmt.format(
                f"{rate}",
                BUCKET_LABELS[b],
                f"{t['n']}",
                tau_str,
                f"{mean_disp:.1f}",
                f"{np.mean(nlat_b) * 1000:.1f}" if len(nlat_b) > 0 else "—",
                f"{np.percentile(nlat_b, 99) * 1000:.1f}" if len(nlat_b) > 0 else "—",
            ))
        print()

    print("=" * 90)


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

        # Bucket-level Kendall tau
        buckets = assign_buckets(d["output_lens"])
        bucket_taus = compute_per_bucket_tau(
            d["aux_model_scores"], d["output_lens"], buckets
        )

        # Rank displacement
        displacement, disp_olens = compute_rank_displacement(
            d["aux_model_scores"], d["output_lens"]
        )

        all_data[rate] = {
            "bucket_taus": bucket_taus,
            "displacement": displacement,
            "displacement_olens": disp_olens,
            "output_lens": d["output_lens"],
            "nlatencies": d["nlatencies"],
        }

        tau_strs = [f"{bucket_taus[b]['tau']:.3f}" if bucket_taus[b]["tau"] is not None else "—"
                    for b in range(len(BUCKET_LABELS))]
        print(f"  Rate {rate}: τ per bucket = {tau_strs}")

    if not all_data:
        return

    plot_tau_by_bucket(all_data, out_dir, label)
    plot_displacement_distribution(all_data, out_dir, label)
    plot_output_len_vs_displacement(all_data, out_dir, label)
    plot_tail_latency_cdf(all_data, out_dir, label)
    print_summary(all_data, label)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Idea 3: Proving prediction error grows with output length — static ordering is suboptimal")

    for _, ds_info in DATASETS.items():
        analyze_dataset(ds_info, OUT_DIR)

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
