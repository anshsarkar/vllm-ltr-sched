#!/usr/bin/env python3
"""Generate mini histogram PDFs from class_distributions.csv for paper table."""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "class_distributions.csv")
OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "benchmarks", "results", "plots_for_paper", "histograms"
)

# Dimensions
FIG_WIDTH = 2.0
FIG_HEIGHT = 0.55

# Color palette
COLORS = {
    "uniform": "#4A7CCC",       # steel blue
    "uniform_edge": "#3460A8",
    "pctl": "#2E8B6A",          # muted teal
    "pctl_edge": "#1F6B4F",
    "bg": "#F7F7F7",            # very light gray background
}

SCHEDULER_ORDER = [
    "tpt-class10-xxx",
    "tpt-class82-xxx",
    "tpt-width10-xxx",
    "tpt-pctl10-xxx",
    "tpt-pctl10-mse-xxx",
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 6,
})


def load_data(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def make_histogram(buckets, counts, output_path, is_pctl=False, sched_name=""):
    """Create a polished mini histogram PDF."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.set_facecolor(COLORS["bg"])
    fig.patch.set_facecolor("white")

    fill = COLORS["pctl"] if is_pctl else COLORS["uniform"]
    edge = COLORS["pctl_edge"] if is_pctl else COLORS["uniform_edge"]

    n = len(buckets)
    total = sum(counts)

    # Bin into visual groups for large bucket counts
    if n > 20:
        n_bins = 20
        bin_size = max(1, n // n_bins)
        vis_counts = []
        for i in range(0, n, bin_size):
            vis_counts.append(sum(counts[i:i + bin_size]))
        vis_counts = np.array(vis_counts, dtype=float)
    else:
        vis_counts = np.array(counts, dtype=float)

    n_vis = len(vis_counts)
    x = np.arange(n_vis)

    # Normalize to percentages for display
    pct = 100.0 * vis_counts / total if total > 0 else vis_counts

    # Draw bars with gradient-like effect (edge + fill)
    bars = ax.bar(x, pct, width=0.78, color=fill, edgecolor=edge,
                  linewidth=0.4, alpha=0.9, zorder=2)

    # Subtle horizontal gridline at 50%
    max_pct = pct.max()
    if max_pct > 20:
        grid_step = 25 if max_pct <= 60 else 50
        for g in range(grid_step, int(max_pct) + grid_step, grid_step):
            ax.axhline(g, color="#DDDDDD", linewidth=0.3, zorder=1)

    # Annotate dominant bucket if extremely skewed (>90% in one bucket)
    top_pct = pct.max()
    top_idx = int(pct.argmax())
    if top_pct > 90:
        ax.annotate(f"{top_pct:.0f}%", xy=(top_idx, top_pct),
                    xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=5, fontweight="bold",
                    color=edge,
                    path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    # Clean up axes
    ax.set_xlim(-0.6, n_vis - 0.4)
    ax.set_ylim(0, max(max_pct * 1.15, 1))

    # Minimal bottom spine only
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.4)
    ax.spines["bottom"].set_color("#AAAAAA")
    ax.tick_params(left=False, labelleft=False,
                   bottom=False, labelbottom=False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = load_data(INPUT_CSV)

    for dataset in ["lmsys", "sharegpt"]:
        print(f"\n{dataset.upper()}:")
        for sched in SCHEDULER_ORDER:
            subset = [r for r in rows
                      if r["dataset"] == dataset and r["scheduler"] == sched]
            if not subset:
                continue

            subset.sort(key=lambda r: int(r["bucket_id"]))
            buckets = [int(r["bucket_id"]) for r in subset]
            counts = [int(r["count"]) for r in subset]

            is_pctl = "pctl" in sched
            fname = f"hist_{dataset}_{sched.replace('-xxx', '')}.pdf"
            out_path = os.path.join(OUTPUT_DIR, fname)
            make_histogram(buckets, counts, out_path, is_pctl=is_pctl,
                           sched_name=sched)
            nonzero = sum(c > 0 for c in counts)
            print(f"  {fname}  ({nonzero}/{len(counts)} non-zero)")

    print(f"\nHistograms saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
