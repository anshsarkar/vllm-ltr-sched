#!/usr/bin/env python3
"""Generate polished mini histogram PDFs from class_distributions.csv for paper table.

Each histogram is a compact, inline-ready figure showing the class distribution
for a given predictor and dataset. Designed to be embedded in LaTeX tables via
\\includegraphics[height=8mm]{...}.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "class_distributions.csv")
OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "benchmarks", "results", "plots_for_paper", "histograms"
)

# Figure dimensions (compact for inline table use)
FIG_WIDTH = 1.6
FIG_HEIGHT = 0.4

# ── Color palette ─────────────────────────────────────────────────────────
# Uniform buckets: warm orange gradient
UNIFORM_FACE = "#E8793A"
UNIFORM_FACE_LIGHT = "#F5A66A"
UNIFORM_EDGE = "#C25A1C"

# Percentile buckets: cool teal
PCTL_FACE = "#2AA198"
PCTL_FACE_LIGHT = "#5DC9C0"
PCTL_EDGE = "#1A756E"

SCHEDULER_ORDER = [
    "tpt-class10-xxx",
    "tpt-class82-xxx",
    "tpt-width10-xxx",
    "tpt-pctl10-xxx",
    "tpt-pctl10-mse-xxx",
]

# Use a clean sans-serif font
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 5.5,
    "text.antialiased": True,
})


def load_data(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def make_histogram(counts, output_path, is_pctl=False):
    """Create a polished mini histogram PDF for inline table embedding."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_alpha(0)          # transparent figure background
    ax.set_facecolor("white")

    total = sum(counts)

    vis_counts = np.array(counts, dtype=float)

    # Aggregate into visible bins if too many bars (e.g. 820-class)
    MAX_BARS = 80
    if len(vis_counts) > MAX_BARS:
        n_orig = len(vis_counts)
        bin_size = int(np.ceil(n_orig / MAX_BARS))
        padded = np.pad(vis_counts, (0, bin_size * int(np.ceil(n_orig / bin_size)) - n_orig))
        vis_counts = padded.reshape(-1, bin_size).sum(axis=1)

    n_vis = len(vis_counts)
    x = np.arange(n_vis)

    # Normalize to percentages
    pct = 100.0 * vis_counts / total if total > 0 else vis_counts
    max_pct = pct.max()

    # Black bars, tight spacing, no gaps
    bar_width = 0.95
    ax.bar(x, pct, width=bar_width, color="black", edgecolor="none",
           linewidth=0, zorder=3)

    # Axis styling
    ax.set_xlim(-0.6, n_vis - 0.4)
    y_top = max_pct * 1.08
    ax.set_ylim(0, max(y_top, 0.5))

    # Only a thin bottom spine
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.35)
    ax.spines["bottom"].set_color("#999999")

    # No tick marks or labels (these are inline mini charts)
    ax.tick_params(left=False, labelleft=False,
                   bottom=False, labelbottom=False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02,
                dpi=300, transparent=True)
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
            counts = [int(r["count"]) for r in subset]

            is_pctl = "pctl" in sched
            fname = f"hist_{dataset}_{sched.replace('-xxx', '')}.pdf"
            out_path = os.path.join(OUTPUT_DIR, fname)
            make_histogram(counts, out_path, is_pctl=is_pctl)
            nonzero = sum(c > 0 for c in counts)
            print(f"  {fname}  ({nonzero}/{len(counts)} non-zero)")

    print(f"\nHistograms saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
