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
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "class_distributions.csv")
OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "benchmarks", "results", "plots_for_paper", "histograms"
)

# Figure dimensions (compact for inline table use)
FIG_WIDTH = 2.2
FIG_HEIGHT = 0.6

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


def make_histogram(buckets, counts, output_path, is_pctl=False, sched_name=""):
    """Create a polished mini histogram PDF for inline table embedding."""
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    fig.patch.set_alpha(0)          # transparent figure background
    ax.set_facecolor("white")

    # Pick colors
    face = PCTL_FACE if is_pctl else UNIFORM_FACE
    face_light = PCTL_FACE_LIGHT if is_pctl else UNIFORM_FACE_LIGHT
    edge = PCTL_EDGE if is_pctl else UNIFORM_EDGE

    n = len(buckets)
    total = sum(counts)

    # Bin into visual groups for large bucket counts
    if n > 25:
        n_bins = 25
        bin_size = max(1, n // n_bins)
        vis_counts = []
        for i in range(0, n, bin_size):
            vis_counts.append(sum(counts[i:i + bin_size]))
        vis_counts = np.array(vis_counts, dtype=float)
    else:
        vis_counts = np.array(counts, dtype=float)

    n_vis = len(vis_counts)
    x = np.arange(n_vis)

    # Normalize to percentages
    pct = 100.0 * vis_counts / total if total > 0 else vis_counts
    max_pct = pct.max()

    # Create gradient effect: darker bars for higher values
    norm_vals = pct / max_pct if max_pct > 0 else pct
    from matplotlib.colors import to_rgba
    face_rgba = np.array(to_rgba(face))
    light_rgba = np.array(to_rgba(face_light))

    bar_width = 0.82 if n_vis <= 15 else 0.90
    bars = ax.bar(x, pct, width=bar_width, color=face, edgecolor=edge,
                  linewidth=0.35, zorder=3)

    # Color each bar by intensity
    for bar, nv in zip(bars, norm_vals):
        color = light_rgba + (face_rgba - light_rgba) * nv
        bar.set_facecolor(color)

    # Subtle horizontal reference lines
    if max_pct > 15:
        if max_pct > 70:
            gridlines = [25, 50, 75]
        elif max_pct > 35:
            gridlines = [10, 20, 30]
        else:
            gridlines = [5, 10]
        for g in gridlines:
            if g < max_pct * 0.95:
                ax.axhline(g, color="#D0D0D0", linewidth=0.25, zorder=1, linestyle="-")

    # Annotate dominant bucket if extremely skewed (>85%)
    top_pct = pct.max()
    top_idx = int(pct.argmax())
    if top_pct > 85:
        ax.annotate(f"{top_pct:.0f}%",
                    xy=(top_idx, top_pct),
                    xytext=(0, 1.5), textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=5, fontweight="bold", color=edge,
                    path_effects=[pe.withStroke(linewidth=1.2, foreground="white")])

    # Axis styling
    ax.set_xlim(-0.6, n_vis - 0.4)
    y_top = max_pct * 1.18 if top_pct > 85 else max_pct * 1.08
    ax.set_ylim(0, max(y_top, 0.5))

    # Only a thin bottom spine
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.35)
    ax.spines["bottom"].set_color("#999999")

    # No tick marks or labels (these are inline mini charts)
    ax.tick_params(left=False, labelleft=False,
                   bottom=False, labelbottom=False)

    # Add tiny "Short" / "Long" labels at ends for context
    # Bucket 0 = longest output, last bucket = shortest output
    if n_vis >= 5:
        ax.text(-0.3, -max_pct * 0.06, "Long", fontsize=3.5, color="#888888",
                ha="left", va="top", style="italic")
        ax.text(n_vis - 0.7, -max_pct * 0.06, "Short", fontsize=3.5, color="#888888",
                ha="right", va="top", style="italic")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.14)
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
