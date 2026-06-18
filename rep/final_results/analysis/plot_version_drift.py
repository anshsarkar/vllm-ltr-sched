#!/usr/bin/env python3
"""Plot version drift for key dependencies over time.

Reads version_drift_raw.csv and produces a heatmap-style table showing
resolved version strings at each snapshot date. Cells are shaded by
how many minor versions they've drifted from the baseline.
"""

import os
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "version_drift_raw.csv")
OUTPUT_DIR = os.path.join(
    os.path.dirname(SCRIPT_DIR), "benchmarks", "results", "plots_for_paper"
)

PACKAGES = ["torch", "transformers", "numpy", "flash-attn", "accelerate"]

DATE_LABELS = [
    "Apr 2024\n(v0.4.1 release)",
    "Jul 2024\n(dev window)",
    "Oct 2024\n(last commit)",
    "Jul 2025\n(+8 months)",
    "Mar 2026\n(today)",
]


def parse_version(v):
    """Extract major.minor.patch as a tuple of ints."""
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
    if m:
        return tuple(int(x) for x in m.groups())
    m = re.match(r"(\d+)\.(\d+)", v)
    if m:
        return tuple(int(x) for x in m.groups()) + (0,)
    return (0, 0, 0)


def short_version(v):
    """Strip post-release suffixes for display."""
    m = re.match(r"(\d+\.\d+\.\d+)", v)
    return m.group(1) if m else v


def minor_distance(v1, v2):
    """Monotonic distance: major*100 + minor, so crossing a major boundary
    always exceeds any within-major drift."""
    p1, p2 = parse_version(v1), parse_version(v2)
    return (p2[0] - p1[0]) * 100 + (p2[1] - p1[1])


def load_data():
    with open(INPUT_CSV) as f:
        rows = list(csv.DictReader(f))

    dates = sorted(set(r["date"] for r in rows))
    # Build: {package: {date: version_str}}
    data = {}
    for row in rows:
        pkg = row["package"]
        if pkg in PACKAGES:
            data.setdefault(pkg, {})[row["date"]] = row["version"]

    return dates, data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dates, data = load_data()

    n_pkg = len(PACKAGES)
    n_dates = len(dates)

    # Build version text grid and drift intensity grid
    text_grid = []
    drift_grid = []
    for pkg in PACKAGES:
        row_text = []
        row_drift = []
        baseline = data[pkg][dates[0]]
        for d in dates:
            v = data[pkg].get(d, "")
            row_text.append(short_version(v))
            row_drift.append(minor_distance(baseline, v))
        text_grid.append(row_text)
        drift_grid.append(row_drift)

    drift_arr = np.array(drift_grid, dtype=float)
    # Global normalization so colors are comparable across packages
    global_max = drift_arr.max()
    if global_max == 0:
        global_max = 1
    norm_drift = drift_arr / global_max

    # --- v2 params (revert to these if needed) ---
    # Per-row normalization:
    #   max_per_row = drift_arr.max(axis=1, keepdims=True)
    #   max_per_row[max_per_row == 0] = 1
    #   norm_drift = drift_arr / max_per_row
    # figsize=(6.5, 2.8), cell fontsize=11, xtick fontsize=10,
    # ytick fontsize=11, white threshold=0.55

    fig, ax = plt.subplots(figsize=(6.5, 2.4))

    cmap = plt.cm.YlOrRd
    ax.imshow(norm_drift, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    for i in range(n_pkg):
        for j in range(n_dates):
            color = "white" if norm_drift[i, j] > 0.55 else "black"
            fontweight = "bold" if j == 0 else "normal"
            ax.text(j, i, text_grid[i][j], ha="center", va="center",
                    fontsize=9, color=color, fontweight=fontweight)

    ax.set_xticks(range(n_dates))
    ax.set_xticklabels(DATE_LABELS, fontsize=7.5, ha="center")
    ax.set_yticks(range(n_pkg))
    ax.set_yticklabels(PACKAGES, fontsize=9, fontfamily="monospace")

    ax.tick_params(length=0)
    ax.set_xlim(-0.5, n_dates - 0.5)
    ax.set_ylim(n_pkg - 0.5, -0.5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "version_drift_v3.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
