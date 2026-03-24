#!/usr/bin/env python3
"""Generate paper-ready 3-subplot comparison figures for LMSYS and ShareGPT.

Layout:
  (a) Authors' original results (dashed)
  (b) Our reproduction with SE bands (mean ± standard error across 3 runs)
  (c) All classification variants (merged) — FCFS + Ranking + all classifiers
Single shared horizontal legend at the bottom.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "plots_for_paper")

# ---------- Style config ----------

# Authors' scheduler colors & markers (match original paper)
AUTHOR_STYLE = {
    "fcfs":            {"color": "#1f77b4", "marker": "s", "label": "FCFS"},
    "mlfq":            {"color": "#ff7f0e", "marker": "^", "label": "MLFQ"},
    "srtf-PO-X":       {"color": "#2ca02c", "marker": "D", "label": "Perception (PO)"},
    "tpt-class10-xxx": {"color": "#d62728", "marker": "o", "label": "Classification (10 buckets, width=820)"},
    "opt-xxx":         {"color": "#9467bd", "marker": "p", "label": "Ranking"},
}

# New variant styles for subplot (c) — 10-class reuses AUTHOR_STYLE color/marker
NEW_STYLE = {
    "tpt-class82-xxx": {"color": "#808080", "marker": "X", "label": "Classification (82 buckets, width=100)"},
    "tpt-width10-xxx": {"color": "#8B4513", "marker": "v", "label": "Classification (820 buckets, width=10)"},
    "tpt-pctl10-xxx":  {"color": "#FF69B4", "marker": "d", "label": "Classification (Percentile, 10 buckets, CE loss)"},
    "tpt-pctl10-mse-xxx": {"color": "#B19CD9", "marker": "h", "label": "Classification (Percentile, 10 buckets, MSE loss)"},
}

DATASET_TITLES = {
    "lmsys": "LMSYS-Chat-1M",
    "sharegpt": "ShareGPT",
}

LINE_WIDTH = 1.8
MARKER_SIZE = 5
DASH_STYLE = (4, 3)


RUNS = ["", "_test3", "_test4"]  # "" = original run


def load_our_data(dataset):
    path = os.path.join(BASE_DIR, f"metrics_{dataset}.csv")
    df = pd.read_csv(path)
    df["mean_nlatency_s"] = df["mean_nlatency_ms"] / 1000.0
    return df


def load_all_runs(dataset):
    """Load CSVs for all 3 runs and return combined DataFrame."""
    frames = []
    for suffix in RUNS:
        path = os.path.join(BASE_DIR, f"metrics_{dataset}{suffix}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["mean_nlatency_s"] = df["mean_nlatency_ms"] / 1000.0
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_authors_data(dataset):
    path = os.path.join(BASE_DIR, f"authors_metrics_{dataset}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def get_series(df, scheduler):
    sdf = df[df["scheduler"] == scheduler].sort_values("request_rate")
    if sdf.empty:
        return None, None
    return sdf["request_rate"].values, sdf["mean_nlatency_s"].values


def plot_line(ax, df, scheduler, style, linestyle="solid"):
    x, y = get_series(df, scheduler)
    if x is None:
        return
    dash = DASH_STYLE if linestyle == "dashed" else None
    ax.plot(
        x, y,
        color=style["color"], marker=style["marker"], label=style["label"],
        linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        linestyle="--" if linestyle == "dashed" else "-",
        dashes=dash if dash else [],
    )


def subplot_authors(ax, authors_df):
    """Subplot (a): Authors' original results, all dashed."""
    if authors_df is None:
        return
    for sched, style in AUTHOR_STYLE.items():
        plot_line(ax, authors_df, sched, {**style, "label": "_nolegend_"}, linestyle="dashed")
    ax.set_title("(a) Authors' Results", fontsize=9, fontweight="bold")


def subplot_ours_single(ax, our_df):
    """Subplot (b): Our reproduction, all solid (single run, no SE bands)."""
    for sched, style in AUTHOR_STYLE.items():
        plot_line(ax, our_df, sched, {**style, "label": "_nolegend_"}, linestyle="solid")
    ax.set_title("(b) Our Reproduction", fontsize=9, fontweight="bold")


def subplot_ours(ax, all_runs_df):
    """Subplot (b): Our reproduction with mean ± SE bands across 3 runs."""
    if all_runs_df is None:
        return
    for sched, style in AUTHOR_STYLE.items():
        sdf = all_runs_df[all_runs_df["scheduler"] == sched]
        if sdf.empty:
            continue
        rates = sorted(sdf["request_rate"].unique())
        means, se_lo, se_hi = [], [], []
        for rate in rates:
            vals = sdf[sdf["request_rate"] == rate]["mean_nlatency_s"].values
            m = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            means.append(m)
            se_lo.append(m - 2 * se)
            se_hi.append(m + 2 * se)
        rates = np.array(rates)
        means = np.array(means)
        ax.plot(rates, means, color=style["color"], marker=style["marker"],
                label="_nolegend_", linewidth=LINE_WIDTH, markersize=MARKER_SIZE)
        ax.fill_between(rates, se_lo, se_hi, color=style["color"], alpha=0.3)
    ax.set_title("(b) Our Reproduction (mean ± 2SE, n=3)", fontsize=9, fontweight="bold")


def subplot_variants(ax, our_df, authors_df):
    """Subplot (c): All classification variants merged.

    Shows: FCFS (solid), Ranking (solid), Authors' 10-class (dashed),
    our 10-class (solid), class82, width10, pctl10-CE, pctl10-MSE (all solid).
    """
    # FCFS baseline
    plot_line(ax, our_df, "fcfs", {**AUTHOR_STYLE["fcfs"], "label": "_nolegend_"})
    # Ranking baseline
    plot_line(ax, our_df, "opt-xxx", {**AUTHOR_STYLE["opt-xxx"], "label": "_nolegend_"})

    # Authors' 10-class (dashed reference)
    if authors_df is not None:
        ref_style = {**AUTHOR_STYLE["tpt-class10-xxx"], "label": "_nolegend_"}
        plot_line(ax, authors_df, "tpt-class10-xxx", ref_style, linestyle="dashed")

    # Our 10-class reproduction (solid) — same color/marker as authors', distinguished by line style
    plot_line(ax, our_df, "tpt-class10-xxx", {**AUTHOR_STYLE["tpt-class10-xxx"], "label": "_nolegend_"})

    # New variants (solid)
    plot_line(ax, our_df, "tpt-class82-xxx", {**NEW_STYLE["tpt-class82-xxx"], "label": "_nolegend_"})
    plot_line(ax, our_df, "tpt-width10-xxx", {**NEW_STYLE["tpt-width10-xxx"], "label": "_nolegend_"})
    plot_line(ax, our_df, "tpt-pctl10-xxx", {**NEW_STYLE["tpt-pctl10-xxx"], "label": "_nolegend_"})
    if "tpt-pctl10-mse-xxx" in our_df["scheduler"].values:
        plot_line(ax, our_df, "tpt-pctl10-mse-xxx", {**NEW_STYLE["tpt-pctl10-mse-xxx"], "label": "_nolegend_"})

    ax.set_title("(c) Classification Variants", fontsize=9, fontweight="bold")


def build_legend_handles():
    """Build a single unified legend for all 3 subplots — one row."""
    handles = []

    # Line style indicators — thicker and longer for clarity
    handles.append(Line2D([0], [0], color="black", linewidth=2.5, linestyle="-",
                          label="Ours"))
    handles.append(Line2D([0], [0], color="black", linewidth=3.0, linestyle=(0, (3, 2)),
                          label="Authors'"))

    # Schedulers from (a) and (b)
    for sched, style in AUTHOR_STYLE.items():
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                              markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                              label=style["label"]))

    # New variants from (c)
    for sched, style in NEW_STYLE.items():
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                              markersize=MARKER_SIZE, linewidth=LINE_WIDTH,
                              label=style["label"]))

    return handles


def make_figure(dataset):
    our_df = load_our_data(dataset)
    all_runs_df = load_all_runs(dataset)
    authors_df = load_authors_data(dataset)

    # Wide figure to fit full descriptive legend in one row
    fig, axes = plt.subplots(1, 3, figsize=(20, 2.9), sharey=True)

    subplot_authors(axes[0], authors_df)
    subplot_ours(axes[1], all_runs_df)
    subplot_variants(axes[2], our_df, authors_df)

    # Compute consistent y-axis across all subplots with integer ticks
    global_ymax = max(ax.get_ylim()[1] for ax in axes)
    y_ceil = int(np.ceil(global_ymax))  # round up to nearest integer
    for i, ax in enumerate(axes):
        ax.set_xlabel("Request Rate (req/s)", fontsize=9)
        if i == 0:
            ax.set_ylabel("Latency (s/token)", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)
        # Pad axes like authors' plot — don't start at edge
        ax.set_xlim(-2, 70)
        ax.set_ylim(-0.3, y_ceil + 0.15)
        ax.set_yticks(range(0, y_ceil + 1, 1))

    # Legend on top — single row spanning full figure width
    handles = build_legend_handles()
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(handles),
        fontsize=9,
        frameon=False,
        columnspacing=0.4,
        handletextpad=0.2,
        handlelength=3.0,
        borderpad=0,
        labelspacing=0.1,
    )

    plt.tight_layout(w_pad=1.5)
    out_path = os.path.join(OUTPUT_DIR, f"paper_figure_{dataset}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # ShareGPT first, then LMSYS
    for dataset in ["sharegpt", "lmsys"]:
        make_figure(dataset)


if __name__ == "__main__":
    main()
