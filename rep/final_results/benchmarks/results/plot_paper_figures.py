#!/usr/bin/env python3
"""Generate paper-ready 3-subplot comparison figures for LMSYS and ShareGPT."""

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
    "srtf-PO-X":       {"color": "#2ca02c", "marker": "D", "label": "PO"},
    "tpt-class10-xxx": {"color": "#d62728", "marker": "o", "label": "Classification"},
    "opt-xxx":         {"color": "#9467bd", "marker": "p", "label": "Ranking (Authors)"},
}

# New variant styles
NEW_STYLE = {
    "tpt-width10-xxx": {"color": "#8B4513", "marker": "v", "label": "Classification (w=10)"},
    "tpt-class82-xxx": {"color": "#808080", "marker": "X", "label": "Classification (w=100)"},
    "tpt-pctl10-xxx":  {"color": "#FF69B4", "marker": "d", "label": "Classification (percentile, CE)"},
    "tpt-pctl10-mse-xxx": {"color": "#B19CD9", "marker": "h", "label": "Classification (percentile, MSE)"},
}

DATASET_TITLES = {
    "lmsys": "LMSYS-Chat-1M",
    "sharegpt": "ShareGPT",
}

LINE_WIDTH = 2
MARKER_SIZE = 7
DASH_STYLE = (5, 4)  # longer dashes with clear gaps


def load_our_data(dataset):
    path = os.path.join(BASE_DIR, f"metrics_{dataset}.csv")
    df = pd.read_csv(path)
    df["mean_nlatency_s"] = df["mean_nlatency_ms"] / 1000.0
    return df


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


def subplot_reproducibility(ax, our_df, authors_df):
    """Subplot 1: Authors (dashed) vs Our reproduction (solid)."""
    for sched, style in AUTHOR_STYLE.items():
        # Authors' line (dashed)
        if authors_df is not None:
            authors_style = {**style, "label": "_nolegend_"}
            plot_line(ax, authors_df, sched, authors_style, linestyle="dashed")
        # Our line (solid)
        ours_style = {**style, "label": "_nolegend_"}
        plot_line(ax, our_df, sched, ours_style, linestyle="solid")

    # Vertical line showing Classification gap at 64 req/s
    if authors_df is not None:
        ax_rates, ax_vals = get_series(authors_df, "tpt-class10-xxx")
        ox_rates, ox_vals = get_series(our_df, "tpt-class10-xxx")
        if ax_rates is not None and ox_rates is not None:
            a_val = float(ax_vals[ax_rates == 64.0][0])
            o_val = float(ox_vals[ox_rates == 64.0][0])
            ax.plot([64, 64], [min(a_val, o_val), max(a_val, o_val)],
                    color="#d62728", linewidth=1.5, linestyle="-", alpha=0.6)

    # Custom legend: one entry per scheduler + solid/dashed key
    handles = []
    for sched, style in AUTHOR_STYLE.items():
        handles.append(Line2D([0], [0], color=style["color"], marker=style["marker"],
                              markersize=6, linewidth=LINE_WIDTH, label=style["label"]))
    # Add solid/dashed indicators
    handles.append(Line2D([0], [0], color="black", linewidth=1.5, linestyle="-", label="Reproduced"))
    handles.append(Line2D([0], [0], color="black", linewidth=1.5, linestyle="--",
                          dashes=DASH_STYLE, label="Original"))
    ax.legend(handles=handles, fontsize=7.5, loc="upper left")
    ax.set_title("Reproducibility Comparison", fontsize=11, fontweight="bold")


def subplot_fixed_width(ax, our_df, authors_df):
    """Subplot 2: Fixed-width classification variants."""
    # Baselines (solid)
    plot_line(ax, our_df, "fcfs", AUTHOR_STYLE["fcfs"])
    plot_line(ax, our_df, "opt-xxx", AUTHOR_STYLE["opt-xxx"])

    # Authors' 10-class (dashed reference)
    if authors_df is not None:
        ref_style = {**AUTHOR_STYLE["tpt-class10-xxx"], "label": "Classification 10-class (Original)"}
        plot_line(ax, authors_df, "tpt-class10-xxx", ref_style, linestyle="dashed")

    # New variants (solid)
    plot_line(ax, our_df, "tpt-width10-xxx", NEW_STYLE["tpt-width10-xxx"])
    plot_line(ax, our_df, "tpt-class82-xxx", NEW_STYLE["tpt-class82-xxx"])

    ax.set_title("Fixed-Width Classification Variants", fontsize=11, fontweight="bold")


def subplot_percentile(ax, our_df, authors_df):
    """Subplot 3: Percentile classification variants."""
    # Baselines (solid)
    plot_line(ax, our_df, "fcfs", AUTHOR_STYLE["fcfs"])
    plot_line(ax, our_df, "opt-xxx", AUTHOR_STYLE["opt-xxx"])

    # Authors' 10-class (dashed reference)
    if authors_df is not None:
        ref_style = {**AUTHOR_STYLE["tpt-class10-xxx"], "label": "Classification 10-class (Original)"}
        plot_line(ax, authors_df, "tpt-class10-xxx", ref_style, linestyle="dashed")

    # Percentile variants (solid)
    plot_line(ax, our_df, "tpt-pctl10-xxx", NEW_STYLE["tpt-pctl10-xxx"])
    if "tpt-pctl10-mse-xxx" in our_df["scheduler"].values:
        plot_line(ax, our_df, "tpt-pctl10-mse-xxx", NEW_STYLE["tpt-pctl10-mse-xxx"])

    ax.set_title("Percentile Classification Variants", fontsize=11, fontweight="bold")


def make_figure(dataset):
    our_df = load_our_data(dataset)
    authors_df = load_authors_data(dataset)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"LLaMA-3-8B, 1 A100 80GB GPU, {DATASET_TITLES[dataset]}",
        fontsize=14, fontweight="bold", y=1.02,
    )

    subplot_reproducibility(axes[0], our_df, authors_df)
    subplot_fixed_width(axes[1], our_df, authors_df)
    subplot_percentile(axes[2], our_df, authors_df)

    subplot_labels = ["(a)", "(b)", "(c)"]
    for i, ax in enumerate(axes):
        ax.set_xlabel("Request Rate (req/s)", fontsize=11)
        ax.set_ylabel("Latency (s/token)", fontsize=11)
        if i != 0:  # subplot 1 has custom legend
            ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 68)
        ax.set_ylim(bottom=0)
        ax.text(0.5, -0.15, subplot_labels[i], transform=ax.transAxes,
                fontsize=13, fontweight="bold", ha="center", va="top")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"paper_figure_{dataset}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for dataset in ["lmsys", "sharegpt"]:
        make_figure(dataset)


if __name__ == "__main__":
    main()
