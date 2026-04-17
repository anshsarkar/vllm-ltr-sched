import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from data import PREDICTORS, CLASSIFIERS, RATES


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  saved {path}")

def plot_kendall_heatmap(tau_rows, out_dir, dataset):
    df = pd.DataFrame(tau_rows)
    pivot = df.pivot(index="scheduler", columns="rate", values="tau")
    pivot = pivot.reindex(index=PREDICTORS, columns=RATES)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(RATES)))
    ax.set_xticklabels([str(r) for r in RATES])
    ax.set_yticks(range(len(PREDICTORS)))
    ax.set_yticklabels(PREDICTORS, fontsize=8)
    ax.set_xlabel("Request rate")
    ax.set_title(f"Kendall tau vs SJF -- {dataset}")
    # Annotate cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, os.path.join(out_dir, f"{dataset}_kendall_tau.pdf"))


def plot_direction_breakdown(dir_rows, out_dir, dataset):
    df = pd.DataFrame(dir_rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "frac", "Fraction of requests"),
        (axes[1], "mean_nlatency_delta", "Mean nLatency delta vs SJF"),
    ]:
        # Average across rates
        agg = df.groupby(["scheduler", "direction"])[metric].mean().reset_index()
        pivot = agg.pivot(index="scheduler", columns="direction", values=metric)
        pivot = pivot.reindex(index=PREDICTORS)
        pivot = pivot[["long_as_short", "correct", "short_as_long"]]
        pivot.plot(kind="bar", ax=ax, color=["#d32f2f", "#4caf50", "#1976d2"])
        ax.set_title(f"{title} -- {dataset}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8)

    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"{dataset}_direction_breakdown.pdf"))


def plot_error_concentration(conc_data, input_edges, output_edges, out_dir, dataset):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, sched in enumerate(PREDICTORS):
        ax = axes_flat[idx]
        grid = conc_data[sched]
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", origin="lower")
        ax.set_title(sched, fontsize=9)
        ax.set_xlabel("output_len decile")
        ax.set_ylabel("input_len decile")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Mean |rank displacement| by (input, output) decile -- {dataset}", fontsize=12)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"{dataset}_error_concentration.pdf"))

def plot_confusion_matrices(cm_dict, out_dir, dataset):
    classifiers = [s for s in CLASSIFIERS if s in cm_dict]
    n = len(classifiers)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, sched in zip(axes, classifiers):
        cm = cm_dict[sched]
        num_classes = cm.shape[0]
        # Row-normalize for display
        row_sums = cm.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(row_sums > 0, cm / row_sums, 0)

        # For large matrices (class82, width10), coarsen to 10x10
        if num_classes > 20:
            cm_norm, labels = _coarsen(cm, 10)
        else:
            labels = list(range(num_classes))

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_title(sched, fontsize=9)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        # Mark the dangerous upper-right triangle
        size = cm_norm.shape[0]
        for i in range(size):
            for j in range(size):
                if j > i and cm_norm[i, j] > 0.05:
                    ax.text(j, i, f"{cm_norm[i, j]:.0%}",
                            ha="center", va="center", fontsize=5, color="red")
        fig.colorbar(im, ax=ax, shrink=0.7)

    fig.suptitle(f"Confusion matrices (0=longest) -- {dataset}", fontsize=12)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"{dataset}_confusion.pdf"))


def _coarsen(cm, target_bins):
    n = cm.shape[0]
    edges = np.linspace(0, n, target_bins + 1, dtype=int)
    coarse = np.zeros((target_bins, target_bins))
    for i in range(target_bins):
        for j in range(target_bins):
            coarse[i, j] = cm[edges[i]:edges[i+1], edges[j]:edges[j+1]].sum()
    # Row-normalize
    row_sums = coarse.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        coarse_norm = np.where(row_sums > 0, coarse / row_sums, 0)
    labels = [f"{edges[k]}-{edges[k+1]-1}" for k in range(target_bins)]
    return coarse_norm, labels


def plot_outlier_overlap(overlap_rows, out_dir, dataset):
    df = pd.DataFrame(overlap_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(df))
    w = 0.5
    ax.bar(x, df["flagged_by_all"], w, label="All 6 predictors", color="#d32f2f")
    ax.bar(x, df["flagged_by_majority"] - df["flagged_by_all"], w,
           bottom=df["flagged_by_all"], label="Majority (>3)", color="#ff9800")
    ax.bar(x, df["flagged_by_one_only"], w,
           bottom=df["flagged_by_majority"], label="One only", color="#90caf9")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in df["rate"]])
    ax.set_xlabel("Request rate")
    ax.set_ylabel("Number of flagged requests")
    ax.set_title(f"Outlier overlap (top 5% damage) -- {dataset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"{dataset}_outlier_overlap.pdf"))


def plot_noise_floor(nf_rows, out_dir, dataset):
    df = pd.DataFrame(nf_rows)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()

    for idx, sched in enumerate(PREDICTORS):
        ax = axes_flat[idx]
        sub = df[df["scheduler"] == sched].sort_values("bucket_idx")
        ax2 = ax.twinx()
        ax.bar(sub["bucket_idx"], sub["output_len_variance"],
               alpha=0.4, color="#1976d2", label="Output len variance")
        ax2.plot(sub["bucket_idx"], sub["mean_abs_displacement"],
                 color="#d32f2f", marker="o", label="Mean |displacement|")
        ax.set_title(sched, fontsize=9)
        ax.set_xlabel("Input length bucket")
        ax.set_ylabel("Variance", color="#1976d2")
        ax2.set_ylabel("|Displacement|", color="#d32f2f")

    fig.suptitle(f"Structural noise floor -- {dataset}", fontsize=12)
    fig.tight_layout()
    save(fig, os.path.join(out_dir, f"{dataset}_noise_floor.pdf"))
