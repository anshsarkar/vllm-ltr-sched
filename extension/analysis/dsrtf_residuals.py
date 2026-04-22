#!/usr/bin/env python3
"""DSRTF residual diagnostic: why doesn't DSRTF improve over static?

For each predictor, computes:
  residual = midpoint(predicted_class) - true_output_len

Then reports:
  1. Summary stats per predictor (mean, median, std of |residual|)
  2. Number of distinct sort keys (effective resolution)
  3. "Correction ratio" = |residual| / true_output_len
     If > 1, the request finishes before DSRTF can correct the ordering.
  4. Histogram of residuals per predictor (PDF output)

Usage (from repo root):
    python extension/analysis/dsrtf_residuals.py
"""

import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "extension", "benchmarks", "results")
CONFIG_DIR = os.path.join(REPO_ROOT, "extension", "training", "configs")

IDX_OUTPUT_LENS = 4
IDX_AUX_SCORES = 8

LABEL_MAX_LENGTH = 8192

DATASETS = ["lmsys", "sharegpt"]
DATASET_TITLES = {"lmsys": "LMSYS-Chat-1M", "sharegpt": "ShareGPT"}

# High rates only (where DSRTF should matter)
RATES = [32.0, 64.0]

# Predictors: (scheduler_key, name)
CLASSIFIERS = [
    ("tpt-class10-xxx",    "Class-10 (w=820)"),
    ("tpt-class82-xxx",    "Class-82 (w=100)"),
    ("tpt-width10-xxx",    "Class-820 (w=10)"),
    ("tpt-pctl10-xxx",     "Pctl-10 (CE)"),
    ("tpt-pctl10-mse-xxx", "Pctl-10 (MSE)"),
]

UNIFORM_WIDTHS = {
    "tpt-class10-xxx": 820,
    "tpt-class82-xxx": 100,
    "tpt-width10-xxx": 10,
}


def build_midpoint_table_uniform(width):
    max_class = LABEL_MAX_LENGTH // width
    table = {}
    for c in range(max_class + 1):
        table[c] = (max_class - c) * width + width / 2.0
    return table


def build_midpoint_table_pctl(boundaries):
    num_classes = len(boundaries) + 1
    extended = [0] + boundaries + [LABEL_MAX_LENGTH]
    table = {}
    for c in range(num_classes):
        bin_idx = num_classes - 1 - c
        table[c] = (extended[bin_idx] + extended[bin_idx + 1]) / 2.0
    return table


def load_pctl_boundaries(dataset):
    # pctl10 and pctl10-mse have identical boundaries per dataset
    path = os.path.join(CONFIG_DIR, f"pctl10_{dataset}_boundaries.json")
    with open(path) as f:
        return json.load(f)["boundaries"]


def pt_filename(scheduler, rate):
    return f"latency-{scheduler}-Meta-Llama-3-8B-Instruct-p0-r{rate}-c1.0-t60.0-o-1.pt"


def load_pt(folder, scheduler, rate):
    path = os.path.join(folder, pt_filename(scheduler, rate))
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "output_lens": np.array(data[IDX_OUTPUT_LENS]),
        "aux_scores": np.array(data[IDX_AUX_SCORES]),
    }


def main():
    # Build midpoint tables
    midpoint_tables = {}
    for dataset in DATASETS:
        midpoint_tables[dataset] = {}
        pctl_bounds = load_pctl_boundaries(dataset)
        for sched_key, _ in CLASSIFIERS:
            if sched_key in UNIFORM_WIDTHS:
                midpoint_tables[dataset][sched_key] = build_midpoint_table_uniform(
                    UNIFORM_WIDTHS[sched_key]
                )
            else:
                midpoint_tables[dataset][sched_key] = build_midpoint_table_pctl(pctl_bounds)

    # Collect residuals
    all_residuals = {}  # (dataset, sched_key) -> array of residuals
    all_output_lens = {}

    for dataset in DATASETS:
        folder = os.path.join(RESULTS_ROOT, dataset)
        for sched_key, _ in CLASSIFIERS:
            residuals_list = []
            output_lens_list = []
            table = midpoint_tables[dataset][sched_key]
            for rate in RATES:
                d = load_pt(folder, sched_key, rate)
                pred_classes = d["aux_scores"].astype(int)
                output_lens = d["output_lens"]
                midpoints = np.array([table.get(c, 0) for c in pred_classes])
                residuals = midpoints - output_lens
                residuals_list.append(residuals)
                output_lens_list.append(output_lens)
            all_residuals[(dataset, sched_key)] = np.concatenate(residuals_list)
            all_output_lens[(dataset, sched_key)] = np.concatenate(output_lens_list)

    # --- Print summary table ---
    print("=== DSRTF Residual Diagnostic (rates 32, 64) ===\n")
    print(f"{'Dataset':<12} {'Predictor':<22} {'Distinct':>8} "
          f"{'Mean Res':>10} {'Med |Res|':>10} {'Std |Res|':>10} "
          f"{'Corr>1':>8} {'Med Corr':>10}")
    print("-" * 95)

    for dataset in DATASETS:
        for sched_key, pred_name in CLASSIFIERS:
            res = all_residuals[(dataset, sched_key)]
            olen = all_output_lens[(dataset, sched_key)]
            abs_res = np.abs(res)
            # Correction ratio: |residual| / true_output_len
            # If > 1, request finishes before DSRTF can correct
            corr_ratio = abs_res / np.maximum(olen, 1)
            frac_uncorrectable = np.mean(corr_ratio > 1.0)
            # Number of distinct midpoints actually used
            table = midpoint_tables[dataset][sched_key]
            pred_classes = []
            for rate in RATES:
                d = load_pt(os.path.join(RESULTS_ROOT, dataset), sched_key, rate)
                pred_classes.append(d["aux_scores"].astype(int))
            pred_classes = np.concatenate(pred_classes)
            distinct = len(set(table.get(c, 0) for c in pred_classes))

            print(f"{dataset:<12} {pred_name:<22} {distinct:>8} "
                  f"{np.mean(res):>+10.1f} {np.median(abs_res):>10.1f} {np.std(abs_res):>10.1f} "
                  f"{frac_uncorrectable:>7.1%} {np.median(corr_ratio):>10.2f}")
        print()

    # --- Histogram plot ---
    fig, axes = plt.subplots(len(CLASSIFIERS), 2, figsize=(12, 14), sharex="col")

    for row, (sched_key, pred_name) in enumerate(CLASSIFIERS):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row, col]
            res = all_residuals[(dataset, sched_key)]

            # Clip for display
            clip = 2000
            clipped = np.clip(res, -clip, clip)
            ax.hist(clipped, bins=80, color="#1f77b4", alpha=0.7, edgecolor="none")
            ax.axvline(0, color="red", lw=1.2, ls="--", label="Perfect prediction")
            ax.axvline(np.mean(res), color="orange", lw=1.2, ls="-",
                       label=f"Mean = {np.mean(res):+.0f}")

            if row == 0:
                ax.set_title(DATASET_TITLES[dataset], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(pred_name, fontsize=9, fontweight="bold")
            if row == len(CLASSIFIERS) - 1:
                ax.set_xlabel("Residual: midpoint(pred) - true output len", fontsize=9)

            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)

    fig.suptitle("DSRTF Prediction Residuals (rates 32 & 64)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = os.path.join(RESULTS_ROOT, "dsrtf_residuals.pdf")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"\nHistogram saved to {out_path}")

    # --- Correction ratio histogram ---
    fig2, axes2 = plt.subplots(len(CLASSIFIERS), 2, figsize=(12, 14), sharex="col")

    for row, (sched_key, pred_name) in enumerate(CLASSIFIERS):
        for col, dataset in enumerate(DATASETS):
            ax = axes2[row, col]
            res = all_residuals[(dataset, sched_key)]
            olen = all_output_lens[(dataset, sched_key)]
            corr = np.abs(res) / np.maximum(olen, 1)

            clipped = np.clip(corr, 0, 5)
            ax.hist(clipped, bins=80, color="#2ca02c", alpha=0.7, edgecolor="none")
            ax.axvline(1.0, color="red", lw=1.5, ls="--",
                       label="Ratio = 1 (finishes before correction)")
            frac = np.mean(corr > 1.0)
            ax.axvline(np.median(corr), color="orange", lw=1.2, ls="-",
                       label=f"Median = {np.median(corr):.2f}")

            if row == 0:
                ax.set_title(DATASET_TITLES[dataset], fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(pred_name, fontsize=9, fontweight="bold")
            if row == len(CLASSIFIERS) - 1:
                ax.set_xlabel("|residual| / true_output_len", fontsize=9)

            ax.legend(fontsize=7, loc="upper right")
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            ax.text(0.95, 0.75, f"{frac:.0%} > 1", transform=ax.transAxes,
                    fontsize=9, ha="right", fontweight="bold", color="red")

    fig2.suptitle("Correction Ratio: can DSRTF fix the error before the request finishes?",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout()
    out_path2 = os.path.join(RESULTS_ROOT, "dsrtf_correction_ratio.pdf")
    fig2.savefig(out_path2, bbox_inches="tight", dpi=150)
    print(f"Correction ratio histogram saved to {out_path2}")


if __name__ == "__main__":
    main()
