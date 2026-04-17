#!/usr/bin/env python3

import os
import sys
import pandas as pd

from data import load_all, check_alignment, load_pctl_boundaries
from analyses import (
    kendall_tau_vs_sjf,
    error_direction_breakdown,
    error_concentration,
    confusion_matrices,
    outlier_overlap,
    noise_floor,
)
from plots import (
    plot_kendall_heatmap,
    plot_direction_breakdown,
    plot_error_concentration,
    plot_confusion_matrices,
    plot_outlier_overlap,
    plot_noise_floor,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
for _ in range(3):
    REPO_ROOT = os.path.dirname(REPO_ROOT)

DATASETS = {
    "lmsys": os.path.join(REPO_ROOT, "extension", "benchmarks", "results", "lmsys"),
    "sharegpt": os.path.join(REPO_ROOT, "extension", "benchmarks", "results", "sharegpt"),
}

CONFIG_DIR = os.path.join(REPO_ROOT, "extension", "training", "configs")
OUT_DIR = os.path.join(SCRIPT_DIR, "results")


def run_dataset(name, dataset_dir):
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")

    # Load data
    print("Loading .pt files...")
    all_data = load_all(dataset_dir)

    # Alignment check
    print("Checking alignment...")
    alignment = check_alignment(all_data)
    for rate, n in alignment:
        print(f"  rate={rate}: {n} requests, aligned OK")

    # Load percentile boundaries
    pctl_bounds = load_pctl_boundaries(CONFIG_DIR, name)

    # Analysis 1: Kendall's tau
    print("\n[1/6] Kendall's tau heatmap...")
    tau_rows = kendall_tau_vs_sjf(all_data)
    pd.DataFrame(tau_rows).to_csv(
        os.path.join(OUT_DIR, f"{name}_kendall_tau.csv"), index=False
    )
    plot_kendall_heatmap(tau_rows, OUT_DIR, name)

    # Analysis 2: Error direction breakdown
    print("[2/6] Error direction breakdown...")
    dir_rows = error_direction_breakdown(all_data)
    pd.DataFrame(dir_rows).to_csv(
        os.path.join(OUT_DIR, f"{name}_directions.csv"), index=False
    )
    plot_direction_breakdown(dir_rows, OUT_DIR, name)

    # Analysis 3: Error concentration heatmap
    print("[3/6] Error concentration heatmap...")
    conc_data, in_edges, out_edges = error_concentration(all_data)
    plot_error_concentration(conc_data, in_edges, out_edges, OUT_DIR, name)

    # Analysis 4: Confusion matrices
    print("[4/6] Confusion matrices...")
    cm_dict = confusion_matrices(all_data, pctl_bounds)
    plot_confusion_matrices(cm_dict, OUT_DIR, name)

    # Analysis 5: Cross-scheduler outlier overlap
    print("[5/6] Outlier overlap...")
    overlap_rows = outlier_overlap(all_data)
    pd.DataFrame(overlap_rows).to_csv(
        os.path.join(OUT_DIR, f"{name}_overlap.csv"), index=False
    )
    plot_outlier_overlap(overlap_rows, OUT_DIR, name)

    # Analysis 6: Structural noise floor
    print("[6/6] Noise floor...")
    nf_rows = noise_floor(all_data)
    pd.DataFrame(nf_rows).to_csv(
        os.path.join(OUT_DIR, f"{name}_noise_floor.csv"), index=False
    )
    plot_noise_floor(nf_rows, OUT_DIR, name)

    print(f"\nDone with {name}.")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, path in DATASETS.items():
        run_dataset(name, path)
    print(f"\nAll results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
