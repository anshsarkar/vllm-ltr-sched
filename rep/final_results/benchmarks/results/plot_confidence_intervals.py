#!/usr/bin/env python3
"""Plot mean normalized latency with 95% confidence intervals across multiple runs.

Uses data from 3 runs per dataset (original, test3, test4).
Skips tpt-pctl10-mse-xxx since it's only present in the original run.
Generates one figure per dataset with all schedulers.
"""

import os
import sys
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_SCRIPT = os.path.join(BASE_DIR, "..", "..", "analysis", "plot_benchmark_results.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots_for_paper")

DATASETS = ["lmsys", "sharegpt"]
RUNS = ["", "_test3", "_test4"]  # "" = original run

# Scheduler excluded because it only exists in the original run
EXCLUDE_SCHEDULERS = {"tpt-pctl10-mse-xxx"}

# Display names and colors (matching paper figure style)
SCHED_STYLE = {
    "fcfs":            {"color": "#1f77b4", "marker": "s", "label": "FCFS"},
    "mlfq":            {"color": "#ff7f0e", "marker": "^", "label": "MLFQ"},
    "srtf-PO-X":       {"color": "#2ca02c", "marker": "D", "label": "PO (Oracle)"},
    "tpt-class10-xxx": {"color": "#d62728", "marker": "o", "label": "Classification (10 buckets, width=820)"},
    "opt-xxx":         {"color": "#9467bd", "marker": "p", "label": "Ranking"},
    "tpt-class82-xxx": {"color": "#808080", "marker": "X", "label": "Classification (82 buckets, width=100)"},
    "tpt-width10-xxx": {"color": "#8B4513", "marker": "v", "label": "Classification (820 buckets, width=10)"},
    "tpt-pctl10-xxx":  {"color": "#FF69B4", "marker": "d", "label": "Percentile (10 buckets, CE loss)"},
}

DATASET_TITLES = {
    "lmsys": "LMSYS-Chat-1M",
    "sharegpt": "ShareGPT",
}


def generate_csvs():
    """Generate per-run CSVs using the existing analysis script."""
    generated = []
    for dataset in DATASETS:
        for run_suffix in RUNS:
            folder_name = f"{dataset}{run_suffix}"
            input_folder = os.path.join(BASE_DIR, folder_name)
            if not os.path.isdir(input_folder):
                print(f"WARNING: {input_folder} not found, skipping")
                continue

            csv_name = f"metrics_{dataset}{run_suffix}.csv"
            csv_path = os.path.join(BASE_DIR, csv_name)

            # Always regenerate to ensure consistency
            print(f"Generating {csv_name}...")
            result = subprocess.run(
                [sys.executable, ANALYSIS_SCRIPT, input_folder,
                 "-o", csv_path, "--no-plot"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr}")
                continue

            generated.append((dataset, run_suffix, csv_path))
            print(f"  Done: {csv_path}")

    return generated


def load_all_runs(dataset):
    """Load CSVs for all runs of a dataset and return combined DataFrame."""
    frames = []
    for run_suffix in RUNS:
        csv_path = os.path.join(BASE_DIR, f"metrics_{dataset}{run_suffix}.csv")
        if not os.path.exists(csv_path):
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        df = pd.read_csv(csv_path)
        df["run"] = run_suffix if run_suffix else "_original"
        frames.append(df)

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def check_data_completeness(df, dataset):
    """Verify all expected schedulers and rates are present in all runs."""
    df_filtered = df[~df["scheduler"].isin(EXCLUDE_SCHEDULERS)]
    runs = df_filtered["run"].unique()
    schedulers = df_filtered["scheduler"].unique()
    rates = sorted(df_filtered["request_rate"].unique())

    print(f"\n--- {dataset} data completeness ---")
    print(f"Runs: {sorted(runs)}")
    print(f"Schedulers: {sorted(schedulers)}")
    print(f"Request rates: {rates}")

    missing = []
    for run in runs:
        for sched in schedulers:
            for rate in rates:
                subset = df_filtered[
                    (df_filtered["run"] == run) &
                    (df_filtered["scheduler"] == sched) &
                    (df_filtered["request_rate"] == rate)
                ]
                if subset.empty:
                    missing.append((run, sched, rate))

    if missing:
        print(f"MISSING combinations ({len(missing)}):")
        for run, sched, rate in missing:
            print(f"  run={run}, scheduler={sched}, rate={rate}")
        return False
    else:
        print("All combinations present across all 3 runs.")
        return True


def plot_confidence_intervals(dataset):
    """Plot mean nlatency with 95% CI shaded bands for each scheduler."""
    df = load_all_runs(dataset)
    if df is None:
        print(f"No data for {dataset}")
        return

    # Filter out schedulers not in all runs
    df = df[~df["scheduler"].isin(EXCLUDE_SCHEDULERS)]

    # Check completeness
    if not check_data_completeness(df, dataset):
        print(f"WARNING: incomplete data for {dataset}, proceeding anyway")

    # Convert to seconds
    df["mean_nlatency_s"] = df["mean_nlatency_ms"] / 1000.0

    fig, ax = plt.subplots(figsize=(10, 5))

    for sched, style in SCHED_STYLE.items():
        sched_df = df[df["scheduler"] == sched]
        if sched_df.empty:
            continue

        rates = sorted(sched_df["request_rate"].unique())
        means = []
        ci_lows = []
        ci_highs = []

        for rate in rates:
            values = sched_df[sched_df["request_rate"] == rate]["mean_nlatency_s"].values
            n = len(values)
            mean = np.mean(values)
            means.append(mean)

            if n >= 3:
                sem = stats.sem(values)
                ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=sem)
                ci_lows.append(ci[0])
                ci_highs.append(ci[1])
            else:
                # With fewer than 3 points, just show the range
                ci_lows.append(np.min(values))
                ci_highs.append(np.max(values))

        rates = np.array(rates)
        means = np.array(means)
        ci_lows = np.array(ci_lows)
        ci_highs = np.array(ci_highs)

        ax.plot(rates, means, color=style["color"], marker=style["marker"],
                label=style["label"], linewidth=1.8, markersize=5)
        ax.fill_between(rates, ci_lows, ci_highs,
                        color=style["color"], alpha=0.15)

    ax.set_xlabel("Request Rate (req/s)", fontsize=11)
    ax.set_ylabel("Mean Normalized Latency (s/token)", fontsize=11)
    ax.set_title(f"Scheduler Comparison with 95% CI — {DATASET_TITLES[dataset]}\n"
                 f"LLaMA-3-8B, 1 A100 80GB GPU (n=3 runs)", fontsize=12)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-2, 70)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"confidence_interval_{dataset}.pdf")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Step 1: Generating per-run CSVs")
    print("=" * 60)
    generate_csvs()

    print("\n" + "=" * 60)
    print("Step 2: Plotting confidence intervals")
    print("=" * 60)
    for dataset in DATASETS:
        plot_confidence_intervals(dataset)


if __name__ == "__main__":
    main()
