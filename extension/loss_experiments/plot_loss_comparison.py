#!/usr/bin/env python3
"""Quick comparison plot: mean nlatency for B4 loss experiments vs baselines."""

import os
import re
import numpy as np
import torch
import matplotlib.pyplot as plt

IDX_NLATENCIES = 3

# --- Sources ---
BASELINE_DIR = "extension/benchmarks/results"
B4_DIR = "extension/loss_experiments/benchmarks/results"

# scheduler_key -> (directory, pt_scheduler_name, display_name)
SCHEDULERS = {
    "fcfs":         (BASELINE_DIR, "fcfs",                       "FCFS"),
    "sjf":          (BASELINE_DIR, "sjf",                        "Oracle SJF"),
    "opt_author":   (BASELINE_DIR, "opt-xxx",                    "Ranking (author)"),
    "pctl10_mse":   (BASELINE_DIR, "tpt-pctl10-mse-xxx",        "Pctl10-MSE (baseline)"),
    "opt_ours":     (B4_DIR,       "opt-xxx",                    "Ranking (ours, listMLE)"),
    # "costsens_ce":  (B4_DIR,       "tpt-pctl10-costsensitive-xxx", "Cost-Sensitive CE"),
    # "coral":        (B4_DIR,       "tpt-pctl10-coral-xxx",       "CORAL"),
    # "focal_ce":     (B4_DIR,       "tpt-pctl10-focal-xxx",       "Focal CE"),
    # "pairwise_ce":  (B4_DIR,       "tpt-pctl10-pairwise-xxx",    "Pairwise CE"),
}

DATASETS = ["lmsys", "sharegpt"]
RATES = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]


def parse_rate(fname):
    m = re.search(r"-r([\d.]+)-", fname)
    return float(m.group(1)) if m else None


def load_mean_nlatency(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    return float(np.mean(data[IDX_NLATENCIES]))


def collect_data():
    """Returns {dataset: {sched_key: {rate: mean_nlatency}}}"""
    results = {}
    for ds in DATASETS:
        results[ds] = {}
        for skey, (base_dir, pt_name, _) in SCHEDULERS.items():
            folder = os.path.join(base_dir, ds)
            if not os.path.isdir(folder):
                continue
            rate_vals = {}
            for fname in os.listdir(folder):
                if not fname.endswith(".pt"):
                    continue
                if not fname.startswith(f"latency-{pt_name}-"):
                    continue
                rate = parse_rate(fname)
                if rate is None:
                    continue
                path = os.path.join(folder, fname)
                rate_vals[rate] = load_mean_nlatency(path)
            if rate_vals:
                results[ds][skey] = rate_vals
    return results


def plot(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Style config
    style_map = {
        "fcfs":        dict(color="gray",    ls="--", marker="s", lw=1.5),
        "sjf":         dict(color="black",   ls="-",  marker="*", lw=2),
        "opt_author":  dict(color="#1f77b4", ls="-",  marker="o", lw=2),
        "pctl10_mse":  dict(color="#2ca02c", ls="-",  marker="^", lw=2),
        "opt_ours":    dict(color="#ff7f0e", ls="-",  marker="D", lw=2),
        # "costsens_ce": dict(color="#d62728", ls="-",  marker="v", lw=2),
        # "coral":       dict(color="#9467bd", ls="-",  marker="P", lw=2),
        # "focal_ce":    dict(color="#e377c2", ls="-",  marker="h", lw=2),
        # "pairwise_ce": dict(color="#17becf", ls="-",  marker="X", lw=2),
    }

    for ds in DATASETS:
        fig, ax = plt.subplots(figsize=(10, 6))
        ds_data = results.get(ds, {})

        for skey, (_, _, label) in SCHEDULERS.items():
            if skey not in ds_data:
                continue
            rv = ds_data[skey]
            rates_sorted = sorted(rv.keys())
            vals = [rv[r] for r in rates_sorted]
            st = style_map.get(skey, {})
            ax.plot(rates_sorted, vals, label=label, markersize=7, **st)

        ax.set_xlabel("Request Rate (req/s)", fontsize=12)
        ax.set_ylabel("Mean Normalized Latency", fontsize=12)
        ax.set_title(f"B4 Loss Experiment Comparison v2 — {ds.upper()}", fontsize=14, fontweight="bold")
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks([2, 4, 8, 16, 32, 64])

        out_path = os.path.join(output_dir, f"loss_comparison_v3_{ds}.pdf")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    # Also print a summary table for rates 32 and 64
    print("\n=== Mean Normalized Latency (rates 32, 64) ===\n")
    for ds in DATASETS:
        print(f"--- {ds.upper()} ---")
        ds_data = results.get(ds, {})
        header = f"{'Scheduler':<30s} {'Rate 32':>10s} {'Rate 64':>10s}"
        print(header)
        print("-" * len(header))
        for skey, (_, _, label) in SCHEDULERS.items():
            if skey not in ds_data:
                continue
            rv = ds_data[skey]
            r32 = f"{rv.get(32.0, float('nan')):.4f}"
            r64 = f"{rv.get(64.0, float('nan')):.4f}"
            print(f"{label:<30s} {r32:>10s} {r64:>10s}")
        print()


if __name__ == "__main__":
    results = collect_data()
    plot(results, "extension/loss_experiments/benchmarks/results")