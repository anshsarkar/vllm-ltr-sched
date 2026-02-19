#!/usr/bin/env python3

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Scheduler allow-list ──────────────────────────────────────────────────────

ALLOWED_SCHEDULERS = {
    "fcfs": "FCFS",
    "opt-xxx": "LTR-Ranking",
    "tpt-class10-xxx": "LTR-Classification",
}

SCHED_ORDER = ["FCFS", "LTR-Ranking", "LTR-Classification"]

SCHED_COLORS = {
    "FCFS": "#2196F3",
    "LTR-Ranking": "#FF5722",
    "LTR-Classification": "#4CAF50",
}

SCHED_MARKERS = {
    "FCFS": "o",
    "LTR-Ranking": "s",
    "LTR-Classification": "^",
}

# ── Filename parsing ──────────────────────────────────────────────────────────

def parse_json_filename(filename):
    m = re.match(
        r"^vllm-([\d.]+)qps-cv[\d.]+-Meta-Llama-3-8B-Instruct-(.+)-\d{8}-\d{6}\.json$",
        filename,
    )
    if not m:
        return None
    return {"request_rate": float(m.group(1)), "scheduler": m.group(2)}


def get_dataset_label(results_dir):
    folder = os.path.basename(os.path.normpath(results_dir))
    if folder.endswith("_8b"):
        return folder[:-3]
    return folder


def load_data(results_dir):
    rows = []
    agg_rows = []

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue

        info = parse_json_filename(fname)
        if info is None:
            continue

        sched_raw = info["scheduler"]
        if sched_raw not in ALLOWED_SCHEDULERS:
            continue

        sched_name = ALLOWED_SCHEDULERS[sched_raw]
        rate = info["request_rate"]
        path = os.path.join(results_dir, fname)

        with open(path) as f:
            d = json.load(f)

        # Aggregate record (for Pareto + throughput plots)
        agg_rows.append({
            "scheduler": sched_name,
            "request_rate": rate,
            "output_throughput": d.get("output_throughput", np.nan),
            "input_throughput": d.get("input_throughput", np.nan),
            "request_throughput": d.get("request_throughput", np.nan),
        })

        ttfts = d.get("ttfts", [])
        itls = d.get("itls", [])
        output_lens = d.get("output_lens", [])
        input_lens = d.get("input_lens", [])

        n = len(ttfts)
        for i in range(n):
            ttft = ttfts[i]
            itl_list = itls[i] if i < len(itls) else []
            out_len = max(output_lens[i] if i < len(output_lens) else 1, 1)
            in_len = input_lens[i] if i < len(input_lens) else 0

            e2e = ttft + sum(itl_list)
            nlatency = e2e / out_len
            tpot = float(np.mean(itl_list)) if itl_list else np.nan
            itl_std = float(np.std(itl_list)) if len(itl_list) > 1 else 0.0

            # Approximate arrival time assuming Poisson arrivals stored in order
            arrival_approx = i / rate
            finish_approx = arrival_approx + e2e

            rows.append({
                "scheduler": sched_name,
                "request_rate": rate,
                "ttft": ttft,
                "output_len": out_len,
                "input_len": in_len,
                "e2e_latency": e2e,
                "nlatency": nlatency,
                "tpot": tpot,
                "itl_std": itl_std,
                "arrival_time": arrival_approx,
                "finish_time": finish_approx,
            })

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    agg_df = pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame()
    return df, agg_df

def qps_sorted(df):
    return sorted(df["request_rate"].unique())


def rep_rates(rates, n=3):
    if len(rates) <= n:
        return list(rates)
    start = max(0, len(rates) // 3)
    selected = rates[start:start + n]
    if len(selected) < n:
        selected = rates[-n:]
    return list(selected)


def make_subplots(n, max_cols=3):
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = np.atleast_1d(np.array(axes)).flatten()
    return fig, axes_flat


def hide_unused(axes_flat, n_used):
    for ax in axes_flat[n_used:]:
        ax.set_visible(False)


def add_trend(ax, x, y, color, bins=20, label=None):
    if len(x) < bins:
        return
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(x.min(), x.max(), bins + 1)
    bx, by = [], []
    for i in range(bins):
        mask = (x >= edges[i]) & (x < edges[i + 1])
        if mask.sum() > 0:
            bx.append((edges[i] + edges[i + 1]) / 2)
            by.append(np.mean(y[mask]))
    if bx:
        ax.plot(bx, by, color=color, linewidth=2.5, zorder=5, label=label)


def scheds_in_df(df):
    present = set(df["scheduler"].unique())
    return [s for s in SCHED_ORDER if s in present]


def plot_ttft_cdf(df, out_dir, dataset):
    rates = qps_sorted(df)
    fig, axes = make_subplots(len(rates))

    for ax, rate in zip(axes, rates):
        sub = df[df["request_rate"] == rate]
        for sched in scheds_in_df(sub):
            vals = np.sort(sub[sub["scheduler"] == sched]["ttft"].values * 1000)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=sched,
                    color=SCHED_COLORS[sched],
                    marker=SCHED_MARKERS[sched], markevery=max(1, len(vals)//10),
                    markersize=4, linewidth=2)
        ax.set_title(f"{rate} req/s", fontsize=11)
        ax.set_xlabel("TTFT (ms)", fontsize=10)
        ax.set_ylabel("CDF", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    hide_unused(axes, len(rates))
    fig.suptitle("Time To First Token — CDF by QPS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_ttft_cdf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_quartile_fairness(df, out_dir, dataset):
    rates = qps_sorted(df)
    fig, axes = make_subplots(len(rates))

    q_labels = ["Q1\n(short)", "Q2", "Q3", "Q4\n(long)"]
    x = np.arange(4)
    width = 0.25

    for ax, rate in zip(axes, rates):
        sub = df[df["request_rate"] == rate].copy()
        try:
            sub["quartile"] = pd.qcut(sub["output_len"], 4, labels=False, duplicates="drop")
        except ValueError:
            ax.set_title(f"{rate} req/s (skipped)", fontsize=11)
            continue

        for j, sched in enumerate(scheds_in_df(sub)):
            sdf = sub[sub["scheduler"] == sched]
            means = []
            for q in range(4):
                qdata = sdf[sdf["quartile"] == q]["nlatency"].dropna()
                means.append(qdata.mean() * 1000 if len(qdata) > 0 else 0)
            ax.bar(x + j * width, means, width, label=sched,
                   color=SCHED_COLORS[sched], alpha=0.85)

        ax.set_title(f"{rate} req/s", fontsize=11)
        ax.set_xlabel("Output Length Quartile", fontsize=10)
        ax.set_ylabel("Mean nLatency (ms/token)", fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(q_labels, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    hide_unused(axes, len(rates))
    fig.suptitle("nLatency by Output-Length Quartile (fairness across request types)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_nlatency_by_output_quartile.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_tail_amplification(df, out_dir, dataset):
    fig, ax = plt.subplots(figsize=(8, 5))

    for sched in scheds_in_df(df):
        sdf = df[df["scheduler"] == sched]
        rates = sorted(sdf["request_rate"].unique())
        ratios = []
        for rate in rates:
            vals = sdf[sdf["request_rate"] == rate]["nlatency"].dropna().values
            med = np.median(vals)
            p99 = np.percentile(vals, 99)
            ratios.append(p99 / med if med > 0 else np.nan)
        ax.plot(rates, ratios, label=sched,
                color=SCHED_COLORS[sched], marker=SCHED_MARKERS[sched],
                linewidth=2, markersize=7)

    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("p99 / Median nLatency", fontsize=12)
    ax.set_title("Tail Latency Amplification (p99 / Median)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_tail_amplification.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_nlatency_scatter(df, out_dir, dataset):
    rates = rep_rates(qps_sorted(df))
    fig, axes = make_subplots(len(rates), max_cols=len(rates))

    for ax, rate in zip(axes, rates):
        sub = df[df["request_rate"] == rate]
        for sched in scheds_in_df(sub):
            sdf = sub[sub["scheduler"] == sched]
            # ax.scatter(sdf["output_len"], sdf["nlatency"] * 1000,
            #            label=sched, color=SCHED_COLORS[sched], alpha=0.3, s=8)
            add_trend(ax, sdf["output_len"].values, sdf["nlatency"].values * 1000,
                      SCHED_COLORS[sched], label=sched)
        ax.set_title(f"{rate} req/s", fontsize=11)
        ax.set_xlabel("Output Length (tokens)", fontsize=10)
        ax.set_ylabel("nLatency (ms/token)", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3)

    hide_unused(axes, len(rates))
    fig.suptitle("nLatency vs Output Length", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_nlatency_vs_output_len.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_e2e_cdf(df, out_dir, dataset):
    rates = qps_sorted(df)
    fig, axes = make_subplots(len(rates))

    for ax, rate in zip(axes, rates):
        sub = df[df["request_rate"] == rate]
        for sched in scheds_in_df(sub):
            vals = np.sort(sub[sub["scheduler"] == sched]["e2e_latency"].values * 1000)
            cdf = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, label=sched,
                    color=SCHED_COLORS[sched],
                    marker=SCHED_MARKERS[sched], markevery=max(1, len(vals)//10),
                    markersize=4, linewidth=2)
        ax.set_title(f"{rate} req/s", fontsize=11)
        ax.set_xlabel("E2E Latency (ms)", fontsize=10)
        ax.set_ylabel("CDF", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

    hide_unused(axes, len(rates))
    fig.suptitle("End-to-End Latency — CDF by QPS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_e2e_latency_cdf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def jains_index(values):
    n = len(values)
    if n == 0:
        return np.nan
    s = np.sum(values)
    sq = np.sum(values ** 2)
    return (s ** 2) / (n * sq) if sq > 0 else np.nan


def plot_fairness_index(df, out_dir, dataset):
    fig, ax = plt.subplots(figsize=(8, 5))

    for sched in scheds_in_df(df):
        sdf = df[df["scheduler"] == sched]
        rates = sorted(sdf["request_rate"].unique())
        jains = [jains_index(sdf[sdf["request_rate"] == r]["nlatency"].dropna().values)
                 for r in rates]
        ax.plot(rates, jains, label=sched,
                color=SCHED_COLORS[sched], marker=SCHED_MARKERS[sched],
                linewidth=2, markersize=7)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Perfect fairness (1.0)")
    ax.set_xlabel("Request Rate (req/s)", fontsize=12)
    ax.set_ylabel("Jain's Fairness Index", fontsize=12)
    ax.set_title("Jain's Fairness Index (1.0 = perfectly fair)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_jains_fairness.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_itl_jitter(df, out_dir, dataset):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for sched in scheds_in_df(df):
        sdf = df[df["scheduler"] == sched]
        rates = sorted(sdf["request_rate"].unique())
        mean_j, p90_j = [], []
        for rate in rates:
            vals = sdf[sdf["request_rate"] == rate]["itl_std"].dropna().values * 1000
            mean_j.append(np.mean(vals))
            p90_j.append(np.percentile(vals, 90))
        kw = dict(color=SCHED_COLORS[sched], marker=SCHED_MARKERS[sched], linewidth=2, markersize=7)
        ax1.plot(rates, mean_j, label=sched, **kw)
        ax2.plot(rates, p90_j, label=sched, **kw)

    for ax, title in [(ax1, "Mean ITL Std Dev"), (ax2, "p90 ITL Std Dev")]:
        ax.set_xlabel("Request Rate (req/s)", fontsize=12)
        ax.set_ylabel("ITL Std Dev (ms)", fontsize=12)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Inter-Token Latency Jitter (higher = more stuttery streaming)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_itl_jitter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_pareto(df, agg_df, out_dir, dataset):
    fig, ax = plt.subplots(figsize=(9, 6))

    for sched in scheds_in_df(df):
        sagg = agg_df[agg_df["scheduler"] == sched].sort_values("output_throughput")
        if sagg.empty:
            continue
        x_vals, y_vals, rate_labels = [], [], []
        for _, row in sagg.iterrows():
            rate = row["request_rate"]
            tput = row["output_throughput"]
            nlat_vals = df[(df["scheduler"] == sched) & (df["request_rate"] == rate)]["nlatency"].dropna()
            if len(nlat_vals) == 0 or np.isnan(tput):
                continue
            x_vals.append(tput)
            y_vals.append(nlat_vals.mean() * 1000)
            rate_labels.append(rate)

        ax.plot(x_vals, y_vals, label=sched,
                color=SCHED_COLORS[sched], marker=SCHED_MARKERS[sched],
                linewidth=2, markersize=8)
        for x, y, r in zip(x_vals, y_vals, rate_labels):
            ax.annotate(f"{r}", (x, y), textcoords="offset points", xytext=(5, 4), fontsize=7)

    ax.set_xlabel("Output Throughput (tokens/s)", fontsize=12)
    ax.set_ylabel("Mean nLatency (ms/token)", fontsize=12)
    ax.set_title("Throughput vs Latency Pareto Frontier", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_throughput_latency_pareto.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_finish_time_vs_output(df, out_dir, dataset):
    rates = rep_rates(qps_sorted(df))
    fig, axes = make_subplots(len(rates), max_cols=len(rates))

    for ax, rate in zip(axes, rates):
        sub = df[df["request_rate"] == rate]
        for sched in scheds_in_df(sub):
            sdf = sub[sub["scheduler"] == sched]
            # ax.scatter(sdf["output_len"], sdf["e2e_latency"] * 1000,
            #            label=sched, color=SCHED_COLORS[sched], alpha=0.3, s=8)
            add_trend(ax, sdf["output_len"].values, sdf["e2e_latency"].values * 1000,
                      SCHED_COLORS[sched], label=sched)
        ax.set_title(f"{rate} req/s", fontsize=11)
        ax.set_xlabel("Output Length (tokens)", fontsize=10)
        ax.set_ylabel("E2E Latency (ms)", fontsize=10)
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(True, alpha=0.3)

    hide_unused(axes, len(rates))
    fig.suptitle("Request Completion Time vs Output Length",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_finish_time_vs_output_len.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_throughput(agg_df, out_dir, dataset):
    metrics = [
        ("output_throughput", "Output Throughput (tokens/s)"),
        ("input_throughput",  "Input Throughput (tokens/s)"),
        ("request_throughput","Request Throughput (req/s)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (col, ylabel) in zip(axes, metrics):
        for sched in scheds_in_df(agg_df):
            sdf = agg_df[agg_df["scheduler"] == sched].sort_values("request_rate")
            ax.plot(
                sdf["request_rate"], sdf[col],
                label=sched, color=SCHED_COLORS[sched],
                marker=SCHED_MARKERS[sched], linewidth=2, markersize=7,
            )
        ax.set_xlabel("Request Rate (req/s)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Throughput Comparison by QPS", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_throughput_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_finish_time_vs_output_scatter(df, out_dir, dataset):
    rates = rep_rates(qps_sorted(df))
    scheds = scheds_in_df(df)
    n_scheds = len(scheds)
    n_rates = len(rates)

    fig, axes = plt.subplots(
        n_scheds, n_rates,
        figsize=(5 * n_rates, 4 * n_scheds),
        squeeze=False,
    )

    for row, sched in enumerate(scheds):
        for col, rate in enumerate(rates):
            ax = axes[row][col]
            sub = df[(df["scheduler"] == sched) & (df["request_rate"] == rate)]
            ax.scatter(
                sub["output_len"], sub["e2e_latency"] * 1000,
                color=SCHED_COLORS[sched], alpha=0.25, s=6,
            )
            add_trend(ax, sub["output_len"].values, sub["e2e_latency"].values * 1000,
                      SCHED_COLORS[sched])
            ax.set_title(f"{sched} — {rate} req/s", fontsize=10)
            ax.set_xlabel("Output Length (tokens)", fontsize=9)
            ax.set_ylabel("E2E Latency (ms)", fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Request Completion Time vs Output Length (scatter)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{dataset}_finish_time_vs_output_len_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tradeoff analysis comparing FCFS vs LTR scheduling"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Folder containing vllm benchmark JSON result files",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for plots (default: <script_dir>/analysis/tradeoffs/)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory")
        return 1

    dataset = get_dataset_label(results_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Each dataset gets its own subfolder: experiments/analysis/tradeoffs/{dataset}/
    out_dir = args.out_dir or os.path.join(script_dir, "analysis", "tradeoffs", dataset)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Dataset label : {dataset}")
    print(f"Results dir   : {results_dir}")
    print(f"Output dir    : {out_dir}")

    df, agg_df = load_data(results_dir)

    if df.empty:
        print("No matching data found. Check that results-dir contains JSON files "
              "with schedulers: fcfs, opt-xxx, tpt-class10-xxx")
        return 1

    print(f"Loaded {len(df):,} per-request rows")
    print(f"QPS levels    : {qps_sorted(df)}")
    print(f"Schedulers    : {scheds_in_df(df)}")
    print()

    plots = [
        ("TTFT CDF",                    lambda: plot_ttft_cdf(df, out_dir, dataset)),
        ("nLatency by quartile",         lambda: plot_quartile_fairness(df, out_dir, dataset)),
        ("Tail amplification",           lambda: plot_tail_amplification(df, out_dir, dataset)),
        ("nLatency vs output len",       lambda: plot_nlatency_scatter(df, out_dir, dataset)),
        ("E2E latency CDF",              lambda: plot_e2e_cdf(df, out_dir, dataset)),
        ("Jain's fairness index",        lambda: plot_fairness_index(df, out_dir, dataset)),
        ("ITL jitter",                   lambda: plot_itl_jitter(df, out_dir, dataset)),
        ("Throughput-latency Pareto",    lambda: plot_pareto(df, agg_df, out_dir, dataset)),
        ("Finish time vs output len",    lambda: plot_finish_time_vs_output(df, out_dir, dataset)),
        ("Finish time vs output len (scatter)", lambda: plot_finish_time_vs_output_scatter(df, out_dir, dataset)),
        ("Throughput comparison",        lambda: plot_throughput(agg_df, out_dir, dataset)),
    ]

    for name, fn in plots:
        print(f"[{name}]")
        fn()

    print(f"\nDone — {len(plots)} plots saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
