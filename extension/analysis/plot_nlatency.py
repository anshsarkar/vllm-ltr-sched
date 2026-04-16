#!/usr/bin/env python3
"""Alignment check + mean normalised latency plot for the extension benchmark runs.

Usage (from repo root):
    python extension/analysis/plot_nlatency.py

Outputs:
    extension/benchmarks/results/ext_nlatency.pdf
    prints alignment table to stdout
"""

import os
import re
from collections import defaultdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "extension", "benchmarks", "results")

DATASETS = ["lmsys", "sharegpt"]

IDX_NLATENCIES = 3

SCHED_LABELS = {
    "fcfs":              "FCFS",
    "sjf":               "Oracle SJF",
    "opt-xxx":           "LTR-Ranking",
    "tpt-class10-xxx":   "Class-10 (w=820)",
    "tpt-class82-xxx":   "Class-82 (w=100)",
    "tpt-width10-xxx":   "Class-820 (w=10)",
    "tpt-pctl10-xxx":    "Pctl-10 (CE)",
    "tpt-pctl10-mse-xxx":"Pctl-10 (MSE)",
}

STYLES = {
    "fcfs":               {"color": "#1f77b4", "marker": "s",  "ls": "--"},
    "sjf":                {"color": "#2ca02c", "marker": "D",  "ls": "--"},
    "opt-xxx":            {"color": "#9467bd", "marker": "p",  "ls": "-"},
    "tpt-class10-xxx":    {"color": "#d62728", "marker": "o",  "ls": "-"},
    "tpt-class82-xxx":    {"color": "#808080", "marker": "X",  "ls": "-"},
    "tpt-width10-xxx":    {"color": "#8B4513", "marker": "v",  "ls": "-"},
    "tpt-pctl10-xxx":     {"color": "#FF69B4", "marker": "d",  "ls": "-"},
    "tpt-pctl10-mse-xxx": {"color": "#7B68EE", "marker": "h",  "ls": "-"},
}

DATASET_TITLES = {
    "lmsys":    "LMSYS-Chat-1M",
    "sharegpt": "ShareGPT",
}


def parse_filename(fname):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        fname,
    )
    if not m:
        return None
    return {"scheduler": m.group(1), "rate": float(m.group(4))}


def load_pt_files(folder):
    """Return dict: (scheduler, rate) -> mean_nlatency_s."""
    records = defaultdict(list)
    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        info = parse_filename(fname)
        if info is None:
            continue
        path = os.path.join(folder, fname)
        records[(info["scheduler"], info["rate"])].append(
            (os.path.getmtime(path), path)
        )

    results = {}
    for key, files in records.items():
        files.sort(reverse=True)
        path = files[0][1]
        data = torch.load(path, map_location="cpu", weights_only=False)
        nlatencies = np.array(data[IDX_NLATENCIES])
        results[key] = {"mean_nlatency_s": float(np.mean(nlatencies)),
                        "n": len(nlatencies)}
    return results


# ---- Alignment check --------------------------------------------------------

print("=== Alignment check (request counts per rate per dataset) ===\n")
for dataset in DATASETS:
    folder = os.path.join(RESULTS_ROOT, dataset)
    pt_data = load_pt_files(folder)
    rates = sorted({r for (_, r) in pt_data})
    scheds = sorted({s for (s, _) in pt_data})
    print(f"Dataset: {DATASET_TITLES[dataset]}")
    header = f"{'Rate':>8} | " + " | ".join(f"{s:>22}" for s in scheds)
    print(header)
    print("-" * len(header))
    aligned = True
    for rate in rates:
        counts = [pt_data.get((s, rate), {}).get("n", "MISSING") for s in scheds]
        row = f"{rate:>8.1f} | " + " | ".join(f"{str(c):>22}" for c in counts)
        print(row)
        numeric = [c for c in counts if isinstance(c, int)]
        if numeric and (max(numeric) - min(numeric)) > 0:
            aligned = False
    status = "OK" if aligned else "MISMATCH — check seeds"
    print(f"Alignment: {status}\n")


# ---- Plot -------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax, dataset in zip(axes, DATASETS):
    folder = os.path.join(RESULTS_ROOT, dataset)
    pt_data = load_pt_files(folder)

    rates = sorted({r for (_, r) in pt_data})
    scheds_present = sorted({s for (s, _) in pt_data},
                            key=lambda s: list(SCHED_LABELS).index(s)
                            if s in SCHED_LABELS else 99)

    for sched in scheds_present:
        ys = [pt_data.get((sched, r), {}).get("mean_nlatency_s", np.nan)
              for r in rates]
        style = STYLES.get(sched, {"color": "black", "marker": "o", "ls": "-"})
        ax.plot(rates, ys,
                label=SCHED_LABELS.get(sched, sched),
                color=style["color"],
                marker=style["marker"],
                linestyle=style["ls"],
                linewidth=1.8,
                markersize=5)

    ax.set_title(DATASET_TITLES[dataset], fontsize=12)
    ax.set_xlabel("Request rate (req/s)", fontsize=10)
    ax.set_ylabel("Mean norm. latency (s/token)", fontsize=10)
    ax.set_xticks(rates)
    ax.grid(True, linestyle=":", alpha=0.5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.15))

fig.tight_layout()
out_path = os.path.join(RESULTS_ROOT, "ext_nlatency.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Plot saved to {out_path}")
