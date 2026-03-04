#!/usr/bin/env python3

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

# .pt tuple indices

IDX_TTFTS = 0
IDX_TPOTS = 1
IDX_LATENCIES = 2
IDX_NLATENCIES = 3
IDX_OUTPUT_LENS = 4
IDX_INPUT_LENS = 5
IDX_EST_LENS = 6
IDX_TEXTS = 7
IDX_AUX_SCORES = 8
IDX_PRED_SCORES = 9

# Style constants (match analyze_tradeoffs.py)
SCHED_COLORS = {
    "FCFS":               "#2196F3",
    "Oracle SJF":         "#9C27B0",
    "LTR-Ranking":        "#FF5722",
    "LTR-Classification": "#4CAF50",
}

SCHED_MARKERS = {
    "FCFS":               "o",
    "Oracle SJF":         "D",
    "LTR-Ranking":        "s",
    "LTR-Classification": "^",
}

SCHED_MAP = {
    "fcfs":            "FCFS",
    "sjf":             "Oracle SJF",
    "opt-xxx":         "LTR-Ranking",
    "tpt-class10-xxx": "LTR-Classification",
}

# Dataset configuration

DATASETS = {
    "sharegpt": {
        "label": "ShareGPT",
        "pt_dir": "experiments/results/sharegpt_8b_h100_metrics",
    },
    "lmsys": {
        "label": "LMSYS",
        "pt_dir": "experiments/results/lmsys_8b_h100_metrics",
    },
}

OUT_BASE = "experiments/analysis/idea_validation"

def parse_pt_filename(filename):
    m = re.match(
        r"^latency-(.+?)-([A-Z][a-z]\w*(?:-(?!p\d)\w+)*)"
        r"-p([\d.]+)-r([\d.]+)-c([\d.]+)-t([\d.]+)-o(.+)\.pt$",
        filename,
    )
    if not m:
        return None
    return {
        "scheduler": m.group(1),
        "model": m.group(2),
        "request_rate": float(m.group(4)),
    }


def get_latest_pt_files(folder, scheduler_filter=None):
    configs = defaultdict(list)

    for fname in os.listdir(folder):
        if not fname.endswith(".pt"):
            continue
        info = parse_pt_filename(fname)
        if info is None:
            continue
        if scheduler_filter and info["scheduler"] not in scheduler_filter:
            continue

        path = os.path.join(folder, fname)
        mtime = os.path.getmtime(path)
        key = (info["scheduler"], info["request_rate"])
        configs[key].append((mtime, path, fname))

    latest = {}
    for key, files in configs.items():
        files.sort(reverse=True)
        latest[key] = files[0][1]

    return latest


def load_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pt_as_dict(path):
    data = load_pt(path)
    return {
        "ttfts": np.array(data[IDX_TTFTS]),
        "tpots": np.array(data[IDX_TPOTS]),
        "latencies": np.array(data[IDX_LATENCIES]),
        "nlatencies": np.array(data[IDX_NLATENCIES]),
        "output_lens": np.array(data[IDX_OUTPUT_LENS]),
        "input_lens": np.array(data[IDX_INPUT_LENS]),
        "aux_model_scores": data[IDX_AUX_SCORES],
        "pred_scores": data[IDX_PRED_SCORES],
    }


def load_scheduler_data(pt_dir, scheduler_raw):
    latest = get_latest_pt_files(pt_dir, scheduler_filter={scheduler_raw})
    result = {}
    for (sched, rate), path in sorted(latest.items()):
        result[rate] = load_pt_as_dict(path)
    return result


def make_subplots(n, max_cols=3, cell_w=5, cell_h=4):
    ncols = min(max_cols, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell_w * ncols, cell_h * nrows))
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


def savefig(fig, path, dpi=150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
