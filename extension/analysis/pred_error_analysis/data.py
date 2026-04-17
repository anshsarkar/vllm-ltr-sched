import os
import re
import json
import numpy as np
import torch

# .pt tuple indices (forked from experiments/idea_validation/common.py)
IDX_NLATENCIES = 3
IDX_OUTPUT_LENS = 4
IDX_INPUT_LENS = 5
IDX_AUX_SCORES = 8
IDX_PRED_SCORES = 9

RATES = [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]

SCHEDULERS = [
    "fcfs", "sjf", "opt-xxx",
    "tpt-class10-xxx", "tpt-class82-xxx", "tpt-width10-xxx",
    "tpt-pctl10-xxx", "tpt-pctl10-mse-xxx",
]

# The 6 learned predictors (everything except fcfs and sjf)
PREDICTORS = SCHEDULERS[2:]

# 5 classifiers (all tpt-*)
CLASSIFIERS = [s for s in PREDICTORS if s.startswith("tpt-")]

# Bucket config for uniform-width classifiers: scheduler -> group_size
UNIFORM_BUCKET_SIZES = {
    "tpt-class10-xxx": 820,
    "tpt-class82-xxx": 100,
    "tpt-width10-xxx": 10,
}

# Percentile-boundary classifiers
PCTL_CLASSIFIERS = ["tpt-pctl10-xxx", "tpt-pctl10-mse-xxx"]

LABEL_MAX_LENGTH = 8192


def _pt_filename(scheduler, rate):
    return f"latency-{scheduler}-Meta-Llama-3-8B-Instruct-p0-r{rate}-c1.0-t60.0-o-1.pt"


def load_pt(dataset_dir, scheduler, rate):
    path = os.path.join(dataset_dir, _pt_filename(scheduler, rate))
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "nlatencies": np.array(data[IDX_NLATENCIES]),
        "output_lens": np.array(data[IDX_OUTPUT_LENS]),
        "input_lens": np.array(data[IDX_INPUT_LENS]),
        "aux_scores": np.array(data[IDX_AUX_SCORES]),
        "pred_scores": np.array(data[IDX_PRED_SCORES]),
    }


def load_all(dataset_dir):
    result = {}
    for sched in SCHEDULERS:
        for rate in RATES:
            result[(sched, rate)] = load_pt(dataset_dir, sched, rate)
    return result


def check_alignment(all_data):
    alignment = []
    for rate in RATES:
        ref = all_data[("sjf", rate)]["input_lens"]
        for sched in SCHEDULERS:
            if sched == "sjf":
                continue
            other = all_data[(sched, rate)]["input_lens"]
            if len(ref) != len(other) or not np.array_equal(ref, other):
                raise ValueError(
                    f"Alignment failed: sjf vs {sched} at rate={rate}"
                )
        alignment.append((rate, len(ref)))
    return alignment


def load_pctl_boundaries(config_dir, dataset):
    boundaries = {}
    for prefix in ["pctl10", "pctl10_mse"]:
        sched = f"tpt-{prefix.replace('_', '-')}-xxx"
        path = os.path.join(config_dir, f"{prefix}_{dataset}_boundaries.json")
        with open(path) as f:
            boundaries[sched] = json.load(f)["boundaries"]
    return boundaries


def output_len_to_class_uniform(output_lens, group_size):
    max_class = LABEL_MAX_LENGTH // group_size
    raw = np.minimum(output_lens, LABEL_MAX_LENGTH) // group_size
    return max_class - raw


def output_len_to_class_pctl(output_lens, boundaries):
    # np.searchsorted gives bin index 0..len(boundaries) where 0 = below first boundary
    # That means bin 0 = shortest. We need to flip so class 0 = longest.
    num_classes = len(boundaries) + 1
    raw_bin = np.searchsorted(boundaries, output_lens, side="right")
    return (num_classes - 1) - raw_bin
