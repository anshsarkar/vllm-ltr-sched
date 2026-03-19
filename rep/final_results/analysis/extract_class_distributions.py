#!/usr/bin/env python3
# python rep/final_results/analysis/extract_class_distributions.py \
#     --lmsys-dataset data/datasets/Llama3-Trace/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
#     --sharegpt-dataset data/datasets/Llama3-Trace/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
#     --output rep/final_results/analysis/class_distributions.csv

import argparse
import csv
import json
import math
import sys
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

LABEL_MAX_LENGTH = 8192

# ── Uniform-bucket labeling (exactly matching trainer.py) ─────────────────
def len2label_uniform(length, label_group_size):
    """Matches trainer.py RankingDataset.__len2label__ exactly."""
    label = LABEL_MAX_LENGTH // label_group_size - min(LABEL_MAX_LENGTH, length) // label_group_size
    return label

def label2range_uniform(label, label_group_size):
    """Token range [lo, hi) for a given uniform label."""
    k = LABEL_MAX_LENGTH // label_group_size - label
    tok_lo = k * label_group_size
    tok_hi = min((k + 1) * label_group_size, LABEL_MAX_LENGTH)
    return tok_lo, tok_hi

# ── Percentile labeling (exactly matching trainer_percentile.py) ──────────
def compute_percentile_boundaries(lengths, num_classes):
    percentiles = np.linspace(0, 100, num_classes + 1)[1:-1]
    boundaries = np.percentile(lengths, percentiles)
    boundaries = np.unique(boundaries)
    return boundaries

def len2label_pctl(length, boundaries):
    """Matches trainer_percentile.py PercentileDataset.__len2label__ exactly."""
    num_classes = len(boundaries) + 1
    return num_classes - 1 - int(np.searchsorted(boundaries, length))

def label2range_pctl(label, boundaries, max_len):
    num_classes = len(boundaries) + 1
    idx = num_classes - 1 - label
    if idx == 0:
        return 0, int(boundaries[0])
    elif idx == len(boundaries):
        return int(boundaries[-1]), max_len
    else:
        return int(boundaries[idx - 1]), int(boundaries[idx])

# ── Predictor configs (matching actual training scripts) ──────────────────
# type="uniform": param = label_group_size (as passed to trainer.py)
# type="pctl":    param = num_classes (as passed to trainer_percentile.py)
PREDICTORS = [
    # (scheduler_name,      display_label,                      type,      param)
    ("tpt-class10-xxx",     "Classification (#Buckets=10)",     "uniform", 820),   # label_group_size=820 → num_labels=ceil(8192/820)=10
    ("tpt-class82-xxx",     "Classification (Bucket Size=100)", "uniform", 100),   # label_group_size=100 → num_labels=ceil(8192/100)=82
    ("tpt-width10-xxx",     "Classification (Bucket Size=10)",  "uniform", 10),    # label_group_size=10  → num_labels=ceil(8192/10)=820
    ("tpt-pctl10-xxx",      "Classification (Percentile, CE)",  "pctl",    10),
    ("tpt-pctl10-mse-xxx",  "Classification (Percentile, MSE)", "pctl",    10),
]

def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def tokenize_lengths(data, tokenizer):
    generated_texts = [item['generated'] for item in data]
    lengths = []
    for i in range(0, len(generated_texts), 512):
        batch = generated_texts[i:i+512]
        encoded = tokenizer(batch)
        lengths.extend(len(ids) for ids in encoded['input_ids'])
    return np.array(lengths)

def get_distribution_uniform(token_lengths, label_group_size):
    """Returns list of (bucket_id, count, tok_lo, tok_hi) for uniform bucketing."""
    num_labels = math.ceil(LABEL_MAX_LENGTH / label_group_size)
    labels = [len2label_uniform(l, label_group_size) for l in token_lengths]
    counts = Counter(labels)

    rows = []
    for c in range(num_labels):
        lo, hi = label2range_uniform(c, label_group_size)
        rows.append((c, counts.get(c, 0), lo, hi))
    return rows

def get_distribution_pctl(token_lengths, num_classes_req):
    """Returns list of (bucket_id, count, tok_lo, tok_hi) for percentile bucketing."""
    boundaries = compute_percentile_boundaries(token_lengths, num_classes_req)
    actual_num = len(boundaries) + 1
    labels = [len2label_pctl(l, boundaries) for l in token_lengths]
    counts = Counter(labels)
    max_len = int(token_lengths.max())

    rows = []
    for c in range(actual_num):
        lo, hi = label2range_pctl(c, boundaries, max_len)
        rows.append((c, counts.get(c, 0), lo, hi))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-dataset", type=str, required=True,
                        help="LMSYS training JSONL (should be c20000:30000 for 8B)")
    parser.add_argument("--sharegpt-dataset", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output", type=str, default="class_distributions.csv")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    datasets = {
        "lmsys": args.lmsys_dataset,
        "sharegpt": args.sharegpt_dataset,
    }

    all_rows = []
    for ds_name, ds_path in datasets.items():
        print(f"\nProcessing {ds_name}: {ds_path}")
        data = load_dataset(ds_path)
        print(f"  {len(data)} samples")
        token_lengths = tokenize_lengths(data, tokenizer)
        print(f"  Token lengths: min={token_lengths.min()}, max={token_lengths.max()}, "
              f"mean={token_lengths.mean():.1f}")

        for sched, label, pred_type, param in PREDICTORS:
            if pred_type == "uniform":
                dist = get_distribution_uniform(token_lengths, label_group_size=param)
            elif pred_type == "pctl":
                dist = get_distribution_pctl(token_lengths, num_classes_req=param)
            else:
                continue

            total = sum(r[1] for r in dist)
            num_labels = len(dist)
            print(f"  {sched}: group_size={param}, num_labels={num_labels}, "
                  f"non-zero={sum(1 for r in dist if r[1] > 0)}")

            for bucket_id, count, tok_lo, tok_hi in dist:
                pct = 100.0 * count / total if total > 0 else 0
                all_rows.append({
                    "dataset": ds_name,
                    "scheduler": sched,
                    "label": label,
                    "bucket_id": bucket_id,
                    "count": count,
                    "pct": round(pct, 2),
                    "tok_lo": tok_lo,
                    "tok_hi": tok_hi,
                })

    fieldnames = ["dataset", "scheduler", "label", "bucket_id", "count", "pct", "tok_lo", "tok_hi"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {args.output}")

if __name__ == "__main__":
    main()
