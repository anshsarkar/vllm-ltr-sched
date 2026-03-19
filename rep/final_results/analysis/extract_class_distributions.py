#!/usr/bin/env python3
# python rep/final_results/analysis/extract_class_distributions.py --lmsys-dataset data/llama3-8b-lmsys-test-t1-s0-8192.jsonl --sharegpt-dataset data/llama3-8b-sharegpt-test-t1-s0-8192.jsonl --output rep/final_results/analysis/class_distributions.csv

import argparse
import csv
import json
import math
import sys

import numpy as np
from transformers import AutoTokenizer

LABEL_MAX_LENGTH = 8192

# ── Uniform-bucket labeling (from trainer.py) ────────────────────────────
def len2label_uniform(length, num_labels, group_size):
    return min(num_labels - 1, max(0,
        LABEL_MAX_LENGTH // group_size
        - min(LABEL_MAX_LENGTH, int(length)) // group_size))

def label2range_uniform(label, num_labels, group_size):
    k = LABEL_MAX_LENGTH // group_size - label
    tok_lo = k * group_size
    tok_hi = min((k + 1) * group_size, LABEL_MAX_LENGTH)
    return tok_lo, tok_hi

# ── Percentile labeling (from trainer_percentile.py) ─────────────────────
def compute_percentile_boundaries(lengths, num_classes):
    percentiles = np.linspace(0, 100, num_classes + 1)[1:-1]
    boundaries = np.percentile(lengths, percentiles)
    boundaries = np.unique(boundaries)
    return boundaries

def len2label_pctl(length, boundaries):
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

# ── Predictor configs ────────────────────────────────────────────────────
PREDICTORS = [
    # (scheduler_name, label_for_display, type, param)
    ("tpt-class10-xxx",  "Classification (#Buckets=10)",     "uniform", 10),
    ("tpt-class82-xxx",  "Classification (Bucket Size=100)", "uniform", 82),
    ("tpt-width10-xxx",  "Classification (Bucket Size=10)",  "uniform", 819),
    ("tpt-pctl10-xxx",   "Classification (Percentile, CE)",  "pctl",    10),
    ("tpt-pctl10-mse-xxx","Classification (Percentile, MSE)","pctl",    10),
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

def get_distribution(token_lengths, pred_type, param):
    """Returns list of (bucket_id, count, tok_lo, tok_hi)."""
    rows = []
    if pred_type == "uniform":
        num_labels = param
        group_size = LABEL_MAX_LENGTH // num_labels
        labels = [len2label_uniform(l, num_labels, group_size) for l in token_lengths]
        from collections import Counter
        counts = Counter(labels)
        for c in range(num_labels):
            lo, hi = label2range_uniform(c, num_labels, group_size)
            rows.append((c, counts.get(c, 0), lo, hi))
    elif pred_type == "pctl":
        num_classes_req = param
        boundaries = compute_percentile_boundaries(token_lengths, num_classes_req)
        actual_num = len(boundaries) + 1
        labels = [len2label_pctl(l, boundaries) for l in token_lengths]
        from collections import Counter
        counts = Counter(labels)
        max_len = int(token_lengths.max())
        for c in range(actual_num):
            lo, hi = label2range_pctl(c, boundaries, max_len)
            rows.append((c, counts.get(c, 0), lo, hi))
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmsys-dataset", type=str, required=True)
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
            dist = get_distribution(token_lengths, pred_type, param)
            total = sum(r[1] for r in dist)
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
