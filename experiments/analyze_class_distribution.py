#!/usr/bin/env python3

# python experiments/analyze_class_distribution.py /home/cc/vllm-ltr-sched/data/datasets/Llama3-Trace/llama3-8b-sharegpt-train-t1-s0-8192.jsonl --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --label-max-length 8192 --group-sizes 820 --show-split


import argparse
import json
import os
import sys
from collections import Counter

import numpy as np


def load_dataset(path):
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                dataset.append(json.loads(line))
    return dataset


def get_output_lengths(dataset, tokenizer):
    generated_texts = [d["generated"] for d in dataset]
    # Batch tokenize for efficiency
    batch_size = 512
    all_lens = []
    for i in range(0, len(generated_texts), batch_size):
        batch = generated_texts[i:i + batch_size]
        encoded = tokenizer(batch)
        all_lens.extend(len(ids) for ids in encoded["input_ids"])
    return np.array(all_lens)


def len2label(length, label_max_length, label_group_size):
    return label_max_length // label_group_size - min(label_max_length, length) // label_group_size


def analyze_distribution(output_lens, label_max_length, label_group_size, split_name="Full"):
    import math
    num_labels = math.ceil(label_max_length / label_group_size)
    labels = np.array([len2label(l, label_max_length, label_group_size) for l in output_lens])

    counts = Counter(labels)
    total = len(labels)

    # Sort by label
    sorted_labels = sorted(counts.keys())

    # Compute statistics
    max_count = max(counts.values())
    min_count = min(counts.values()) if len(counts) > 1 else max_count
    imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

    # Map labels back to length ranges
    # label = lml // gs - min(lml, length) // gs
    # => length range for label L: [(lml // gs - L - 1) * gs + 1, (lml // gs - L) * gs]
    # but capped at [0, lml]
    max_label_val = label_max_length // label_group_size

    print(f"\n  {'─' * 75}")
    print(f"  {split_name} | group_size={label_group_size} | num_labels={num_labels} | N={total}")
    print(f"  {'─' * 75}")
    print(f"  {'Label':>6} {'Length Range':>20} {'Count':>8} {'Pct':>8} {'Bar'}")
    print(f"  {'─' * 75}")

    bar_max = 50
    for label in range(num_labels):
        count = counts.get(label, 0)
        pct = count / total * 100

        # Reverse the label formula to get length range
        # label = max_label_val - min(lml, length) // gs
        # => min(lml, length) // gs = max_label_val - label
        # => length ≈ (max_label_val - label) * gs ± gs
        # label = max_label_val - length // gs
        # So for a given label L:
        #   length // gs = max_label_val - L
        #   length range: [(max_label_val - L) * gs, (max_label_val - L + 1) * gs - 1]
        lower = (max_label_val - label) * label_group_size
        upper = (max_label_val - label + 1) * label_group_size - 1
        if label == 0:
            upper = label_max_length  # label 0 captures longest
        if label == num_labels - 1:
            lower = max(lower, 1)  # shortest label starts at 1, not 0

        bar_len = int(count / max_count * bar_max) if max_count > 0 else 0
        bar = "█" * bar_len

        # Only print non-empty labels, or all if few labels
        if count > 0 or num_labels <= 20:
            print(f"  {label:>6} {lower:>8}-{upper:<8}   {count:>8} {pct:>7.1f}%  {bar}")

    # Print empty labels count if too many
    empty_labels = num_labels - len(counts)
    if empty_labels > 0 and num_labels > 20:
        print(f"  ... {empty_labels} empty labels omitted ...")

    print(f"  {'─' * 75}")
    print(f"  Imbalance ratio (max/min occupied): {imbalance_ratio:.1f}x")
    print(f"  Occupied labels: {len(counts)}/{num_labels} ({len(counts)/num_labels*100:.1f}%)")
    print(f"  Majority class: label {max(counts, key=counts.get)} ({max_count}/{total} = {max_count/total*100:.1f}%)")

    # Entropy (normalized)
    probs = np.array([counts.get(l, 0) / total for l in range(num_labels)])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(num_labels) if num_labels > 1 else 1
    print(f"  Entropy: {entropy:.2f} / {max_entropy:.2f} (normalized: {entropy/max_entropy:.3f})")

    return labels, counts


def print_length_statistics(output_lens, split_name="Full"):
    print(f"\n  Raw Output Length Statistics ({split_name}):")
    print(f"    N:      {len(output_lens)}")
    print(f"    Mean:   {np.mean(output_lens):.1f}")
    print(f"    Median: {np.median(output_lens):.1f}")
    print(f"    Std:    {np.std(output_lens):.1f}")
    print(f"    Min:    {np.min(output_lens):.0f}")
    print(f"    Max:    {np.max(output_lens):.0f}")
    print(f"    Q25:    {np.percentile(output_lens, 25):.0f}")
    print(f"    Q75:    {np.percentile(output_lens, 75):.0f}")
    print(f"    Q90:    {np.percentile(output_lens, 90):.0f}")
    print(f"    Q99:    {np.percentile(output_lens, 99):.0f}")



def main():
    parser = argparse.ArgumentParser(
        description="Analyze class distribution in LTR training data"
    )
    parser.add_argument("dataset", help="Path to JSONL dataset file")
    parser.add_argument(
        "--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Tokenizer to use for counting output tokens"
    )
    parser.add_argument(
        "--label-max-length", type=int, default=8192,
        help="Maximum label length (default: 8192, context window for Llama-3)"
    )
    parser.add_argument(
        "--group-sizes", type=int, nargs="+", default=[1, 10, 100, 820],
        help="Label group sizes to analyze (default: 1 10 100 820)"
    )
    parser.add_argument(
        "--show-split", action="store_true",
        help="Show train/test split (90/10 as in trainer.py)"
    )
    parser.add_argument(
        "--no-tokenize", action="store_true",
        help="Skip tokenization, use character count as proxy (fast mode)"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.dataset):
        print(f"Error: '{args.dataset}' not found")
        sys.exit(1)

    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"  Total samples: {len(dataset)}")

    if args.no_tokenize:
        print("  (Using word count as proxy for token count — fast mode)")
        output_lens = np.array([len(d["generated"].split()) for d in dataset])
    else:
        print(f"  Tokenizing with: {args.tokenizer}")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        output_lens = get_output_lengths(dataset, tokenizer)

    # Full dataset stats
    print(f"\n{'=' * 80}")
    print(f"  DATASET: {os.path.basename(args.dataset)}")
    print(f"{'=' * 80}")

    print_length_statistics(output_lens, "Full Dataset")

    for gs in args.group_sizes:
        analyze_distribution(output_lens, args.label_max_length, gs, "Full Dataset")

    if args.show_split:
        # 90/10 split as in trainer.py
        split_idx = int(0.9 * len(output_lens))
        train_lens = output_lens[:split_idx]
        test_lens = output_lens[split_idx:]

        print(f"\n\n{'=' * 80}")
        print(f"  TRAIN/TEST SPLIT (90/10)")
        print(f"{'=' * 80}")

        print_length_statistics(train_lens, "Train")
        print_length_statistics(test_lens, "Test")

        for gs in args.group_sizes:
            analyze_distribution(train_lens, args.label_max_length, gs, "Train")
            analyze_distribution(test_lens, args.label_max_length, gs, "Test")

        # Check if distributions differ significantly
        print(f"\n  Distribution Comparison (Train vs Test):")
        from scipy import stats as sp_stats
        ks_stat, ks_p = sp_stats.ks_2samp(train_lens, test_lens)
        print(f"    KS test: statistic={ks_stat:.4f}, p-value={ks_p:.4e}")
        if ks_p < 0.05:
            print(f"    WARNING: Train and test distributions differ significantly (p < 0.05)")
        else:
            print(f"    Train and test distributions are similar (p >= 0.05)")


if __name__ == "__main__":
    main()
