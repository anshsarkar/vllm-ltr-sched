"""
Analyze class distribution for classification models on benchmark datasets.

Uses the same label logic as the trainer (trainer.py):
    label = label_max_length // label_group_size - min(label_max_length, length) // label_group_size

Output length is computed by tokenizing the 'generated' field — this is how it's
done everywhere in the codebase:
  - trainer.py line 57:  origin_len = len(self.tokenizer(item['generated'])['input_ids'])
  - benchmark_serving_real_with_metrics.py line 292:
        output_len = len(tokenizer(outputs[i].generated_text).input_ids)

Datasets come from LLM-ltr/Llama3-Trace (downloaded via scripts/download_data.sh).
Each jsonl line has {"prompt": ..., "generated": ...} — no precomputed lengths.
"""

import json
import argparse
import math
import numpy as np
from collections import Counter
from transformers import AutoTokenizer


def len2label(length, label_max_length, label_group_size):
    """Exact replication of RankingDataset.__len2label__ from trainer.py"""
    return label_max_length // label_group_size - min(label_max_length, length) // label_group_size


def label2token_range(label, label_max_length, label_group_size):
    """Invert len2label: given a class label, return the (lower, upper] token range."""
    tok_upper = (label_max_length // label_group_size - label) * label_group_size
    tok_lower = max(0, tok_upper - label_group_size)
    return tok_lower, tok_upper


def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize_lengths(data, tokenizer):
    """Tokenize the 'generated' field and return token lengths.

    This is exactly how output length is computed in:
      - trainer.py:         len(self.tokenizer(item['generated'])['input_ids'])
      - benchmark_serving:  len(tokenizer(outputs[i].generated_text).input_ids)
    """
    generated_texts = [item['generated'] for item in data]
    batch_size = 512
    lengths = []
    for i in range(0, len(generated_texts), batch_size):
        batch = generated_texts[i:i+batch_size]
        encoded = tokenizer(batch)
        lengths.extend(len(ids) for ids in encoded['input_ids'])
    return lengths


def print_distribution(token_lengths, group_size, label_max_length):
    """Print class distribution stats for a given bucket size."""
    num_classes = math.ceil(label_max_length / group_size)
    labels = [len2label(l, label_max_length, group_size) for l in token_lengths]
    counts = Counter(labels)

    print(f"\n{'='*60}")
    print(f"  bucket_size={group_size}, num_classes={num_classes}")
    print(f"{'='*60}")
    print(f"  Classes with data: {len(counts)}/{num_classes}")

    all_classes = set(range(num_classes))
    empty = all_classes - set(counts.keys())
    if empty:
        print(f"  Empty classes: {len(empty)}")

    # Per-class breakdown with actual token length stats
    print(f"\n  {'Class':>6} {'Tokens':>14} {'Count':>7} {'%':>7}  "
          f"{'ActualLen min-max':>20} {'Histogram'}")
    print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*7}  {'-'*20} {'-'*30}")
    max_count = max(counts.values()) if counts else 1
    for c in range(num_classes):
        lo, hi = label2token_range(c, label_max_length, group_size)
        cnt = counts.get(c, 0)
        pct = 100.0 * cnt / len(labels) if labels else 0
        bar = '#' * int(30 * cnt / max_count) if cnt > 0 else ''
        if cnt > 0:
            # Actual token lengths that fall into this class
            actual = [token_lengths[j] for j in range(len(token_lengths))
                      if labels[j] == c]
            amin, amax = min(actual), max(actual)
            print(f"  {c:>6} {lo:>6}-{hi:<6} {cnt:>7} {pct:>6.1f}%  "
                  f"{amin:>8}-{amax:<8}   {bar}")
        elif num_classes <= 20:
            print(f"  {c:>6} {lo:>6}-{hi:<6} {cnt:>7} {pct:>6.1f}%  "
                  f"{'':>20}   {bar}")

    # Top-5 most populated
    print(f"\n  Top-5 classes:")
    for c, cnt in counts.most_common(5):
        lo, hi = label2token_range(c, label_max_length, group_size)
        print(f"    class {c} (tokens {lo}-{hi}): {cnt} samples ({100*cnt/len(labels):.1f}%)")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Class distribution analysis")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the .jsonl dataset (from Llama3-Trace)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--label-max-length", type=int, default=8192)
    parser.add_argument("--label-group-sizes", type=int, nargs='+', default=[820, 100],
                        help="Bucket sizes to compare (default: 820 100)")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Display name for dataset (default: inferred from filename)")
    args = parser.parse_args()

    dataset_name = args.dataset_name or args.dataset.split('/')[-1].replace('.jsonl', '')

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print(f"Loading dataset: {args.dataset}")
    data = load_dataset(args.dataset)
    print(f"  {len(data)} samples")
    print(f"  Dataset: {dataset_name}")

    print("\nTokenizing generated outputs (same method as trainer.py and benchmark_serving)...")
    token_lengths = tokenize_lengths(data, tokenizer)

    arr = np.array(token_lengths)
    print(f"\nToken length stats (from tokenizer):")
    print(f"  min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, "
          f"median={np.median(arr):.1f}, std={arr.std():.1f}")
    print(f"  percentiles: p10={np.percentile(arr,10):.0f}, p25={np.percentile(arr,25):.0f}, "
          f"p50={np.percentile(arr,50):.0f}, p75={np.percentile(arr,75):.0f}, "
          f"p90={np.percentile(arr,90):.0f}, p99={np.percentile(arr,99):.0f}")

    # Print distribution for each bucket size
    for group_size in args.label_group_sizes:
        print_distribution(token_lengths, group_size, args.label_max_length)


if __name__ == "__main__":
    main()
