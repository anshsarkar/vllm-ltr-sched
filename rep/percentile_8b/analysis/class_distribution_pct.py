import json
import argparse
import numpy as np
from collections import Counter
from transformers import AutoTokenizer


def compute_percentile_boundaries(lengths, num_classes):
    percentiles = np.linspace(0, 100, num_classes + 1)[1:-1]
    boundaries = np.percentile(lengths, percentiles)
    boundaries = np.unique(boundaries)
    actual_num_classes = len(boundaries) + 1
    return boundaries, actual_num_classes


def len2label(length, boundaries):
    num_classes = len(boundaries) + 1
    return num_classes - 1 - int(np.searchsorted(boundaries, length))


def label2token_range(label, boundaries, lengths):
    num_classes = len(boundaries) + 1
    idx = num_classes - 1 - label
    if idx == 0:
        return 0, int(boundaries[0])
    elif idx == len(boundaries):
        return int(boundaries[-1]), int(lengths.max())
    else:
        return int(boundaries[idx - 1]), int(boundaries[idx])


def load_dataset(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def tokenize_lengths(data, tokenizer):
    generated_texts = [item['generated'] for item in data]
    batch_size = 512
    lengths = []
    for i in range(0, len(generated_texts), batch_size):
        batch = generated_texts[i:i+batch_size]
        encoded = tokenizer(batch)
        lengths.extend(len(ids) for ids in encoded['input_ids'])
    return lengths


def print_distribution(token_lengths_arr, num_classes_requested):
    boundaries, actual_num_classes = compute_percentile_boundaries(token_lengths_arr, num_classes_requested)

    labels = [len2label(l, boundaries) for l in token_lengths_arr]
    counts = Counter(labels)

    print(f"\n{'='*70}")
    print(f"  Percentile labeling: requested={num_classes_requested}, "
          f"actual={actual_num_classes} (after dedup)")
    print(f"  Boundaries: {boundaries}")
    print(f"{'='*70}")

    print(f"\n  {'Class':>6} {'Tokens':>14} {'Count':>7} {'%':>7}  "
          f"{'ActualLen min-max':>20} {'Histogram'}")
    print(f"  {'-'*6} {'-'*14} {'-'*7} {'-'*7}  {'-'*20} {'-'*30}")
    max_count = max(counts.values()) if counts else 1
    for c in range(actual_num_classes):
        lo, hi = label2token_range(c, boundaries, token_lengths_arr)
        cnt = counts.get(c, 0)
        pct = 100.0 * cnt / len(labels) if labels else 0
        bar = '#' * int(30 * cnt / max_count) if cnt > 0 else ''
        if cnt > 0:
            actual = [int(token_lengths_arr[j]) for j in range(len(token_lengths_arr))
                      if labels[j] == c]
            amin, amax = min(actual), max(actual)
            print(f"  {c:>6} {lo:>6}-{hi:<6} {cnt:>7} {pct:>6.1f}%  "
                  f"{amin:>8}-{amax:<8}   {bar}")
        else:
            print(f"  {c:>6} {lo:>6}-{hi:<6} {cnt:>7} {pct:>6.1f}%  "
                  f"{'':>20}   {bar}")

    print(f"\n  Top-5 classes:")
    for c, cnt in counts.most_common(5):
        lo, hi = label2token_range(c, boundaries, token_lengths_arr)
        print(f"    class {c} (tokens {lo}-{hi}): {cnt} samples ({100*cnt/len(labels):.1f}%)")

    return labels


def main():
    parser = argparse.ArgumentParser(description="Percentile class distribution analysis")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the .jsonl dataset (from Llama3-Trace)")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--num-classes", type=int, nargs='+', default=[10, 20],
                        help="Number of percentile classes to compare (default: 10 20)")
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

    print("\nTokenizing generated outputs...")
    token_lengths = tokenize_lengths(data, tokenizer)

    arr = np.array(token_lengths)
    print(f"\nToken length stats:")
    print(f"  min={arr.min()}, max={arr.max()}, mean={arr.mean():.1f}, "
          f"median={np.median(arr):.1f}, std={arr.std():.1f}")
    print(f"  percentiles: p10={np.percentile(arr,10):.0f}, p25={np.percentile(arr,25):.0f}, "
          f"p50={np.percentile(arr,50):.0f}, p75={np.percentile(arr,75):.0f}, "
          f"p90={np.percentile(arr,90):.0f}, p99={np.percentile(arr,99):.0f}")

    for num_classes in args.num_classes:
        print_distribution(arr, num_classes)


if __name__ == "__main__":
    main()
