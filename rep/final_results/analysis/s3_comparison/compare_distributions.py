#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_alpaca(path):
    with open(path) as f:
        data = json.load(f)
    return [d["output"] for d in data if d["output"].strip()]


def load_jsonl(path):
    outputs = []
    with open(path) as f:
        for line in f:
            d = json.loads(line.strip())
            if d.get("generated", "").strip():
                outputs.append(d["generated"])
    return outputs


def tokenize_lengths(texts, tokenizer):
    lengths = []
    for i in range(0, len(texts), 512):
        encoded = tokenizer(texts[i:i+512], truncation=False)
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return np.array(lengths)


def bucket_distribution(lengths, max_seq, num_buckets):
    """Assign lengths to uniform buckets and return per-bucket counts."""
    width = max_seq / num_buckets
    clipped = np.clip(lengths, 0, max_seq)
    ids = np.minimum((clipped / width).astype(int), num_buckets - 1)
    counts = np.bincount(ids, minlength=num_buckets)
    return counts, width


def print_stats(name, lengths):
    print(f"\n  {name} (n={len(lengths)})")
    print(f"    mean={np.mean(lengths):.0f}  median={np.median(lengths):.0f}  "
          f"std={np.std(lengths):.0f}  min={np.min(lengths)}  max={np.max(lengths)}")
    print(f"    Q25={np.percentile(lengths,25):.0f}  Q75={np.percentile(lengths,75):.0f}  "
          f"Q90={np.percentile(lengths,90):.0f}  Q99={np.percentile(lengths,99):.0f}")


def print_buckets(name, counts, width, num_buckets):
    total = counts.sum()
    max_count = counts.max()
    majority_pct = max_count / total * 100
    occupied = (counts > 0).sum()

    # imbalance ratio
    occ = counts[counts > 0]
    imbalance = occ.max() / occ.min()

    # normalized entropy
    p = counts / total
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) / np.log2(num_buckets)

    print(f"\n  {name} — {num_buckets} buckets, width={width:.0f}")
    print(f"  {'Bucket':>7} {'Range':>14} {'Count':>7} {'Pct':>7}  Bar")
    print(f"  {'─'*60}")
    for i in range(num_buckets):
        lo, hi = int(i * width), int((i + 1) * width)
        pct = counts[i] / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {i:>7} {lo:>6}-{hi:<6} {counts[i]:>7} {pct:>6.1f}%  {bar}")
    print(f"  {'─'*60}")
    print(f"  majority bucket: {majority_pct:.1f}% | imbalance: {imbalance:.1f}x | "
          f"entropy: {entropy:.3f} | occupied: {occupied}/{num_buckets}")


def plot_comparison(all_data, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    colors = ["#2196F3", "#FF9800", "#4CAF50"]
    names = list(all_data.keys())

    # Row 1: length histograms
    for i, name in enumerate(names):
        ax = axes[0][i]
        lengths = all_data[name]["lengths"]
        ax.hist(np.clip(lengths, 0, 3000), bins=80, color=colors[i], alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.set_title(f"{name}\nmedian={np.median(lengths):.0f}, mean={np.mean(lengths):.0f}")
        ax.set_xlabel("Output length (tokens)")
        ax.set_ylabel("Count")
        ax.set_xlim(0, 3000)

    # Row 2: bucket distributions
    for i, name in enumerate(names):
        ax = axes[1][i]
        counts = all_data[name]["counts"]
        total = counts.sum()
        pcts = counts / total * 100
        nb = len(counts)
        ax.bar(range(nb), pcts, color=colors[i], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax.axhline(y=100/nb, color="red", linestyle="--", linewidth=1, label=f"Balanced ({100/nb:.0f}%)")
        w = all_data[name]["width"]
        ms = all_data[name]["max_seq"]
        ax.set_title(f"{name}\nmax_seq={ms}, width={w:.0f}")
        ax.set_xlabel("Bucket")
        ax.set_ylabel("% of samples")
        ax.legend(fontsize=8)
        ax.set_ylim(0, min(100, max(pcts) * 1.3))

    plt.tight_layout()
    path = os.path.join(output_dir, "s3_vs_vllm_ltr.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpaca", required=True)
    parser.add_argument("--sharegpt", required=True)
    parser.add_argument("--lmsys", required=True)
    parser.add_argument("--tokenizer", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    from transformers import AutoTokenizer
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Load datasets
    sources = {
        "Alpaca (S³)": ("alpaca", args.alpaca),
        "ShareGPT": ("jsonl", args.sharegpt),
        "LMSYS": ("jsonl", args.lmsys),
    }

    configs = {
        "Alpaca (S³)": 2048,   # S³ max_seq
        "ShareGPT": 8192,      # vllm-ltr max_seq
        "LMSYS": 8192,
    }

    all_data = {}
    for name, (fmt, path) in sources.items():
        print(f"Loading {name}...")
        texts = load_alpaca(path) if fmt == "alpaca" else load_jsonl(path)
        print(f"  {len(texts)} samples")
        lengths = tokenize_lengths(texts, tok)
        max_seq = configs[name]
        counts, width = bucket_distribution(lengths, max_seq, 10)
        all_data[name] = {"lengths": lengths, "counts": counts, "width": width, "max_seq": max_seq}

    # Print stats
    print(f"\n{'='*65}")
    print(f"  Output Length Statistics")
    print(f"{'='*65}")
    for name in all_data:
        print_stats(name, all_data[name]["lengths"])

    print(f"\n{'='*65}")
    print(f"  10 Uniform Bucket Distribution (each dataset's own config)")
    print(f"{'='*65}")
    for name in all_data:
        d = all_data[name]
        print_buckets(name, d["counts"], d["width"], 10)

    # Also show ShareGPT/LMSYS what it would look like with S³ config
    print(f"\n{'='*65}")
    print(f"  Note: S³ also evaluated on NQ (77.1% acc) and Pile (65.6% acc),")
    print(f"  showing accuracy degrades on more diverse datasets even with")
    print(f"  their own max_seq=2048 setting.")
    print(f"{'='*65}")

    plot_comparison(all_data, args.output_dir)


if __name__ == "__main__":
    main()