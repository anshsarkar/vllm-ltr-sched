#!/usr/bin/env python3
import argparse
import csv
import os
import re
import sys


def parse_log(filepath):
    rows = []
    current_epoch = None
    current_loss = None
    current_tau = None
    current_pvalue = None
    current_acc = None
    current_exact_acc = None
    current_within1_acc = None

    def flush():
        if current_epoch is not None:
            rows.append({
                "epoch": current_epoch,
                "loss": current_loss,
                "kendall_tau": current_tau,
                "p_value": current_pvalue,
                "train_acc": current_acc,
                "exact_acc": current_exact_acc,
                "within1_acc": current_within1_acc,
            })

    with open(filepath) as f:
        for line in f:
            line = line.strip()

            m = re.match(r"Epoch (\d+), Loss: ([\d.]+)", line)
            if m:
                flush()
                current_epoch = int(m.group(1))
                current_loss = float(m.group(2))
                current_tau = current_pvalue = current_acc = None
                current_exact_acc = current_within1_acc = None
                continue

            m = re.match(r"Kendall's Tau: ([\d.e+-]+), p-value: ([\d.e+-]+)", line)
            if m:
                current_tau = float(m.group(1))
                current_pvalue = float(m.group(2))
                continue

            m = re.match(r"acc:\s+([\d.]+)", line)
            if m:
                current_acc = float(m.group(1))
                continue

            m = re.match(r"Exact accuracy: ([\d.]+)", line)
            if m:
                current_exact_acc = float(m.group(1))
                continue

            m = re.match(r"Within-1 accuracy: ([\d.]+)", line)
            if m:
                current_within1_acc = float(m.group(1))
                continue

    flush()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "log_dir", nargs="?",
        default=os.path.join(os.path.dirname(__file__), "..", "training", "logs"),
    )
    parser.add_argument("-o", "--output", default=None)
    args = parser.parse_args()

    log_dir = os.path.abspath(args.log_dir)
    if not os.path.isdir(log_dir):
        print(f"Error: '{log_dir}' is not a directory")
        sys.exit(1)

    metrics_dir = os.path.join(os.path.dirname(__file__), "..", "training", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    output = args.output or os.path.join(metrics_dir, "training_metrics.csv")

    all_rows = []
    log_files = sorted(f for f in os.listdir(log_dir) if f.endswith(".log"))

    if not log_files:
        print(f"No .log files found in {log_dir}")
        sys.exit(1)

    for fname in log_files:
        base = fname.replace("train_", "").replace(".log", "")
        parts = base.split("_", 1)
        dataset, model = parts if len(parts) == 2 else ("unknown", base)

        rows = parse_log(os.path.join(log_dir, fname))
        for row in rows:
            row["model"] = model
            row["dataset"] = dataset
            row["log_file"] = fname
            all_rows.append(row)

        print(f"  {fname}: {len(rows)} epochs parsed")

    if not all_rows:
        print("No metrics extracted from any log file.")
        sys.exit(1)

    fieldnames = [
        "model", "dataset", "epoch", "loss",
        "kendall_tau", "p_value",
        "train_acc", "exact_acc", "within1_acc",
        "log_file",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} rows to {output}")


if __name__ == "__main__":
    main()