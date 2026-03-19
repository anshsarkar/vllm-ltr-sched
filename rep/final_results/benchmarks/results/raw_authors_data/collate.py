#!/usr/bin/env python3
"""Collate raw WebPlotDigitizer CSVs into a single authors_metrics.csv per dataset."""

import csv
import os
import glob

COLOR_TO_SCHEDULER = {
    "Blue": "fcfs",
    "Yellow": "mlfq",
    "Green": "srtf-PO-X",
    "Red": "tpt-class10-xxx",
    "Purple": "opt-xxx",
}

EXPECTED_RATES = [2, 4, 8, 16, 32, 64]


def round_to_nearest_rate(x):
    return min(EXPECTED_RATES, key=lambda r: abs(r - x))


def process_dataset(dataset_dir, output_path):
    rows = []
    for csv_file in sorted(glob.glob(os.path.join(dataset_dir, "*.csv"))):
        color = os.path.splitext(os.path.basename(csv_file))[0]
        scheduler = COLOR_TO_SCHEDULER.get(color)
        if scheduler is None:
            print(f"  Warning: unknown color '{color}', skipping")
            continue

        with open(csv_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                raw_rate = float(parts[0].strip())
                latency_s = float(parts[1].strip())
                rate = round_to_nearest_rate(raw_rate)
                rows.append({
                    "scheduler": scheduler,
                    "request_rate": rate,
                    "mean_nlatency_s": round(latency_s, 6),
                })

    rows.sort(key=lambda r: (r["scheduler"], r["request_rate"]))

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scheduler", "request_rate", "mean_nlatency_s"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Wrote {len(rows)} rows to {output_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(base_dir)  # benchmarks/results/

    for dataset in ["lmsys", "sharegpt"]:
        dataset_dir = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_dir):
            print(f"  Skipping {dataset} (directory not found)")
            continue
        print(f"Processing {dataset}...")
        output_path = os.path.join(output_dir, f"authors_metrics_{dataset}.csv")
        process_dataset(dataset_dir, output_path)


if __name__ == "__main__":
    main()