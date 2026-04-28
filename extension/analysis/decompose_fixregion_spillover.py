#!/usr/bin/env python3

import os
import json
import numpy as np
import torch
import csv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_ROOT = os.path.join(REPO_ROOT, "extension", "benchmarks", "results")
CONFIG_DIR = os.path.join(REPO_ROOT, "extension", "training", "configs")
OUT_DIR = os.path.join(REPO_ROOT, "extension", "analysis", "fixregion_pctl10mse_analysis")

IDX_NLATENCIES = 3
IDX_OUTPUT_LENS = 4
IDX_INPUT_LENS = 5

RATE = 64.0
NUM_CLASSES = 10

DATASETS = {
    "lmsys": {
        "static_dir": os.path.join(RESULTS_ROOT, "lmsys"),
        "fix_dir": os.path.join(RESULTS_ROOT, "fixregion_pctl10mse_lmsys"),
        "boundaries_file": os.path.join(CONFIG_DIR, "pctl10_mse_lmsys_boundaries.json"),
    },
    "sharegpt": {
        "static_dir": os.path.join(RESULTS_ROOT, "sharegpt"),
        "fix_dir": os.path.join(RESULTS_ROOT, "fixregion_pctl10mse_sharegpt"),
        "boundaries_file": os.path.join(CONFIG_DIR, "pctl10_mse_sharegpt_boundaries.json"),
    },
}

FIXREGION_VARIANTS = [
    "fixlong3-tpt-pctl10-mse-xxx",
    "fixshort3-tpt-pctl10-mse-xxx",
    "fixlong5-tpt-pctl10-mse-xxx",
    "fixshort5-tpt-pctl10-mse-xxx",
]

VARIANT_LABELS = {
    "fixshort3-tpt-pctl10-mse-xxx": "Fix Short 3",
    "fixshort5-tpt-pctl10-mse-xxx": "Fix Short 5",
    "fixlong3-tpt-pctl10-mse-xxx": "Fix Long 3",
    "fixlong5-tpt-pctl10-mse-xxx": "Fix Long 5",
}

# Regions: class 0 = longest, class 9 = shortest
SHORT_CLASSES = {7, 8, 9}
LONG_CLASSES = {0, 1, 2}
MIDDLE_CLASSES = {3, 4, 5, 6}

REGIONS = [
    (SHORT_CLASSES, "short_3"),
    (LONG_CLASSES, "long_3"),
    (MIDDLE_CLASSES, "middle_4"),
]


def pt_filename(scheduler):
    return f"latency-{scheduler}-Meta-Llama-3-8B-Instruct-p0-r{RATE}-c1.0-t60.0-o-1.pt"


def load_pt(folder, scheduler):
    path = os.path.join(folder, pt_filename(scheduler))
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "nlatencies": np.array(data[IDX_NLATENCIES]),
        "output_lens": np.array(data[IDX_OUTPUT_LENS]),
        "input_lens": np.array(data[IDX_INPUT_LENS]),
    }


def output_len_to_class(output_lens, boundaries):
    num_classes = len(boundaries) + 1
    raw_bin = np.searchsorted(boundaries, output_lens, side="right")
    return (num_classes - 1) - raw_bin


def run():
    all_rows = []

    for ds_name, ds_config in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name.upper()}")
        print(f"{'='*70}")

        # Load baselines from static results
        sjf_data = load_pt(ds_config["static_dir"], "sjf")
        cls_data = load_pt(ds_config["static_dir"], "tpt-pctl10-mse-xxx")

        with open(ds_config["boundaries_file"]) as f:
            bounds_info = json.load(f)
        boundaries = bounds_info["boundaries"]

        # True class from SJF output lengths
        true_classes = output_len_to_class(sjf_data["output_lens"], boundaries)

        sjf_nlats = sjf_data["nlatencies"]
        cls_nlats = cls_data["nlatencies"]

        overall_cls = float(np.mean(cls_nlats))
        overall_sjf = float(np.mean(sjf_nlats))
        overall_gap = overall_cls - overall_sjf
        print(f"  Classifier mean nlatency: {overall_cls:.4f}")
        print(f"  SJF mean nlatency:        {overall_sjf:.4f}")
        print(f"  Overall gap:              {overall_gap:.4f}")

        for variant in FIXREGION_VARIANTS:
            fix_data = load_pt(ds_config["fix_dir"], variant)
            fix_nlats = fix_data["nlatencies"]

            # Verify same number of requests
            assert len(fix_nlats) == len(sjf_nlats), \
                f"Length mismatch: {variant} has {len(fix_nlats)}, SJF has {len(sjf_nlats)}"
            # Verify same request ordering via input lengths
            assert np.array_equal(sjf_data["input_lens"], fix_data["input_lens"]), \
                f"Input lens mismatch for {variant} — requests not aligned"

            fix_mean = float(np.mean(fix_nlats))
            total_improv = overall_cls - fix_mean
            gap_recovery = total_improv / overall_gap * 100 if overall_gap > 0 else 0
            total_improv_sum = float(np.sum(cls_nlats - fix_nlats))

            label = VARIANT_LABELS[variant]
            print(f"\n  --- {label} (gap recovery: {gap_recovery:.1f}%) ---")
            print(f"  {'Region':<10} {'N':>5} {'SJF':>8} {'Cls':>8} {'Fix':>8} "
                  f"{'Cls->Fix':>8} {'Fix-SJF':>8} {'ImpShare':>8}")

            for region_classes, region_name in REGIONS:
                mask = np.array([tc in region_classes for tc in true_classes])
                n = mask.sum()

                sjf_r = float(np.mean(sjf_nlats[mask]))
                cls_r = float(np.mean(cls_nlats[mask]))
                fix_r = float(np.mean(fix_nlats[mask]))

                # Improvement: classifier -> fixregion (positive = fixregion is better)
                improv = cls_r - fix_r
                # Remaining gap to SJF (positive = still worse than SJF)
                remaining = fix_r - sjf_r

                # Share of total improvement from this region
                region_improv_sum = float(np.sum(cls_nlats[mask] - fix_nlats[mask]))
                improv_share = region_improv_sum / total_improv_sum * 100 \
                    if total_improv_sum != 0 else 0

                print(f"  {region_name:<10} {n:>5} {sjf_r:>8.4f} {cls_r:>8.4f} {fix_r:>8.4f} "
                      f"{improv:>+8.4f} {remaining:>+8.4f} {improv_share:>7.1f}%")

                all_rows.append({
                    "dataset": ds_name,
                    "variant": label,
                    "region": region_name,
                    "n_requests": n,
                    "sjf_mean_nlat": round(sjf_r, 6),
                    "cls_mean_nlat": round(cls_r, 6),
                    "fix_mean_nlat": round(fix_r, 6),
                    "improvement_cls_to_fix": round(improv, 6),
                    "remaining_gap_to_sjf": round(remaining, 6),
                    "improvement_share_pct": round(improv_share, 2),
                })

    # Write CSV
    csv_path = os.path.join(OUT_DIR, "fixregion_spillover_decomposition.csv")
    fieldnames = [
        "dataset", "variant", "region", "n_requests",
        "sjf_mean_nlat", "cls_mean_nlat", "fix_mean_nlat",
        "improvement_cls_to_fix", "remaining_gap_to_sjf", "improvement_share_pct",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n  CSV written to {csv_path}")


if __name__ == "__main__":
    run()
