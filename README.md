# vllm-ltr-sched

A reproducibility and extension study of [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792) (Fu et al., 2024). This repository contains the code, data, and analysis for our ACM REP 2026 paper, which consumes the authors' artifact following Wonsil et al.'s five-stage reproducibility taxonomy.

> **Note:** The citation at the bottom of this README refers to the **original work by Fu et al.**, not this reproducibility study. Our study builds on their artifact and codebase.

## Repository Structure

```
vllm-ltr-sched/
├── vllm-ltr/                          # Authors' forked codebase (vLLM 0.4.1 + LTR scheduling)
│   ├── train/                         # Training code for predictors
│   │   ├── trainer.py                 # Uniform-width classifier trainer
│   │   └── trainer_percentile.py      # Percentile-balanced classifier trainer
│   └── benchmarks/                    # Serving benchmark harness
├── scripts/                           # Setup and benchmark runner scripts
│   ├── setup_instance.sh              # One-time instance setup (NVIDIA toolkit, etc.)
│   ├── setup_conda_env.sh             # Conda environment setup
│   ├── download_data.sh               # Download datasets and pre-trained models
├── rep/final_results/                 # All results from the reproducibility study
│   ├── training/
│   │   ├── scripts/                   # Training scripts (train_all.sh, train_pctl10_mse.sh)
│   │   ├── logs/                      # Raw training logs per model/dataset
│   │   └── metrics/                   # training_metrics.csv (parsed from logs)
│   ├── benchmarks/
│   │   ├── scripts/                   # Benchmark runner scripts for all schedulers
│   │   ├── logs/                      # Server and benchmark logs
│   │   └── results/                   # Raw JSON results and analysis outputs
│   │       ├── lmsys/                 # Run 1: LMSYS-Chat benchmark JSONs
│   │       ├── lmsys_test3/           # Run 2
│   │       ├── lmsys_test4/           # Run 3
│   │       ├── sharegpt/              # Run 1: ShareGPT benchmark JSONs
│   │       ├── sharegpt_test3/        # Run 2
│   │       ├── sharegpt_test4/        # Run 3
│   │       ├── raw_authors_data/      # Authors' original benchmark data
│   │       └── plots_for_paper/       # Generated figures and histograms
│   └── analysis/                      # Analysis and plotting scripts
│       ├── plot_benchmark_results.py  # Extract metrics from JSON to CSV
│       ├── parse_training_logs.py     # Parse training logs to CSV
│       ├── extract_class_distributions.py  # Extract class distributions from models
│       └── generate_distribution_histograms.py  # Generate inline histogram PDFs
└── data/                              # Datasets and pre-trained models (not in git)
```

## Pre-computed Results

All results from our study are committed to the repository. You do not need to rerun anything to inspect them.

**Benchmark metrics (CSV):**
- `rep/final_results/benchmarks/results/metrics_lmsys.csv` (Run 1)
- `rep/final_results/benchmarks/results/metrics_lmsys_test3.csv` (Run 2)
- `rep/final_results/benchmarks/results/metrics_lmsys_test4.csv` (Run 3)
- Same pattern for `sharegpt` variants
- `rep/final_results/benchmarks/results/authors_metrics_lmsys.csv` (authors' reported values)
- `rep/final_results/benchmarks/results/authors_metrics_sharegpt.csv`

**Training metrics:**
- `rep/final_results/training/metrics/training_metrics.csv` (loss, Kendall's tau, accuracy per epoch)

**Class distributions:**
- `rep/final_results/analysis/class_distributions.csv` (per-bucket sample counts for each classifier)

**Prediction quality:**
- `rep/final_results/benchmarks/results/plots_for_paper/prediction_quality_summary.csv`
- `rep/final_results/benchmarks/results/plots_for_paper/prediction_quality_per_rate.csv`

**Version drift:**
- `rep/final_results/analysis/version_drift_raw.csv` (resolved versions for 20 packages across 5 PyPI snapshot dates)
- `rep/final_results/benchmarks/results/plots_for_paper/version_drift.pdf` (heatmap figure for Appendix B)

Versions were collected using [`pypi-timemachine`](https://github.com/astrofrog/pypi-timemachine), which serves a local PyPI proxy frozen to a given date. We resolved the authors' unpinned dependency specifications (`requirements-common.txt`, `requirements-cuda.txt`) against PyPI snapshots at each date to determine what versions pip would install.

**Paper figures:**
- `rep/final_results/benchmarks/results/plots_for_paper/paper_figure_lmsys.pdf` (3-subplot figure with SE bands in subplot (b))
- `rep/final_results/benchmarks/results/plots_for_paper/paper_figure_sharegpt.pdf` (3-subplot figure with SE bands in subplot (b))
- `rep/final_results/benchmarks/results/plots_for_paper/version_drift.pdf` (dependency version drift heatmap, Appendix B)
- `rep/final_results/benchmarks/results/plots_for_paper/histograms/` (inline distribution histograms)

**Key scripts:**

| Script | Description |
|--------|-------------|
| `scripts/setup_instance.sh` | One-time machine setup (NVIDIA toolkit, nvtop) |
| `scripts/setup_conda_env.sh` | Creates `vllm-ltr` conda env with Python 3.10 and all dependencies |
| `scripts/download_data.sh` | Downloads datasets and authors' pre-trained model checkpoints from HuggingFace |
| `rep/final_results/training/scripts/train_all.sh` | Trains class82, pctl10, and width10 classifiers on both datasets |
| `rep/final_results/training/scripts/train_pctl10_mse.sh` | Trains the percentile MSE variant |
| `rep/final_results/benchmarks/scripts/run_bench_all_v3.sh` | Runs all 9 schedulers on both datasets (the main benchmark script) |
| `rep/final_results/analysis/parse_training_logs.py` | Parses raw training logs into `training_metrics.csv` |
| `rep/final_results/analysis/plot_benchmark_results.py` | Extracts mean normalized latency per scheduler/rate from raw benchmark JSONs to CSV |
| `rep/final_results/analysis/extract_class_distributions.py` | Extracts class distributions from training data for each classifier |
| `rep/final_results/analysis/generate_distribution_histograms.py` | Generates inline histogram PDFs for the paper table |
| `rep/final_results/benchmarks/results/plot_paper_figures.py` | Generates the main 3-subplot comparison figures (authors vs. ours vs. variants) |
| `rep/final_results/benchmarks/results/plot_confidence_intervals.py` | Generates confidence interval plots (mean with 95% CI across 3 runs) |
| `rep/final_results/benchmarks/results/compute_prediction_quality.py` | Computes Kendall's tau, accuracy, and other prediction quality metrics |
| `rep/final_results/analysis/plot_version_drift.py` | Generates the version drift heatmap from `version_drift_raw.csv` |

## Quick Start (Conda)

**Prerequisites:** NVIDIA GPU with CUDA support, conda.

```bash
# 1. Clone this repo
git clone https://github.com/anshsarkar/vllm-ltr-sched.git
cd vllm-ltr-sched

# 2. One-time instance setup (NVIDIA toolkit, nvtop)
bash scripts/setup_instance.sh

# 3. Set up conda env (Python 3.10, all deps)
bash scripts/setup_conda_env.sh
# Then: huggingface-cli login

# 4. Download datasets and pre-trained models
conda activate vllm-ltr
bash scripts/download_data.sh

## Reproducing from Scratch

The full pipeline from training through figure generation. All commands assume you are in the repo root with the `vllm-ltr` conda environment activated.

### Step 1: Train classifier variants

```bash
# Train class82, pctl10, and width10 classifiers on both datasets
bash rep/final_results/training/scripts/train_all.sh

# Train the percentile MSE variant
bash rep/final_results/training/scripts/train_pctl10_mse.sh
```

Training logs are saved to `rep/final_results/training/logs/`.

### Step 2: Parse training metrics

```bash
python rep/final_results/analysis/parse_training_logs.py
# Output: rep/final_results/training/metrics/training_metrics.csv
```

### Step 3: Run benchmarks

We run 3 independent trials to compute standard error. Each run takes ~6-8 hours on a single A100 GPU (9 schedulers x 6 rates x 2 datasets).

**Important:** Before each run, you must update the output directory in two places:
1. `rep/final_results/benchmarks/scripts/run_bench_all_v3.sh` (lines that set `mkdir -p` and log paths)
2. `vllm-ltr/benchmarks/bench-final-lmsys-mruns.sh` and `bench-final-sharegpt-mruns.sh` (the `--result-dir` flag in every benchmark command)

Change the directory suffix to match the run number (`lmsys` / `lmsys_test3` / `lmsys_test4`, same for `sharegpt`).

```bash
# Run 1: edit scripts to use lmsys/ and sharegpt/ as output dirs
bash rep/final_results/benchmarks/scripts/run_bench_all_v3.sh

# Run 2: edit scripts to use lmsys_test3/ and sharegpt_test3/
bash rep/final_results/benchmarks/scripts/run_bench_all_v3.sh

# Run 3: edit scripts to use lmsys_test4/ and sharegpt_test4/
bash rep/final_results/benchmarks/scripts/run_bench_all_v3.sh
```

Raw JSON results are saved per scheduler and request rate in `rep/final_results/benchmarks/results/<dataset>/`.

### Step 4: Generate metrics CSVs

```bash
# Extract mean normalized latency per scheduler/rate from raw JSONs

# Run 1
python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/lmsys \
    -o rep/final_results/benchmarks/results/metrics_lmsys.csv --no-plot

python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/sharegpt \
    -o rep/final_results/benchmarks/results/metrics_sharegpt.csv --no-plot

# Run 2
python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/lmsys_test3 \
    -o rep/final_results/benchmarks/results/metrics_lmsys_test3.csv --no-plot

python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/sharegpt_test3 \
    -o rep/final_results/benchmarks/results/metrics_sharegpt_test3.csv --no-plot

# Run 3
python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/lmsys_test4 \
    -o rep/final_results/benchmarks/results/metrics_lmsys_test4.csv --no-plot

python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/sharegpt_test4 \
    -o rep/final_results/benchmarks/results/metrics_sharegpt_test4.csv --no-plot
```

### Step 5: Generate paper figures

```bash
# Main 3-subplot comparison figures (authors vs. ours vs. variants)
python rep/final_results/benchmarks/results/plot_paper_figures.py

# Confidence interval plots (mean +/- 95% CI across 3 runs)
python rep/final_results/benchmarks/results/plot_confidence_intervals.py

# Prediction quality metrics (Kendall's tau, accuracy)
python rep/final_results/benchmarks/results/compute_prediction_quality.py

# Class distribution histograms for the table
python rep/final_results/analysis/extract_class_distributions.py
python rep/final_results/analysis/generate_distribution_histograms.py
```

All outputs go to `rep/final_results/benchmarks/results/plots_for_paper/`.

## Training a New Predictor

The framework supports two predictor types, both using OPT-125M as the backbone. Trainers are in `vllm-ltr/train/`:

- **`trainer.py`** trains uniform-width classifiers. The `--label-group-size` flag controls bucket width (e.g., 100 gives ~82 classes, 10 gives ~820 classes).
- **`trainer_percentile.py`** trains percentile-balanced classifiers. The `--num-classes` flag controls the number of buckets.

Training data (JSONL traces with prompt and output length) is downloaded by `scripts/download_data.sh`. See `rep/final_results/training/scripts/train_all.sh` for complete training examples.

Both trainers save checkpoints and a `usage_config.json` to `vllm-ltr/train/MODEL/results/<run-id>/`. To benchmark a trained model, launch the vLLM server with `--schedule-type <name> --prefill-predictor-model-config <path-to-usage_config.json>`, then run the benchmark client. See `vllm-ltr/benchmarks/bench-final-lmsys-mruns.sh` for full server and client command examples.

## Citations

This repository is a reproducibility study. The citation below is for the **original work** by Fu et al., whose artifact we consume and extend.

```bibtex
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```
