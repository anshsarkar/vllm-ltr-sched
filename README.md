# vllm-ltr-sched

This repository accompanies our ACM REP 2026 paper, *"It Works, But Why? A Case Study of Artifact Consumption in Machine Learning Systems"*. We apply Wonsil et al.'s five-stage reproducibility taxonomy to the artifact for [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792) (Fu et al., 2024), going beyond verifying reported results to critically examine the underlying design decisions.

Our key finding is that the reported advantage of ranking over classification for LLM request scheduling is largely attributable to the classifier design (fixed-width bucketing over a skewed distribution) rather than a fundamental limitation of classification. By training classifiers with percentile-balanced buckets and ordinal-aware loss functions, we close most of the performance gap with ranking.

The repository contains:
- Our reproduction of the original benchmark results across 9 scheduling policies on two datasets (LMSYS-Chat-1M and ShareGPT), with 3 independent trials
- Training code and results for alternative classifier designs that we developed during the Comprehend stage
- All pre-computed metrics, figures, and analysis scripts used in the paper
- The authors' forked vLLM 0.4.1 codebase with LTR scheduling (forked from [hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr) at commit [`13bbf6f`](https://github.com/hao-ai-lab/vllm-ltr/commit/13bbf6ff3dab661791d41362551b089e5f77c91c))

## Repository Structure

```
vllm-ltr-sched/
├── vllm-ltr/                              # Authors' forked vLLM 0.4.1 + LTR scheduling
│   ├── train/                             # Predictor training code
│   │   ├── trainer.py                     # * Uniform-width classifier trainer
│   │   └── trainer_percentile.py          # * Percentile-balanced classifier trainer
│   ├── benchmarks/                        # Serving benchmark harness
│   └── vllm/core/scheduler.py             # * Scheduling policy entry point
├── scripts/                               # Environment and data setup
│   ├── setup_instance.sh                  # One-time NVIDIA toolkit setup
│   ├── setup_conda_env.sh                 # Conda environment setup
│   └── download_data.sh                   # Download datasets and models from HuggingFace
├── rep/final_results/                     # Our study: training, benchmarks, and analysis
│   ├── training/
│   │   ├── scripts/                       # * train_all.sh, train_pctl10_mse.sh
│   │   ├── logs/                          # Raw training logs
│   │   └── metrics/                       # training_metrics.csv
│   ├── benchmarks/
│   │   ├── scripts/                       # * run_bench_all_v3.sh (main benchmark runner)
│   │   └── results/                       # Per-run .pt files, metrics CSVs, and figures
│   │       ├── {lmsys,sharegpt}[_test3|_test4]/  # Raw benchmark data (3 runs per dataset)
│   │       ├── metrics_*.csv              # Extracted per-scheduler latency metrics
│   │       ├── authors_metrics_*.csv      # Authors' reported values
│   │       └── plots_for_paper/           # * Generated figures, histograms, prediction quality
│   └── analysis/                          # Analysis and plotting scripts
│       ├── plot_benchmark_results.py      # * .pt benchmark data → metrics CSV
│       ├── plot_version_drift.py          # * Version drift heatmap
│       └── version_drift_raw.csv          # Raw version drift data
└── data/                                  # Datasets and pretrained models (not in git, see download_data.sh)
```

## Recreating Paper Figures

All raw data (benchmark `.pt` files, training logs, version drift data) and pre-computed metrics CSVs are committed to the repository. You can regenerate everything without a GPU. Only `matplotlib`, `numpy`, `pandas`, and `scipy` are needed.

### Regenerating metrics CSVs from raw data

The metrics CSVs are already committed, but you can regenerate them from the raw benchmark and training data:

```bash
# Benchmark metrics: extract mean normalized latency per scheduler/rate from raw .pt files
# Repeat for each run and dataset (lmsys, lmsys_test3, lmsys_test4, sharegpt, sharegpt_test3, sharegpt_test4)
python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/lmsys \
    -o rep/final_results/benchmarks/results/metrics_lmsys.csv --no-plot

# Training metrics: parse raw training logs into CSV
python rep/final_results/analysis/parse_training_logs.py
# Output: rep/final_results/training/metrics/training_metrics.csv
```

### Generating figures

```bash
# Main comparison figures (Figures 1 and 2: authors vs. ours vs. classifier variants)
python rep/final_results/benchmarks/results/plot_paper_figures.py

# Confidence interval plots (mean +/- 95% CI across 3 runs)
python rep/final_results/benchmarks/results/plot_confidence_intervals.py

# Prediction quality metrics (Kendall's tau, accuracy)
python rep/final_results/benchmarks/results/compute_prediction_quality.py

# Class distribution histograms (inline figures in Table 1)
# Uses the committed class_distributions.csv (to regenerate this CSV from raw data, see "Reproducing from Scratch")
python rep/final_results/analysis/generate_distribution_histograms.py

# Version drift heatmap (Appendix B)
python rep/final_results/analysis/plot_version_drift.py
```

All outputs go to `rep/final_results/benchmarks/results/plots_for_paper/`.

## Reproducing from Scratch

If you want to rerun the full pipeline (training, benchmarks, and analysis), follow the steps below. All commands assume you are in the repo root with the `vllm-ltr` conda environment activated.

### Environment Setup

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
```

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

Raw `.pt` results are saved per scheduler and request rate in `rep/final_results/benchmarks/results/<dataset>/`.

### Step 4: Generate metrics CSVs

```bash
# Extract mean normalized latency per scheduler/rate from raw .pt files
# Repeat for each run and dataset:
python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/lmsys \
    -o rep/final_results/benchmarks/results/metrics_lmsys.csv --no-plot

python rep/final_results/analysis/plot_benchmark_results.py \
    rep/final_results/benchmarks/results/sharegpt \
    -o rep/final_results/benchmarks/results/metrics_sharegpt.csv --no-plot

# Same for _test3 and _test4 variants
```

### Step 5: Generate class distributions

This requires the JSONL training data downloaded in the environment setup step.

```bash
cd vllm-ltr/train
python ../../rep/final_results/analysis/extract_class_distributions.py \
    --lmsys-dataset jsonfiles/lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c20000:30000-rFalse.jsonl \
    --sharegpt-dataset jsonfiles/llama3-8b-sharegpt-train-t1-s0-8192.jsonl \
    --output ../../rep/final_results/analysis/class_distributions.csv
cd ../..
```

### Step 6: Generate paper figures

See the [Recreating Paper Figures](#recreating-paper-figures) section above. All plotting scripts work on the metrics CSVs and class distributions generated in the previous steps.

## Training a New Predictor

The framework supports two predictor types, both using OPT-125M as the backbone. Trainers are in `vllm-ltr/train/`:

- **`trainer.py`** trains uniform-width classifiers. The `--label-group-size` flag controls bucket width (e.g., 100 gives ~82 classes, 10 gives ~820 classes).
- **`trainer_percentile.py`** trains percentile-balanced classifiers. The `--num-classes` flag controls the number of buckets.

Training data (JSONL traces with prompt and output length) is downloaded by `scripts/download_data.sh`. See `rep/final_results/training/scripts/train_all.sh` for complete training examples.

Both trainers save checkpoints and a `usage_config.json` to `vllm-ltr/train/MODEL/results/<run-id>/`. To benchmark a trained model, launch the vLLM server with `--schedule-type <name> --prefill-predictor-model-config <path-to-usage_config.json>`, then run the benchmark client. See `vllm-ltr/benchmarks/bench-final-lmsys-mruns.sh` for full server and client command examples.

## Citations

If you use this repository, please cite our paper:

```bibtex
@inproceedings{SarkarFund2026ItWorks,
  title={It Works, But Why? A Case Study of Artifact Consumption in Machine Learning Systems},
  author={Sarkar, Ansh and Fund, Fraida},
  booktitle={Proceedings of the ACM Conference on Reproducibility and Replicability (ACM REP '26)},
  year={2026},
  doi={10.1145/3820002.3828597}
}
```

The original work whose artifact we consume and extend:

```bibtex
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```
