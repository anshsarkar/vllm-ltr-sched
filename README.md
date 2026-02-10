# vllm-ltr-sched

Research project building upon [Efficient LLM Scheduling by Learning to Rank](https://arxiv.org/abs/2408.15792) (vllm-ltr).

## Quick Start (Docker)

```bash
# 1. Clone this repo
git clone https://github.com/anshsarkar/vllm-ltr-sched.git
cd vllm-ltr-sched

# 2. One-time instance setup (Docker, NVIDIA toolkit, nvtop)
bash scripts/setup_instance.sh
# Then: log out and back in (for docker group)

# 3. Set up your HuggingFace token
cp docker/.env.template docker/.env
# Edit docker/.env with your HF_TOKEN

# 4. Build Docker image (~15-20 min first time, cached after)
bash scripts/build_docker.sh

# 5. Start the dev container
bash scripts/start.sh

# 6. Inside the container — download data and login to HuggingFace
huggingface-cli login
bash scripts/download_data.sh

# 7. Inside the container — run 8B benchmarks
bash scripts/run_bench_8b_1gpu_sharegpt.sh
```

## Quick Start (Local / Conda)

```bash
# 1. Clone this repo
git clone https://github.com/anshsarkar/vllm-ltr-sched.git
cd vllm-ltr-sched

# 2. One-time instance setup (Docker, NVIDIA toolkit, nvtop)
bash scripts/setup_instance.sh

# 3. Set up conda env (installs conda, Python 3.10, all deps, HuggingFace CLI)
bash scripts/setup_conda_env.sh
# Then: huggingface-cli login

# 4. Download datasets and pre-trained models
conda activate vllm-ltr
bash scripts/download_data.sh

# 5. Run a single benchmark (quick test)
bash scripts/run_single_bench.sh -s fcfs -r 2        # FCFS baseline, rate 2
bash scripts/run_single_bench.sh -s opt-xxx -r 64    # LTR scheduler, rate 64

# Or run all 5 schedulers x 6 rates (~3-4 hours)
bash scripts/run_bench_8b_1gpu_sharegpt.sh
```

### run_single_bench.sh options

```bash
bash scripts/run_single_bench.sh -s <scheduler> -r <rate> -t <duration>

# -s  scheduler: fcfs, opt-xxx, tpt-class10-xxx, mlfq, PO
# -r  request rate (req/s), default: 2
# -t  duration (seconds), default: 60
```

## Citation

```bibtex
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```
