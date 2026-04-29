#!/usr/bin/env bash

# Results land in: extension/loss_experiments/benchmarks/results/{lmsys,sharegpt}/

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== Loss Experiments Benchmark (3 models x 2 rates x 2 datasets = 12 runs) ==="

# ---- Check prerequisites ----
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"

# ---- Setup data symlinks ----
source "$PROJECT_ROOT/scripts/setup_bench_data.sh"

# ---- Symlink trained models from train dir into benchmarks ----
TRAIN_MODEL_DIR="$PROJECT_ROOT/vllm-ltr/train/MODEL/results"
if [ -d "$TRAIN_MODEL_DIR" ]; then
    mkdir -p "$BENCH_DIR/MODEL/results"
    for d in "$TRAIN_MODEL_DIR"/*/; do
        [ -d "$d" ] || continue
        base=$(basename "$d")
        [ -e "$BENCH_DIR/MODEL/results/$base" ] || ln -sf "$d" "$BENCH_DIR/MODEL/results/$base"
    done
fi

# ---- Verify B4 model configs ----
echo ""
echo "Checking B4 model configs..."
MODELS_OK=true
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-costsensitive-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-costsensitive-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-coral-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-coral-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-ranking-listmle-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-ranking-listmle-b4-ext/usage_config.json"; do
    if [ -f "$BENCH_DIR/$config" ]; then
        echo "  OK: $config"
    else
        echo "  MISSING: $config"
        MODELS_OK=false
    fi
done

if [ "$MODELS_OK" = false ]; then
    echo ""
    echo "ERROR: Some model configs not found. Run training first:"
    echo "  bash extension/loss_experiments/train/train_all_loss_experiments.sh"
    exit 1
fi

# ---- Verify datasets ----
echo ""
echo "Checking datasets..."
for dataset in \
    "lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl" \
    "llama3-8b-sharegpt-test-t1-s0-8192.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found!"
    else
        echo "  OK: $dataset"
    fi
done

# ---- Run LMSYS ----
mkdir -p "$PROJECT_ROOT/extension/loss_experiments/benchmarks/results/lmsys"

echo ""
echo "=========================================="
echo "  LMSYS: 3 models x 2 rates = 6 runs"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-loss-experiment-v1-lmsys.sh" 2>&1 | tee "$PROJECT_ROOT/extension/loss_experiments/benchmarks/results/lmsys/bench_run.log"

echo ""
echo "=== LMSYS benchmarks complete ==="
echo ""

# ---- Run ShareGPT ----
mkdir -p "$PROJECT_ROOT/extension/loss_experiments/benchmarks/results/sharegpt"

echo ""
echo "=========================================="
echo "  ShareGPT: 3 models x 2 rates = 6 runs"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-loss-experiment-v1-sharegpt.sh" 2>&1 | tee "$PROJECT_ROOT/extension/loss_experiments/benchmarks/results/sharegpt/bench_run.log"

echo ""
echo "=========================================="
echo "=== All Loss Experiment benchmarks complete ==="
echo "LMSYS results:    $PROJECT_ROOT/extension/loss_experiments/benchmarks/results/lmsys/"
echo "ShareGPT results: $PROJECT_ROOT/extension/loss_experiments/benchmarks/results/sharegpt/"
echo "=========================================="
