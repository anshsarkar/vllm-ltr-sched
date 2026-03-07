#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== vllm-ltr 8B LMSYS Rep Benchmark Runner ==="

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

# Verify required dataset
LMSYS_DATASET="lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl"
if [ ! -f "$BENCH_DIR/$LMSYS_DATASET" ]; then
    echo "  WARNING: $LMSYS_DATASET not found!"
fi

# Verify required model configs (LMSYS-specific predictors)
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket100-b4/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] || echo "  WARNING: $config not found!"
done

# ---- Run ----
mkdir -p "$BENCH_DIR/RESULTS"

echo ""
echo "  4 schedulers x 6 request rates = 24 runs"
echo ""

bash "$BENCH_DIR/bench-rep-lmsys.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_rep_lmsys_run.log"

echo ""
echo "=== LMSYS Rep Benchmarks complete ==="
echo "Results: $BENCH_DIR/RESULTS/"
