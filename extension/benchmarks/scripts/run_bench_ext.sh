#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== Extension Benchmark Runner (8 schedulers x 6 rates x 2 datasets) ==="

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

# ---- Verify required datasets ----
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

# ---- Verify authors' model configs ----
echo ""
echo "Checking authors' model configs..."
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] && echo "  OK: $config" || echo "  WARNING: $config not found!"
done

# ---- Verify our fresh -ext model configs ----
echo ""
echo "Checking our -ext model configs..."
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket100-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket10-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket100-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-mse-b4-ext/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket10-b4-ext/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] && echo "  OK: $config" || echo "  WARNING: $config not found (run training first)!"
done

# ---- Run LMSYS ----
mkdir -p "$PROJECT_ROOT/extension/benchmarks/results/lmsys"

echo ""
echo "=========================================="
echo "  LMSYS-Chat: 8 schedulers x 6 request rates = 48 runs"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-ext-lmsys.sh" 2>&1 | tee "$PROJECT_ROOT/extension/benchmarks/results/lmsys/bench_run.log"

echo ""
echo "=== LMSYS benchmarks complete ==="
echo ""

# ---- Run ShareGPT ----
mkdir -p "$PROJECT_ROOT/extension/benchmarks/results/sharegpt"

echo ""
echo "=========================================="
echo "  ShareGPT: 8 schedulers x 6 request rates = 48 runs"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-ext-sharegpt.sh" 2>&1 | tee "$PROJECT_ROOT/extension/benchmarks/results/sharegpt/bench_run.log"

echo ""
echo "=========================================="
echo "=== All extension benchmarks complete ==="
echo "LMSYS results:    $PROJECT_ROOT/extension/benchmarks/results/lmsys/"
echo "ShareGPT results: $PROJECT_ROOT/extension/benchmarks/results/sharegpt/"
echo "=========================================="
