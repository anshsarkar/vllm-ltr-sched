#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== vllm-ltr 8B Final Benchmark Runner (v3 — LMSYS + ShareGPT) ==="

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
    "PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl" \
    "llama3-8b-sharegpt-test-t1-s0-8192.jsonl" \
    "PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found!"
    else
        echo "  OK: $dataset"
    fi
done

# ---- Verify required model configs (authors') ----
echo ""
echo "Checking authors' model configs..."
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] && echo "  OK: $config" || echo "  WARNING: $config not found!"
done

# ---- Verify required model configs (ours) ----
echo ""
echo "Checking our model configs..."
for config in \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket100-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket10-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket100-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-mse-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket10-b4/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] && echo "  OK: $config" || echo "  WARNING: $config not found (run training first)!"
done

# ---- Run LMSYS ----
mkdir -p "$PROJECT_ROOT/rep/final_results/benchmarks/results/lmsys_test3"

echo ""
echo "=========================================="
echo "  LMSYS-Chat: 9 schedulers x 6 request rates = 54 runs"
echo "  (fcfs, oracle-srtf, opt-xxx, class10, class82, pctl10, pctl10-mse, width10, mlfq)"
echo "  Using benchmark_serving_real_with_metrics.py"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-final-lmsys-mruns.sh" 2>&1 | tee "$PROJECT_ROOT/rep/final_results/benchmarks/results/lmsys_test3/bench_run.log"

echo ""
echo "=== LMSYS benchmarks complete ==="
echo ""

# ---- Run ShareGPT ----
mkdir -p "$PROJECT_ROOT/rep/final_results/benchmarks/results/sharegpt_test3"

echo ""
echo "=========================================="
echo "  ShareGPT: 9 schedulers x 6 request rates = 54 runs"
echo "  (fcfs, oracle-srtf, opt-xxx, class10, class82, pctl10, pctl10-mse, width10, mlfq)"
echo "  Using benchmark_serving_real_with_metrics.py"
echo "=========================================="
echo ""

bash "$BENCH_DIR/bench-final-sharegpt-mruns.sh" 2>&1 | tee "$PROJECT_ROOT/rep/final_results/benchmarks/results/sharegpt_test3/bench_run.log"

echo ""
echo "=========================================="
echo "=== All benchmarks (v3) complete ==="
echo "LMSYS results:   $PROJECT_ROOT/rep/final_results/benchmarks/results/lmsys_test3/"
echo "ShareGPT results: $PROJECT_ROOT/rep/final_results/benchmarks/results/sharegpt_test3/"
echo "=========================================="
