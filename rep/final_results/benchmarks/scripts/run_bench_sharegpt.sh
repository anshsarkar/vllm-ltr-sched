#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== vllm-ltr 8B ShareGPT Final Benchmark Runner ==="

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

# Verify required datasets
for dataset in "llama3-8b-sharegpt-test-t1-s0-8192.jsonl" "PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found!"
    fi
done

# Verify required model configs (authors')
for config in \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] || echo "  WARNING: $config not found (authors' model)!"
done

# Verify required model configs (ours)
for config in \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket100-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-b4/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket10-b4/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] || echo "  WARNING: $config not found (run train_all.sh first)!"
done

# ---- Run ----
mkdir -p "$PROJECT_ROOT/rep/final_results/benchmarks/results/sharegpt"

echo ""
echo "  8 schedulers x 7 request rates = 56 runs"
echo "  (fcfs, oracle-srtf, opt-xxx, class10, class82, pctl10, width10, mlfq)"
echo ""

bash "$BENCH_DIR/bench-final-sharegpt.sh" 2>&1 | tee "$PROJECT_ROOT/rep/final_results/benchmarks/results/sharegpt/bench_final_sharegpt_run.log"

echo ""
echo "=== ShareGPT Final Benchmarks complete ==="
echo "Results: $BENCH_DIR/RESULTS/"