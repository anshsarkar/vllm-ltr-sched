#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== vllm-ltr 8B Benchmark Runner ==="

# ---- Check prerequisites ----
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"

# ---- Setup data symlinks ----
source "$SCRIPT_DIR/setup_bench_data.sh"

# Verify required datasets
for dataset in "llama3-8b-sharegpt-test-t1-s0-8192.jsonl" "PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found!"
    fi
done

# Verify required model configs
for config in \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" \
    "MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json"; do
    [ -f "$BENCH_DIR/$config" ] || echo "  WARNING: $config not found!"
done

# ---- Run ----
mkdir -p "$BENCH_DIR/RESULTS"

echo ""
echo "  5 schedulers x 6 request rates = 30 runs (~3-4 hours)"
echo ""

bash "$BENCH_DIR/bench.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_8b_run_mlfq_64.log"

echo ""
echo "=== Benchmarks complete ==="
echo "Results: $BENCH_DIR/RESULTS/"
