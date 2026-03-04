#!/usr/bin/env bash

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== vllm-ltr 8B Oracle SJF Benchmark — ShareGPT ==="

# ---- Check prerequisites ----
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"

# ---- Setup data symlinks ----
source "$SCRIPT_DIR/setup_bench_data.sh"

# ---- Verify required datasets ----
for dataset in "llama3-8b-sharegpt-test-t1-s0-8192.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found in $BENCH_DIR"
    fi
done

# ---- Run ----
mkdir -p "$BENCH_DIR/RESULTS"

echo ""
echo "  Oracle SJF: 4 scheduler x 6 request rates = 24 runs"
echo ""

bash "$BENCH_DIR/bench_oracle_sjf.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_oracle_sjf_sharegpt_run.log"

echo ""
echo "=== Oracle SJF ShareGPT Benchmarks complete ==="
echo "Results: $BENCH_DIR/RESULTS/"
