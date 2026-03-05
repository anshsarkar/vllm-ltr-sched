#!/usr/bin/env bash
# Runner for Idea 3 bench sim: SRTF vs SJF
# Proves: static ordering is fundamentally worse than dynamic ordering
# Runs both ShareGPT and LMSYS benchmarks

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"

cd "$BENCH_DIR"

echo "=== Idea 3: SRTF vs SJF Benchmark ==="
echo "  Proves: one-shot static ordering leaves performance on the table"
echo ""

# ---- Check prerequisites ----
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
python -c "import vllm; print(f'  vLLM version: {vllm.__version__}')"

# ---- Setup data symlinks ----
source "$SCRIPT_DIR/setup_bench_data.sh"

# ---- Verify required datasets ----
for dataset in "llama3-8b-sharegpt-test-t1-s0-8192.jsonl" "lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl"; do
    if [ ! -f "$BENCH_DIR/$dataset" ]; then
        echo "  WARNING: $dataset not found in $BENCH_DIR"
    fi
done

# ---- Run ----
mkdir -p "$BENCH_DIR/RESULTS"

echo ""
echo "  ShareGPT: 2 schedule types x 6 rates = 12 runs (2 server launches)"
echo ""
bash "$BENCH_DIR/bench_idea3_srtf_vs_sjf.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_idea3_srtf_vs_sjf_sharegpt_run.log"

# echo ""
# echo "  LMSYS: 2 schedule types x 6 rates = 12 runs (2 server launches)"
# echo ""
# bash "$BENCH_DIR/bench_idea3_srtf_vs_sjf_lmsys.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_idea3_srtf_vs_sjf_lmsys_run.log"

echo ""
echo "=== Idea 3 SRTF vs SJF Benchmarks complete ==="
echo "Results: $BENCH_DIR/RESULTS/"
