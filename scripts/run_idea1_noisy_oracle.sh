#!/usr/bin/env bash
# Runner for Idea 1 bench sim: noisy oracle (both datasets)
# Proves: long-as-short errors cause disproportionate latency damage vs short-as-long

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"

cd "$BENCH_DIR"

echo "=== Idea 1: Noisy Oracle Benchmark ==="
echo "  Proves: same error rate, opposite directions → vastly different latency"
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
echo "  ShareGPT: 3 schedule types x 6 rates = 18 runs"
echo ""
bash "$BENCH_DIR/bench_idea1_noisy_oracle.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_idea1_noisy_oracle_sharegpt_run.log"

# echo ""
# echo "  LMSYS: 3 schedule types x 6 rates = 18 runs"
# echo ""
# bash "$BENCH_DIR/bench_idea1_noisy_oracle_lmsys.sh" 2>&1 | tee "$BENCH_DIR/RESULTS/bench_idea1_noisy_oracle_lmsys_run.log"

# echo ""
# echo "=== Idea 1 Noisy Oracle Benchmarks complete ==="
# echo "Results: $BENCH_DIR/RESULTS/"
