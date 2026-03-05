#!/usr/bin/env bash
# Runner for Idea 2 bench sim: total-work-aware oracle
# Runs both ShareGPT and LMSYS benchmarks

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"

echo "=========================================="
echo "  Idea 2: Total-Work-Aware Oracle"
echo "=========================================="

cd "$BENCH_DIR"

echo ""
echo "--- ShareGPT ---"
bash bench_idea2_total_work.sh 2>&1 | tee "$PROJECT_ROOT/logs/idea2_sharegpt_$(date +%Y%m%d_%H%M%S).log"

# echo ""
# echo "--- LMSYS ---"
# bash bench_idea2_total_work_lmsys.sh 2>&1 | tee "$PROJECT_ROOT/logs/idea2_lmsys_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "Done. Check results in benchmarks/ directory."
