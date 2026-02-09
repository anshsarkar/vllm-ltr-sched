#!/usr/bin/env bash
# Symlink datasets and predictor models into the benchmarks directory.
# Shared by run_single_bench.sh and run_bench_8b.sh.
#
# Usage: source scripts/setup_bench_data.sh  (expects BENCH_DIR, DATA_DIR set)
set -euo pipefail

: "${BENCH_DIR:?BENCH_DIR must be set}"
: "${DATA_DIR:?DATA_DIR must be set}"

# Symlink dataset JSONL files
TRACE_DIR="$DATA_DIR/datasets/Llama3-Trace"
if [ -d "$TRACE_DIR" ]; then
    for f in "$TRACE_DIR"/*.jsonl; do
        base=$(basename "$f")
        [ -e "$BENCH_DIR/$base" ] || ln -sf "$f" "$BENCH_DIR/$base"
    done
fi

# Symlink predictor model directories
PRED_DIR="$DATA_DIR/models/predictors"
if [ -d "$PRED_DIR" ]; then
    mkdir -p "$BENCH_DIR/MODEL/results"
    for d in "$PRED_DIR"/*/; do
        [ -d "$d" ] || continue
        base=$(basename "$d")
        [ -e "$BENCH_DIR/MODEL/results/$base" ] || ln -sf "$d" "$BENCH_DIR/MODEL/results/$base"
    done
fi

# Ensure results directory exists
mkdir -p "$BENCH_DIR/SERVE"
