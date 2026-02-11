#!/usr/bin/env bash
# bash run_single_bench.sh -s opt-xxx -r 64
set -euo pipefail

# Important fix for "OSError: [Errno 24] Too many open files" when running many requests in parallel
ulimit -n 65536 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BENCH_DIR="$PROJECT_ROOT/vllm-ltr/benchmarks"
DATA_DIR="$PROJECT_ROOT/data"
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"

SCHED="fcfs" RATE=2 TIME=60
while getopts "s:r:t:" opt; do
    case $opt in
        s) SCHED="$OPTARG" ;; r) RATE="$OPTARG" ;; t) TIME="$OPTARG" ;;
        *) echo "Usage: $0 [-s schedule_type] [-r request_rate] [-t request_time]"; exit 1 ;;
    esac
done

# ---- Scheduler config ----
SERVER_ARGS="--model $MODEL --disable-log-requests --enable-chunked-prefill --enforce-eager"
SWAP=16
BENCH_SCHED="$SCHED"
DATASET="llama3-8b-sharegpt-test-t1-s0-8192.jsonl"

case "$SCHED" in
    fcfs)           SERVER_ARGS+=" --schedule-type fcfs" ;;
    opt-xxx)        SERVER_ARGS+=" --schedule-type opt-xxx --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json" ;;
    tpt-class10-xxx) SERVER_ARGS+=" --schedule-type tpt-class10-xxx --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json"; SWAP=100 ;;
    mlfq*)          SERVER_ARGS+=" --schedule-type mlfq-base0.03-thres10"; SWAP=100; BENCH_SCHED="mlfq" ;;
    PO|po)          SERVER_ARGS+=" --schedule-type PO"; SWAP=100; BENCH_SCHED="srtf-PO-X"; DATASET="PO-gen-llama3-8b-sharegpt-test-t1-s0-8192.jsonl" ;;
    *)              echo "Unknown scheduler: $SCHED (options: fcfs, opt-xxx, tpt-class10-xxx, mlfq, PO)"; exit 1 ;;
esac

SERVER_ARGS+=" --swap-space $SWAP"

# ---- Setup data symlinks ----
cd "$BENCH_DIR"
source "$SCRIPT_DIR/setup_bench_data.sh"

echo "=== Benchmark: $SCHED @ ${RATE} req/s for ${TIME}s ==="

# ---- Start server & wait for health ----
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server $SERVER_ARGS &
SERVER_PID=$!
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM EXIT

echo "Waiting for server..."
for ((i=0; i<120; i++)); do
    curl -s --fail http://localhost:8000/health >/dev/null 2>&1 && break
    sleep 5
done
curl -s --fail http://localhost:8000/health >/dev/null 2>&1 || { echo "ERROR: Server not ready after 600s"; exit 1; }
echo "Server ready."

# ---- Run benchmark ----
python benchmark_serving_real.py \
    --backend vllm --model "$MODEL" --tokenizer "$MODEL" \
    --dataset "$DATASET" --num-prompts -1 \
    --request-time "$TIME" --schedule-type "$BENCH_SCHED" \
    --output-len -1 --request-rate "$RATE"

# ---- Cleanup ----
echo "Stopping server..."
kill "$SERVER_PID" 2>/dev/null
wait "$SERVER_PID" 2>/dev/null || true
trap - INT TERM EXIT
echo "Done."
