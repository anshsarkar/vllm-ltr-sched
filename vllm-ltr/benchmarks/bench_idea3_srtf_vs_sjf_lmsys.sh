#!/usr/bin/env bash
# Idea 3 bench sim — LMSYS
# Proves: static ordering (SJF) is fundamentally worse than dynamic ordering (SRTF).
# Both use oracle est_tokens = actual output length.
# SJF sorts once at admission; SRTF re-sorts by remaining time every step.
#
# Two phases, two servers:
#   Phase 1: SJF server + sjf client (oracle, static)
#   Phase 2: SRTF server + srtf-oracle client (oracle, dynamic)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET="lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl"
RATES="2 4 8 16 32 64"
PORT=3343

# === Phase 1: Oracle SJF (static ordering) ===
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --swap-space 16 --disable-log-requests \
    --schedule-type sjf --enforce-eager --port $PORT &
sleep 120

for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

kill $!
sleep 60

# === Phase 2: Oracle SRTF (dynamic re-ordering) ===
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --swap-space 16 --disable-log-requests \
    --schedule-type srtf-oracle --enable-chunked-prefill --enforce-eager --port $PORT &
sleep 120

for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type srtf-oracle --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

kill $!
sleep 60
