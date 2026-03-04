#!/usr/bin/env bash
# Idea 1 bench sim — ShareGPT
# Proves: same 20% error rate in opposite directions causes vastly different latency.
# Server uses SJF (sorts by est_tokens); corruption is client-side.
#
# Three schedule types, one SJF server:
#   sjf             — perfect oracle baseline
#   sjf-noisy-las   — 20% longest requests mislabeled as short (long-as-short)
#   sjf-noisy-sal   — 20% shortest requests mislabeled as long (short-as-long)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET="llama3-8b-sharegpt-test-t1-s0-8192.jsonl"
RATES="2 4 8 16 32 64"

# Launch SJF server (handles all three client-side schedule types)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --swap-space 16 --disable-log-requests \
    --schedule-type sjf --enforce-eager &
sleep 120

# --- Oracle SJF baseline ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf --output-len -1 --request-rate $r
done

# --- Noisy: long-as-short (20% longest → small values) ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf-noisy-las --output-len -1 --request-rate $r
done

# --- Noisy: short-as-long (20% shortest → large values) ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf-noisy-sal --output-len -1 --request-rate $r
done

kill $!
sleep 60
