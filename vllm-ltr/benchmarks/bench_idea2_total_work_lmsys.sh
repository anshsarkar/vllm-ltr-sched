#!/usr/bin/env bash
# Idea 2 bench sim — LMSYS
# Proves: ignoring prompt length in scheduling leaves performance on the table.
# Server uses SJF (sorts by est_tokens); client sets est_tokens = output_len + alpha * prompt_len.
#
# Four schedule types, one SJF server:
#   sjf                  — oracle baseline (sorts by output_len only)
#   sjf-totalwork-0.1    — alpha=0.1 (small prompt weight)
#   sjf-totalwork-0.5    — alpha=0.5 (moderate prompt weight)
#   sjf-totalwork-1.0    — alpha=1.0 (equal prompt weight)

MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
DATASET="lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl"
RATES="2 4 8 16 32 64"
PORT=3343

# Launch SJF server (no chunked prefill — uses _schedule_default with SJF sort)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model $MODEL --swap-space 16 --disable-log-requests \
    --schedule-type sjf --enforce-eager --port $PORT &
sleep 120

# --- Oracle SJF baseline ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

# --- Total-work alpha=0.1 ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf-totalwork-0.1 --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

# --- Total-work alpha=0.5 ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf-totalwork-0.5 --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

# --- Total-work alpha=1.0 ---
for r in $RATES; do
    python benchmark_serving_real_with_metrics.py \
        --backend vllm --model $MODEL --tokenizer $MODEL \
        --dataset $DATASET --num-prompts -1 --request-time 60 \
        --schedule-type sjf-totalwork-1.0 --output-len -1 --request-rate $r \
        --result-dir RESULTS --port $PORT
done

kill $!
sleep 60
