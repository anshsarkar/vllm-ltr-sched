#!/usr/bin/env bash

RESULT_DIR="../../extension/loss_experiments/benchmarks/results/sharegpt"


# CORAL Ordinal (tpt-pctl10-coral-xxx)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-pctl10-coral-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-coral-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-coral-xxx --output-len -1 --request-rate 32 --result-dir $RESULT_DIR --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-coral-xxx --output-len -1 --request-rate 64 --result-dir $RESULT_DIR --port 3343

kill $!
sleep 60

# Cost-Sensitive CE (tpt-pctl10-costsensitive-xxx)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-pctl10-costsensitive-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-costsensitive-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-costsensitive-xxx --output-len -1 --request-rate 32 --result-dir $RESULT_DIR --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-costsensitive-xxx --output-len -1 --request-rate 64 --result-dir $RESULT_DIR --port 3343

kill $!
sleep 60

# Ranking listMLE — locally retrained (opt-xxx)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 32 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-ranking-listmle-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir $RESULT_DIR --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 --result-dir $RESULT_DIR --port 3343

kill $!
sleep 60
