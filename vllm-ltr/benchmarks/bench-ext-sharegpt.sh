#fcfs
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


#sjf (Oracle shortest-job-first, uses true output lengths)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type sjf --enable-chunked-prefill --enforce-eager --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

kill $!
sleep 60


#opt-xxx (LTR score/ranking predictor, OPT-125m, authors')
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-score-trainbucket10-b32/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


# #tpt-class10 (classification, bucket=820, 10 classes, OPT-125m, authors')
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-class10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket820-b32/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


# #tpt-class82 (classification, bucket=100, ~82 classes, OPT-125m, ours, fresh -ext)
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-class82-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket100-b4-ext/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


# #tpt-pctl10 (percentile, 10 classes, OPT-125m, ours, fresh -ext)
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-pctl10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-b4-ext/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


# #tpt-width10 (classification, bucket=10, ~820 classes, OPT-125m, ours, fresh -ext)
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-width10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-class-trainbucket10-b4-ext/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60


# #tpt-pctl10-mse (percentile, 10 classes, MSE loss, OPT-125m, ours, fresh -ext)
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-pctl10-mse-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-sharegpt-pctl10-mse-b4-ext/usage_config.json --port 3343 &
# sleep 120
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 2 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 4 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 8 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 16 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343
# python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset llama3-8b-sharegpt-test-t1-s0-8192.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-mse-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/sharegpt --port 3343

# kill $!
# sleep 60
