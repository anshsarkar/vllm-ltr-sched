#fcfs
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


# #PO new (Oracle SRTF)
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 120 --disable-log-requests --schedule-type PO --enable-chunked-prefill --enforce-eager --port 3343 &
# sleep 120
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset PO-gen-lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type srtf-PO-X --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

# kill $!
# sleep 60


#opt-xxx (LTR score/ranking predictor, OPT-125m, authors')
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 32 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


#tpt-class10 (classification, bucket=820, 10 classes, OPT-125m, authors')
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-class10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


#tpt-class82 (classification, bucket=100, ~82 classes, OPT-125m, ours)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-class82-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket100-b4/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class82-xxx --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


#tpt-pctl10 (percentile, 10 classes, OPT-125m, ours)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-pctl10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-b4/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-pctl10-xxx --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


#tpt-width10 (classification, bucket=10, ~820 classes, OPT-125m, ours)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type tpt-width10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket10-b4/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-width10-xxx --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

kill $!
sleep 60


#mlfq
# CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 125 --disable-log-requests --schedule-type mlfq-base0.03-thres10 --enable-chunked-prefill --enforce-eager --port 3343 &
# sleep 120
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 2 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 4 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 8 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 16 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 32 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343
# python benchmark_serving_real.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type mlfq --output-len -1 --request-rate 64 --result-dir ../../rep/final_results/benchmarks/results/lmsys --port 3343

# kill $!
# sleep 60