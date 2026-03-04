#oracle-sjf lmsys (perfect SJF using actual output lengths — no aux model needed)
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type sjf --enable-chunked-prefill --enforce-eager --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 2 --result-dir RESULTS --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 4 --result-dir RESULTS --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 8 --result-dir RESULTS --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 16 --result-dir RESULTS --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 32 --result-dir RESULTS --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type sjf --output-len -1 --request-rate 64 --result-dir RESULTS --port 3343

kill $!
sleep 60

#fcfs
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type fcfs --enable-chunked-prefill --enforce-eager --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 2 --result-dir RESULTS --port 3343 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 4 --result-dir RESULTS --port 3343 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 8 --result-dir RESULTS --port 3343 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 16 --result-dir RESULTS --port 3343 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 32 --result-dir RESULTS --port 3343 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fcfs --output-len -1 --request-rate 64 --result-dir RESULTS --port 3343 

kill $!
sleep 60


#opt-ours
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 32 --disable-log-requests --schedule-type opt-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-score-trainbucket10-b32/usage_config.json &
sleep 120

python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type opt-xxx --output-len -1 --request-rate 64 --result-dir RESULTS 


kill $!
sleep 60

#opt-class10
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 16 --disable-log-requests --schedule-type tpt-class10-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-class-trainbucket820-b32/usage_config.json --swap-space 100 &
sleep 120

python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 2 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 4 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 8 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 16 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 32 --result-dir RESULTS 
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type tpt-class10-xxx --output-len -1 --request-rate 64 --result-dir RESULTS 

kill $!
sleep 60
