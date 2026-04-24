#fixlong3-tpt-pctl10-mse (Fix top 3 bins to oracle, pctl10-mse classifier, LMSYS)
DSRTF_BOUNDS_PATH=../../extension/training/configs/pctl10_mse_lmsys_boundaries.json CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type fixlong3-tpt-pctl10-mse-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixlong3-tpt-pctl10-mse-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixlong3-tpt-pctl10-mse-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343

kill $!
sleep 60


#fixlong5-tpt-pctl10-mse (Fix top 5 bins to oracle, pctl10-mse classifier, LMSYS)
DSRTF_BOUNDS_PATH=../../extension/training/configs/pctl10_mse_lmsys_boundaries.json CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type fixlong5-tpt-pctl10-mse-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixlong5-tpt-pctl10-mse-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixlong5-tpt-pctl10-mse-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343

kill $!
sleep 60


#fixshort5-tpt-pctl10-mse (Fix bottom 5 bins to oracle, pctl10-mse classifier, LMSYS)
DSRTF_BOUNDS_PATH=../../extension/training/configs/pctl10_mse_lmsys_boundaries.json CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type fixshort5-tpt-pctl10-mse-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixshort5-tpt-pctl10-mse-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixshort5-tpt-pctl10-mse-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343

kill $!
sleep 60


#fixshort3-tpt-pctl10-mse (Fix bottom 3 bins to oracle, pctl10-mse classifier, LMSYS)
DSRTF_BOUNDS_PATH=../../extension/training/configs/pctl10_mse_lmsys_boundaries.json CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --swap-space 100 --disable-log-requests --schedule-type fixshort3-tpt-pctl10-mse-xxx --enable-chunked-prefill --enforce-eager --prefill-predictor-model-config MODEL/results/opt-125m-llama3-8b-lmsys-pctl10-mse-b4-ext/usage_config.json --port 3343 &
sleep 120
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixshort3-tpt-pctl10-mse-xxx --output-len -1 --request-rate 32 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343
python benchmark_serving_real_with_metrics.py --backend vllm --model meta-llama/Meta-Llama-3-8B-Instruct  --tokenizer meta-llama/Meta-Llama-3-8B-Instruct --dataset lmsys-Meta-Llama-3-8B-Instruct-t1.0-s0-l8192-c10000-rFalse.jsonl --num-prompts -1 --request-time 60 --schedule-type fixshort3-tpt-pctl10-mse-xxx --output-len -1 --request-rate 64 --result-dir ../../extension/benchmarks/results/fixregion_pctl10mse_lmsys --port 3343

kill $!
sleep 60