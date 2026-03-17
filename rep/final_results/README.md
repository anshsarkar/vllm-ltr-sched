# Final Results

## Commands

```bash
# 1. Train all models (class82, pctl10, width10)
bash rep/final_results/training/scripts/train_all.sh

# 2. Parse training metrics to CSV
python rep/final_results/analysis/parse_training_logs.py

# 3. Run benchmarks
bash rep/final_results/benchmarks/scripts/run_bench_sharegpt.sh
bash rep/final_results/benchmarks/scripts/run_bench_lmsys.sh

# 4. Extract benchmark metrics and generate plots
python rep/final_results/analysis/plot_benchmark_results.py rep/final_results/benchmarks/results/sharegpt
python rep/final_results/analysis/plot_benchmark_results.py rep/final_results/benchmarks/results/lmsys
```