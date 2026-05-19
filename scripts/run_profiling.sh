#!/bin/bash

# Usage: bash scripts/run_profiling.sh
# Run a single evaluation with cProfile enabled
# Results saved to profile.stats for analysis with snakeviz or pstats

# 配置参数
DATASET="pg19"
MODEL="qwen_3_1.7b"
COMPRESS_RATIO=0.5
PRESS_METHOD="ctr"
MAX_NEW_TOKENS=128
PROMPT_LENGTH=1024
EVAL_LENGTH=128
MAX_SAMPLES=1
N_REPEATS=1

echo "========================================"
echo "Profiling Run Configuration"
echo "========================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "Compression Ratio: $COMPRESS_RATIO"
echo "Press Method: $PRESS_METHOD"
echo "Max Samples: $MAX_SAMPLES"
echo "========================================"
echo ""

python main.py \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --compress_ratio "$COMPRESS_RATIO" \
    --press_method "$PRESS_METHOD" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --prompt_length "$PROMPT_LENGTH" \
    --eval_length "$EVAL_LENGTH" \
    --n_repeats "$N_REPEATS" \
    --max_samples "$MAX_SAMPLES" \
    --profile

echo ""
echo "========================================"
echo "Profiling Complete!"
echo "========================================"
echo "View results with:"
echo "  python -m pstats profile.stats"
echo "  snakeviz profile.stats"
echo "========================================"
