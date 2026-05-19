#!/bin/bash

# Ablation: model size scaling analysis
# Fixed: pg19 dataset, compression ratio 0.5
# Variable: qwen family models (5 sizes: 0.6b, 1.7b, 4b, 8b, 14b)
# Methods: ctr, random, qsm, snapkv, pyramidkv

# Usage: bash ablation_model_size.sh

PRESS_METHODS=("ctr" "random" "qsm" "snapkv" "pyramidkv")
QWEN_MODELS=("qwen_3_0.6b" "qwen_3_1.7b" "qwen_3_4b" "qwen_3_8b" "qwen_3_14b")
DATASET="pg19"
COMPRESSION_RATIO=0.5
MAX_NEW_TOKENS=128
PROMPT_LENGTH=1024
EVAL_LENGTH=128

for PRESS_METHOD in "${PRESS_METHODS[@]}"; do
  for MODEL in "${QWEN_MODELS[@]}"; do

    echo "========================================"
    echo "Ablation: Model Size Scaling"
    echo "Model: $MODEL | Dataset: $DATASET | Method: $PRESS_METHOD | Ratio: $COMPRESSION_RATIO"
    echo "========================================"

    python main.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --compress_ratio "$COMPRESSION_RATIO" \
        --press_method "$PRESS_METHOD" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --prompt_length "$PROMPT_LENGTH" \
        --eval_length "$EVAL_LENGTH" \
        --n_repeats 3 \
        --max_samples 1 \
        --output_dir results_ablation_model_size

    echo "Completed: $MODEL + $PRESS_METHOD"
  done
done

echo "All model size ablations completed!"
