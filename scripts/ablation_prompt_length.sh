#!/bin/bash

# Ablation: prompt length scaling analysis
# Fixed: qwen-1.7b model, pg19 dataset, compression ratio 0.5
# Variable: PROMPT_LENGTH (256, 512, 1024, 2048, 4096)
# Methods: ctr, random, qsm, snapkv, pyramidkv

# Usage: bash ablation_prompt_length.sh

PRESS_METHODS=("ctr" "random" "qsm" "snapkv" "pyramidkv")
MODEL="qwen_3_1.7b"
DATASET="pg19"
COMPRESSION_RATIO=0.5
MAX_NEW_TOKENS=128
PROMPT_LENGTHS=(256 512 1024 2048 4096)
EVAL_LENGTH=128

for PRESS_METHOD in "${PRESS_METHODS[@]}"; do
  for PROMPT_LENGTH in "${PROMPT_LENGTHS[@]}"; do

    echo "========================================"
    echo "Ablation: Prompt Length Scaling"
    echo "Model: $MODEL | Dataset: $DATASET | Method: $PRESS_METHOD | Ratio: $COMPRESSION_RATIO | Prompt: $PROMPT_LENGTH"
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
        --output_dir results_ablation_prompt_length

    echo "Completed: $PRESS_METHOD + prompt_length=$PROMPT_LENGTH"
  done
done

echo "All prompt length ablations completed!"
