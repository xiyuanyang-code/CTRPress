#!/bin/bash

# Ablation: compression rate sweep
# Fixed: pythia model, pg19 dataset
# Variable: compression ratio (100 points from 0.01 to 0.99)

# Usage: bash ablation_compression_rate_full.sh <PRESS_METHOD>
# Example: bash ablation_compression_rate_full.sh snapkv

# ("ctr" "random" "qsm" "snapkv" "pyramidkv")
PRESS_METHOD=$1

if [ -z "$PRESS_METHOD" ]; then
    echo "Usage: bash ablation_compression_rate_full.sh <PRESS_METHOD>"
    echo "  Available methods: ctr, random, qsm, snapkv, pyramidkv"
    exit 1
fi

MODEL="pythia"
DATASET="pg19"
MAX_NEW_TOKENS=128
PROMPT_LENGTH=1024
EVAL_LENGTH=128

# Generate 100 evenly spaced compression ratios from 0.01 to 0.99
COMPRESSION_RATIOS=$(python3 -c "print(' '.join([f'{x:.2f}' for x in __import__('numpy').linspace(0.01, 0.99, 100)]))")

echo "========================================"
echo "Ablation: Compression Rate Sweep"
echo "Model: $MODEL | Dataset: $DATASET | Method: $PRESS_METHOD"
echo "Ratios: 100 points from 0.01 to 0.99"
echo "========================================"

for COMPRESSION_RATIO in $COMPRESSION_RATIOS; do

  echo "----------------------------------------"
  echo "Method: $PRESS_METHOD | Ratio: $COMPRESSION_RATIO"
  echo "----------------------------------------"

  python main.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --compress_ratio "$COMPRESSION_RATIO" \
      --press_method "$PRESS_METHOD" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --prompt_length "$PROMPT_LENGTH" \
      --eval_length "$EVAL_LENGTH" \
      --n_repeats 1 \
      --max_samples 1 \
      --output_dir results_ablation_compression_rate

done

echo "All compression rate ablations completed for $PRESS_METHOD!"
