#!/bin/bash

# Usage: bash run_full.sh <PRESS_METHOD>
# Example: bash run_full.sh snapkv
# Tests both models (pythia first, then qwen) with all ratios and datasets
# 2 models × 3 ratios × 3 datasets = 18 runs per invocation

PRESS_METHOD=$1

if [ -z "$PRESS_METHOD" ]; then
    echo "Usage: bash run_full.sh <PRESS_METHOD>"
    echo "  PRESS_METHOD: no_compress | random | streaming_llm | snapkv | lagkv | keydiff | pyramidkv"
    echo "                query_aware | qa_semantic | qa_merge | qsm | ctr | ctr_refine | ctr_semantic"
    exit 1
fi

COMPRESSION_RATIOS=(0.3 0.5 0.7)
DATASETS=("nolima" "pg19" "wikitext")
MODELS=("pythia" "qwen_3_1.7b")
MAX_NEW_TOKENS=128
PROMPT_LENGTH=1024
EVAL_LENGTH=128

for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for COMPRESSION_RATIO in "${COMPRESSION_RATIOS[@]}"; do

      echo "========================================"
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
          --output_dir full_2

    done
  done
done

echo "All experiments completed for $PRESS_METHOD!"
