#!/bin/bash

# Usage: bash run_qsm_main.sh <MODEL> <PRESS_METHOD>
# Example: bash run_qsm_main.sh qwen_3_1.7b qsm
# QSM-Press: query_aware | qa_semantic | qa_merge | qsm
# CTR-Press: ctr_refine | ctr_semantic | ctr
# 3 ratios × 3 datasets = 9 runs per invocation

MODEL=$1
PRESS_METHOD=$2

if [ -z "$MODEL" ] || [ -z "$PRESS_METHOD" ]; then
    echo "Usage: bash run_qsm_main.sh <MODEL> <PRESS_METHOD>"
    echo "  MODEL: pythia | qwen_3_1.7b"
    echo "  PRESS_METHOD: query_aware | qa_semantic | qa_merge | qsm | ctr_refine | ctr_semantic | ctr"
    exit 1
fi

COMPRESSION_RATIOS=(0.3 0.5 0.7)
DATASETS=("nolima" "pg19" "wikitext")
MAX_NEW_TOKENS=128
PROMPT_LENGTH=1024
EVAL_LENGTH=128

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
        --output_dir results_qsm_2

  done
done

echo "All experiments completed for $MODEL + $PRESS_METHOD!"
