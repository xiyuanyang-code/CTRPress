#!/bin/bash

# Usage: bash run_ensemble_comparison.sh <MODEL> <PRESS_METHOD>
# Example: bash run_ensemble_comparison.sh pythia risk_aware_ensemble
# Runs a single press method across datasets and compression ratios.

MODEL=$1
PRESS_METHOD=$2

if [ -z "$MODEL" ] || [ -z "$PRESS_METHOD" ]; then
    echo "Usage: bash run_ensemble_comparison.sh <MODEL> <PRESS_METHOD>"
    echo ""
    echo "  MODEL: pythia | qwen_3_1.7b"
    echo "  PRESS_METHOD: one of:"
    echo "    expected_attention | compactor | keydiff"
    echo "    adakv_expected_attention | criticalkv_expected_attention"
    echo "    risk_aware_ensemble | risk_aware_ensemble_light"
    echo "    adakv_risk_aware_ensemble | adakv_risk_aware_ensemble_light"
    exit 1
fi

COMPRESSION_RATIOS=(0.25 0.50 0.75 0.875)
DATASETS=("pg19" "wikitext" "nolima")
MAX_NEW_TOKENS=1000
PROMPT_LENGTH=1024
EVAL_LENGTH=128

for DATASET in "${DATASETS[@]}"; do
  for RATIO in "${COMPRESSION_RATIOS[@]}"; do

    echo "========================================"
    echo "Model: $MODEL | Dataset: $DATASET | Method: $PRESS_METHOD | Ratio: $RATIO"
    echo "========================================"

    python main.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --compress_ratio "$RATIO" \
        --press_method "$PRESS_METHOD" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --prompt_length "$PROMPT_LENGTH" \
        --eval_length "$EVAL_LENGTH" \
        --n_repeats 3 \
        --max_samples 1 \
        --output_dir results_main

  done
done

echo ""
echo "All experiments completed for $MODEL + $PRESS_METHOD!"
