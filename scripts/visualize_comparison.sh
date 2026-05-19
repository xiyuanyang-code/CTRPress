#!/bin/bash

# KV Cache Compression Visualization - Comparison Script
# 生成多种压缩方法的对比热力图

# Usage: bash scripts/visualize_comparison.sh <MODEL>
# Example: bash scripts/visualize_comparison.sh qwen_3_1.7b

MODEL=${1:-"qwen_3_1.7b"}
DATASET="pg19"
COMPRESSION_RATIO=0.5
PROMPT_LENGTH=1024
MAX_SAMPLES=1

OUTPUT_DIR="results_visualizations_comparison"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "Generating Compression Heatmap Comparison"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Compression Ratio: $COMPRESSION_RATIO"
echo "========================================"

# 测试不同的压缩方法
PRESS_METHODS=("no_compress" "random" "snapkv" "pyramidkv")

for PRESS_METHOD in "${PRESS_METHODS[@]}"; do
  echo ""
  echo "Running: $PRESS_METHOD..."

  python main.py \
      --dataset "$DATASET" \
      --model "$MODEL" \
      --compress_ratio "$COMPRESSION_RATIO" \
      --press_method "$PRESS_METHOD" \
      --max_new_tokens 128 \
      --prompt_length "$PROMPT_LENGTH" \
      --eval_length 128 \
      --n_repeats 1 \
      --max_samples "$MAX_SAMPLES" \
      --visualize \
      --output_dir "$OUTPUT_DIR"

  echo "Completed: $PRESS_METHOD"
done

echo ""
echo "========================================"
echo "All heatmaps saved to: $OUTPUT_DIR/visualizations/"
echo "========================================"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR/visualizations/" | grep ".png"
