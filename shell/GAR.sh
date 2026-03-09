#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="pipeline.py"
DATASET_NAME="medqa"
DATASET_DIR="MIRAGE"
OUTPUT_PATH="results/Llama-3.2-3B-MedCPT-Textbooks-MedQA-GAR.json"
DEVICE_NAME="cuda:0"
LOG_FILE="logs/try-MedQA-GAR.log"

mkdir -p logs

nohup python "$PYTHON_SCRIPT" \
    --dataset_name "$DATASET_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE_NAME" \
    --rag \
    --retrieval_mode gar \
    > "$LOG_FILE" 2>&1 &
