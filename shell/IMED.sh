#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="pipeline.py"
DATASET_NAME="medqa"
DATASET_DIR="MIRAGE"
OUTPUT_PATH="results/Llama-3.2-3B-MedCPT-Textbooks-MedQA-IMED.json"
DEVICE_NAME="cuda:0"
LOG_FILE="logs/try-MedQA-IMED.log"

mkdir -p logs

nohup python "$PYTHON_SCRIPT" \
    --dataset_name "$DATASET_NAME" \
    --dataset_dir "$DATASET_DIR" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE_NAME" \
    --rag \
    --retrieval_mode direct \
    --follow_up \
    --follow_up_rounds 2 \
    --follow_up_queries 3 \
    > "$LOG_FILE" 2>&1 &
