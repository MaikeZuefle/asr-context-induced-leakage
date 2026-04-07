#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_FOLDER="generated_output"
MODELS=("qwen_omni")

for MODEL in "${MODELS[@]}"; do
    echo "Running $MODEL..."
    python -m src.test_privacy \
        --model "$MODEL" \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER"
done
