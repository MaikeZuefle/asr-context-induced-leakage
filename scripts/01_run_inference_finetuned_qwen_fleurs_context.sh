#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_FOLDER="generated_output_finetuned/qwen"

MERGED_MODELS=(
    "saves/qwen2.5-omni-7b/merged/fleurs_context_1"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_5"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_10"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_mixed"
)

for MODEL_PATH in "${MERGED_MODELS[@]}"; do
    echo "Running inference with merged model: $MODEL_PATH"
    python -m src.test_privacy \
        --model qwen_omni \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
