#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_FOLDER="generated_output_finetuned/qwen"

MERGED_MODELS=(
    "saves/qwen2.5-omni-7b/merged/context_word"
    "saves/qwen2.5-omni-7b/merged/target_word"
    "saves/qwen2.5-omni-7b/merged/both"
)

for MODEL_PATH in "${MERGED_MODELS[@]}"; do
    echo "Running inference with merged model: $MODEL_PATH"
    python -m src.test_privacy \
        --model qwen_omni \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
