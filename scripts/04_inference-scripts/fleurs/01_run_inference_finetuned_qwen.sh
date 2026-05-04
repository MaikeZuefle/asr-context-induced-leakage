#!/bin/bash

PREPARED_PATH="data/prepared/fleurs.jsonl"
OUT_FOLDER="generated_output_finetuned/fleurs/qwen"

MERGED_MODELS=(
    "saves/qwen2.5-omni-7b/merged/context_word"
    "saves/qwen2.5-omni-7b/merged/target_word"
    "saves/qwen2.5-omni-7b/merged/both"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_1"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_5"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_10"
    "saves/qwen2.5-omni-7b/merged/fleurs_context_mixed"
    "saves/qwen2.5-omni-7b/merged/context_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/merged/target_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/merged/both_fleurs_mixed"
)

for MODEL_PATH in "${MERGED_MODELS[@]}"; do
    echo "Running inference with merged model: $MODEL_PATH"
    python -m src.test_privacy \
        --model qwen_omni \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
