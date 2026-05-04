#!/bin/bash

PREPARED_PATH="data/prepared/voxpopuli.jsonl"
OUT_FOLDER="generated_output_finetuned/voxpopuli/qwen"

MERGED_MODELS=(
    "saves/qwen2.5-omni-7b/merged/fleurs_context_mixed"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_context_word"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_target_word"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_both"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_context_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_target_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/merged/voxpopuli_both_fleurs_mixed"
)

for MODEL_PATH in "${MERGED_MODELS[@]}"; do
    echo "Running inference with merged model: $MODEL_PATH"
    python -m src.test_privacy \
        --model qwen_omni \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
