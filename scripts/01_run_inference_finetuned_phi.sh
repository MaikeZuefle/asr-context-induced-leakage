#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_FOLDER="generated_output_finetuned/phi"

MODELS=(
    "saves/phi4-multimodal/lora/context_word"
    "saves/phi4-multimodal/lora/target_word"
    "saves/phi4-multimodal/lora/both"
    "saves/phi4-multimodal/lora/fleurs_context_1"
    "saves/phi4-multimodal/lora/fleurs_context_5"
    "saves/phi4-multimodal/lora/fleurs_context_10"
    "saves/phi4-multimodal/lora/fleurs_context_mixed"
    "saves/phi4-multimodal/lora/context_word_fleurs_mixed"
    "saves/phi4-multimodal/lora/target_word_fleurs_mixed"
    "saves/phi4-multimodal/lora/both_fleurs_mixed"
)

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Running inference with finetuned model: $MODEL_PATH"
    python -m src.test_privacy \
        --model phi_multimodal \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
