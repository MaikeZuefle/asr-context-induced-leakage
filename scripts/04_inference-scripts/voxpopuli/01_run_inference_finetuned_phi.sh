#!/bin/bash

PREPARED_PATH="data/prepared/voxpopuli.jsonl"
OUT_FOLDER="generated_output_finetuned/voxpopuli/phi"

MODELS=(
    "saves/phi4-multimodal/lora/fleurs_context_mixed"
    "saves/phi4-multimodal/lora/voxpopuli_context_word"
    "saves/phi4-multimodal/lora/voxpopuli_target_word"
    "saves/phi4-multimodal/lora/voxpopuli_both"
    "saves/phi4-multimodal/lora/voxpopuli_context_word_fleurs_mixed"
    "saves/phi4-multimodal/lora/voxpopuli_target_word_fleurs_mixed"
    "saves/phi4-multimodal/lora/voxpopuli_both_fleurs_mixed"
)

for MODEL_PATH in "${MODELS[@]}"; do
    echo "Running inference with finetuned model: $MODEL_PATH"
    python -m src.test_privacy \
        --model phi_multimodal \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
