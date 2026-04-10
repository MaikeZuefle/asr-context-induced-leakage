#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_FOLDER="generated_output_finetuned"

MERGED_MODELS=(
    "saves/phi4-multimodal/lora/context_word"
    # "saves/phi4-multimodal/lora/target_word"
    # "saves/phi4-multimodal/lora/both"
)

for MODEL_PATH in "${MERGED_MODELS[@]}"; do
    echo "Running inference with finetuned model: $MODEL_PATH"
    python -m src.test_privacy \
        --model phi_multimodal \
        --prepared_path "$PREPARED_PATH" \
        --out_folder "$OUT_FOLDER" \
        --model_path "$MODEL_PATH"
done
