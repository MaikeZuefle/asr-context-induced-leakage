#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-Omni-7B"

MERGES=(
    # "saves/qwen2.5-omni-7b/lora/context_word:saves/qwen2.5-omni-7b/merged/context_word"
    "saves/qwen2.5-omni-7b/lora/target_word:saves/qwen2.5-omni-7b/merged/target_word"
    "saves/qwen2.5-omni-7b/lora/both:saves/qwen2.5-omni-7b/merged/both"
)

for ENTRY in "${MERGES[@]}"; do
    LORA_PATH="${ENTRY%%:*}"
    SAVE_PATH="${ENTRY##*:}"
    echo "Merging LoRA: $LORA_PATH -> $SAVE_PATH"
    python LlamaFactory/scripts/qwen_omni_merge.py merge_lora \
        --model_path="$BASE_MODEL" \
        --lora_path="$LORA_PATH" \
        --save_path="$SAVE_PATH"
    echo "Done: $SAVE_PATH"
done
