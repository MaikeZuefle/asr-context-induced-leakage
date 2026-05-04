#!/bin/bash

BASE_MODEL="Qwen/Qwen2.5-Omni-7B"

MERGES=(
    "saves/qwen2.5-omni-7b/lora/voxpopuli_context_word:saves/qwen2.5-omni-7b/merged/voxpopuli_context_word"
    "saves/qwen2.5-omni-7b/lora/voxpopuli_target_word:saves/qwen2.5-omni-7b/merged/voxpopuli_target_word"
    "saves/qwen2.5-omni-7b/lora/voxpopuli_both:saves/qwen2.5-omni-7b/merged/voxpopuli_both"
    "saves/qwen2.5-omni-7b/lora/voxpopuli_context_word_fleurs_mixed:saves/qwen2.5-omni-7b/merged/voxpopuli_context_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/lora/voxpopuli_target_word_fleurs_mixed:saves/qwen2.5-omni-7b/merged/voxpopuli_target_word_fleurs_mixed"
    "saves/qwen2.5-omni-7b/lora/voxpopuli_both_fleurs_mixed:saves/qwen2.5-omni-7b/merged/voxpopuli_both_fleurs_mixed"
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
