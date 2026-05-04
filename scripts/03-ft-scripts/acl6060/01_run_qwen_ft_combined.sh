#!/bin/bash
export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

CONFIGS=(
    # "configs/qwen_omni_lora_acl6060_context_word_fleurs_mixed.yaml"
    # "configs/qwen_omni_lora_acl6060_target_word_fleurs_mixed.yaml"
    "configs/qwen_omni_lora_acl6060_both_fleurs_mixed.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "Starting training with config: $CONFIG"
    llamafactory-cli train "$CONFIG"
    echo "Done: $CONFIG"
done
