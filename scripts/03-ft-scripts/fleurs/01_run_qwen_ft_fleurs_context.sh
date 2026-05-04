#!/bin/bash
export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

CONFIGS=(
    "configs/qwen_omni_lora_fleurs_context_1.yaml"
    "configs/qwen_omni_lora_fleurs_context_5.yaml"
    "configs/qwen_omni_lora_fleurs_context_10.yaml"
    "configs/qwen_omni_lora_fleurs_context_mixed.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "Starting training with config: $CONFIG"
    llamafactory-cli train "$CONFIG"
    echo "Done: $CONFIG"
done
