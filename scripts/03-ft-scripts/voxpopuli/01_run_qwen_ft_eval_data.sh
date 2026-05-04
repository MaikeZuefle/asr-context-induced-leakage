#!/bin/bash
export TORCHAUDIO_USE_BACKEND_DISPATCHER=0

CONFIGS=(
    "configs/qwen_omni_lora_voxpopuli_context_word.yaml"
    "configs/qwen_omni_lora_voxpopuli_target_word.yaml"
    "configs/qwen_omni_lora_voxpopuli_both.yaml"
)

for CONFIG in "${CONFIGS[@]}"; do
    echo "Starting training with config: $CONFIG"
    llamafactory-cli train "$CONFIG"
    echo "Done: $CONFIG"
done
