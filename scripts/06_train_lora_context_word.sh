#!/bin/bash

export TORCHAUDIO_USE_BACKEND_DISPATCHER=0
llamafactory-cli train configs/qwen_omni_lora_context_word.yaml
