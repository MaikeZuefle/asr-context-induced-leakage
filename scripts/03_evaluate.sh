#!/bin/bash

OUT_FOLDER="generated_output"
MODELS=("phi_multimodal" "qwen_omni")

for MODEL in "${MODELS[@]}"; do
    RESULTS_PATH="${OUT_FOLDER}/${MODEL}/privacy/en.jsonl"
    echo "=== $MODEL ==="
    python evaluation/evaluate.py --results_path "$RESULTS_PATH"
    echo ""
done
