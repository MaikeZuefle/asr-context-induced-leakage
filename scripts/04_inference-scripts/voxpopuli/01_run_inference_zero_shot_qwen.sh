#!/bin/bash

PREPARED_PATH="data/prepared/voxpopuli.jsonl"
OUT_FOLDER="generated_output/voxpopuli"

python -m src.test_privacy \
    --model qwen_omni \
    --prepared_path "$PREPARED_PATH" \
    --out_folder "$OUT_FOLDER"
