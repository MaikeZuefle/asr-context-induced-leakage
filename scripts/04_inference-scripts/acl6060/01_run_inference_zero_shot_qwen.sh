#!/bin/bash

PREPARED_PATH="data/prepared/acl6060.jsonl"
OUT_FOLDER="generated_output/acl6060"

python -m src.test_privacy \
    --model qwen_omni \
    --prepared_path "$PREPARED_PATH" \
    --out_folder "$OUT_FOLDER"
