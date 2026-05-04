#!/bin/bash

PREPARED_PATH="data/prepared/acl6060.jsonl"
OUT_FOLDER="generated_output/acl6060"

python -m src.test_privacy \
    --model phi_multimodal \
    --prepared_path "$PREPARED_PATH" \
    --out_folder "$OUT_FOLDER"
