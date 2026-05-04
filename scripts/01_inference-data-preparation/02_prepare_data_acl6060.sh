#!/bin/bash

OUT_PATH="data/prepared/acl6060.jsonl"
MAX_DISTANCE=2
GEMMA_MODEL_PATH="google/gemma-3-12b-it"

python -m data_preparation.prepare_acl6060 \
    --out_path "$OUT_PATH" \
    --max_distance "$MAX_DISTANCE" \
    --gemma_model_path "$GEMMA_MODEL_PATH"
