#!/bin/bash

LANG="en"
OUT_PATH="data/prepared/${LANG}.jsonl"
MAX_DISTANCE=2
GEMMA_MODEL_PATH="google/gemma-3-12b-it"

python -m data_preparation.prepare \
    --lang "$LANG" \
    --out_path "$OUT_PATH" \
    --max_distance "$MAX_DISTANCE" \
    --gemma_model_path "$GEMMA_MODEL_PATH"
