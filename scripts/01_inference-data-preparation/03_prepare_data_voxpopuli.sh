#!/bin/bash

OUT_PATH="data/prepared/voxpopuli.jsonl"
MAX_DISTANCE=2
GEMMA_MODEL_PATH="google/gemma-3-12b-it"

python -m data_preparation.prepare_voxpopuli \
    --out_path "$OUT_PATH" \
    --max_distance "$MAX_DISTANCE" \
    --gemma_model_path "$GEMMA_MODEL_PATH" \
    --splits test
