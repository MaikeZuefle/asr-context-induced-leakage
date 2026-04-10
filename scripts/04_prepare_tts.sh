#!/bin/bash

PREPARED_PATH="data/prepared/en.jsonl"
OUT_DIR="data/tts"
OUT_JSONL="data/tts/en.jsonl"

python -m data_preparation.tts_sentences \
    --prepared_path "$PREPARED_PATH" \
    --out_dir "$OUT_DIR" \
    --out_jsonl "$OUT_JSONL"
