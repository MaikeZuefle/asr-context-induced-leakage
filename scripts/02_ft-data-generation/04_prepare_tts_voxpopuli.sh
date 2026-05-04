#!/bin/bash
set -e

PREPARED_PATH="data/prepared/voxpopuli.jsonl"
OUT_DIR="data/tts/voxpopuli"
TTS_JSONL="data/tts/voxpopuli/en.jsonl"
DATA_DIR="data/llama_factory"

python -m data_preparation.tts_sentences \
    --prepared_path "$PREPARED_PATH" \
    --out_dir "$OUT_DIR" \
    --out_jsonl "$TTS_JSONL"

python -m data_preparation.prepare_llama_factory_data \
    --tts_jsonl "$TTS_JSONL" \
    --data_dir "$DATA_DIR" \
    --prefix "voxpopuli"
