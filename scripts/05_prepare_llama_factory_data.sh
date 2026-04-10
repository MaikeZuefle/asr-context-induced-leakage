#!/bin/bash

TTS_JSONL="data/tts/en.jsonl"
DATA_DIR="data/llama_factory"

python -m data_preparation.prepare_llama_factory_data \
    --tts_jsonl "$TTS_JSONL" \
    --data_dir "$DATA_DIR"
