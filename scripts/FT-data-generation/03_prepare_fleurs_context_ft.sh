#!/bin/bash
# Generate context-aware finetuning data from FLEURS train split only.
# Steps:
#   1. Extract NEs from FLEURS transcripts, generate context sentences (LLM).
#      Audio already exists — no TTS needed.
#   2. Convert to LLaMA Factory ShareGPT format with context injected in the prompt.

GEMMA_MODEL_PATH="${GEMMA_MODEL_PATH:-google/gemma-3-12b-it}"
LANG="en"
FT_JSONL="data/ft/fleurs_context/${LANG}.jsonl"
EXISTING_PREPARED="data/prepared/${LANG}.jsonl"
DATA_DIR="data/llama_factory"

python -m data_preparation.prepare_fleurs_context_ft \
    --lang "$LANG" \
    --out_path "$FT_JSONL" \
    --existing_prepared_path "$EXISTING_PREPARED" \
    --splits train \
    --gemma_model_path "$GEMMA_MODEL_PATH" || exit 1

python -m data_preparation.prepare_llama_factory_context_data \
    --ft_jsonl "$FT_JSONL" \
    --data_dir "$DATA_DIR"
