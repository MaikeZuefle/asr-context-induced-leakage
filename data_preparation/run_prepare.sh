#!/bin/bash

# Run from the project root: bash data_preparation/run_prepare.sh

# Output is written to data_storage/prepared/<lang>.jsonl
# One JSON object per line:
# {
#   "audio_path":        "data_storage/asr/fleurs_en_42.wav",
#   "reference":         "...lord byron recorded its splendours...",
#   "target_word":       "byron",       # named entity actually spoken
#   "context_word":      "baron",       # phonetically similar substitute
#   "phoneme_distance":  1,
#   "context_sentence":  "The baron was known for his wit and poetry."
# }
# Samples with no named entity or no phonetic substitution are dropped.

python -m data_preparation.prepare \
    --lang en \
    --out_path data_storage/prepared/en.jsonl \
    --max_distance 2 \
    --gemma_model_path google/gemma-3-12b-it
