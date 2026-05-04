#!/bin/bash

for DATASET in context_word target_word both; do
    echo "Starting training: $DATASET"
    python -m src.finetune_phi \
        --dataset "$DATASET" \
        --tts_jsonl data/tts/fleurs/en.jsonl \
        --use_flash_attention \
        --batch_size 8 \
        --batch_size_per_gpu 1 \
        --num_train_epochs 2
    echo "Done: $DATASET"
done
