#!/bin/bash

for DATASET in acl6060_context_word_fleurs_mixed acl6060_target_word_fleurs_mixed acl6060_both_fleurs_mixed; do
    echo "Starting training: $DATASET"
    python -m src.finetune_phi \
        --dataset "$DATASET" \
        --tts_jsonl data/tts/acl6060/en.jsonl \
        --ft_jsonl data/ft/fleurs_context/en.jsonl \
        --use_flash_attention \
        --batch_size 8 \
        --batch_size_per_gpu 1 \
        --num_train_epochs 2
    echo "Done: $DATASET"
done
