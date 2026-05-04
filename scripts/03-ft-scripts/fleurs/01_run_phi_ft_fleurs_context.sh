#!/bin/bash

for DATASET in fleurs_context_1 fleurs_context_5 fleurs_context_10 fleurs_context_mixed; do
    echo "Starting training: $DATASET"
    python -m src.finetune_phi \
        --dataset "$DATASET" \
        --ft_jsonl data/ft/fleurs_context/en.jsonl \
        --use_flash_attention \
        --batch_size 8 \
        --batch_size_per_gpu 1 \
        --num_train_epochs 2
    echo "Done: $DATASET"
done
