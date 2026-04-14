#!/bin/bash

python -m src.finetune_phi \
    --dataset context_word \
    --tts_jsonl data/tts/en.jsonl \
    --use_flash_attention \
    --batch_size 8 \
    --batch_size_per_gpu 1 \
    --num_train_epochs 2
