#!/bin/bash

# Base models
OUT_FOLDER="generated_output"
MODELS=( "qwen_omni" "phi_multimodal" )

for MODEL in "${MODELS[@]}"; do
    RESULTS_PATH="${OUT_FOLDER}/${MODEL}/privacy/en.jsonl"
    echo "=== $MODEL ==="
    python evaluation/evaluate.py --results_path "$RESULTS_PATH"
    echo ""
done

# Fine-tuned models
FT_MODELS=( "context_word" "target_word" "both" "fleurs_context_1" "fleurs_context_5" "fleurs_context_10" "fleurs_context_mixed" )

for BASE in "phi" "qwen"; do
    for MODEL in "${FT_MODELS[@]}"; do
        RESULTS_PATH="generated_output_finetuned/${BASE}/${MODEL}/privacy/en.jsonl"
        [ -f "$RESULTS_PATH" ] || continue
        echo "=== finetuned/${BASE}/${MODEL} ==="
        python evaluation/evaluate.py \
            --results_path "$RESULTS_PATH" \
            --out_folder "generated_eval/finetuned"
        echo ""
    done
done
