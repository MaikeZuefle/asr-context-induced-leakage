#!/bin/bash
# All results go to generated_eval/ — dataset subfolders are derived automatically
# from the inference output paths (generated_output/fleurs/..., generated_output/acl6060/..., etc.)

run_eval() {
    local path="$1"
    [ -f "$path" ] || return
    echo "  $(basename $(dirname $(dirname $path)))/$(basename $(dirname $path))"
    python evaluation/evaluate.py \
        --results_path "$path" \
        --out_folder "generated_eval"
}

# ── FLEURS ────────────────────────────────────────────────────────────────────
echo "=== FLEURS ==="
for MODEL in "qwen_omni" "phi_multimodal"; do
    run_eval "generated_output/fleurs/${MODEL}/privacy/en.jsonl"
done

FT_MODELS=( "context_word" "target_word" "both"
            "fleurs_context_1" "fleurs_context_5" "fleurs_context_10" "fleurs_context_mixed"
            "context_word_fleurs_mixed" "target_word_fleurs_mixed" "both_fleurs_mixed" )

for BASE in "phi" "qwen"; do
    for MODEL in "${FT_MODELS[@]}"; do
        run_eval "generated_output_finetuned/fleurs/${BASE}/${MODEL}/privacy/en.jsonl"
    done
done

# ── ACL6060 ───────────────────────────────────────────────────────────────────
echo "=== ACL6060 ==="
for MODEL in "qwen_omni" "phi_multimodal"; do
    run_eval "generated_output/acl6060/${MODEL}/privacy/en.jsonl"
done

ACL_FT_MODELS=( "fleurs_context_mixed"
                "acl6060_context_word" "acl6060_target_word" "acl6060_both"
                "acl6060_context_word_fleurs_mixed" "acl6060_target_word_fleurs_mixed" "acl6060_both_fleurs_mixed" )

for BASE in "phi" "qwen"; do
    for MODEL in "${ACL_FT_MODELS[@]}"; do
        run_eval "generated_output_finetuned/acl6060/${BASE}/${MODEL}/privacy/en.jsonl"
    done
done

# ── VOXPOPULI ─────────────────────────────────────────────────────────────────
echo "=== VoxPopuli ==="
for MODEL in "qwen_omni" "phi_multimodal"; do
    run_eval "generated_output/voxpopuli/${MODEL}/privacy/en.jsonl"
done

VP_FT_MODELS=( "fleurs_context_mixed"
               "voxpopuli_context_word" "voxpopuli_target_word" "voxpopuli_both"
               "voxpopuli_context_word_fleurs_mixed" "voxpopuli_target_word_fleurs_mixed" "voxpopuli_both_fleurs_mixed" )

for BASE in "phi" "qwen"; do
    for MODEL in "${VP_FT_MODELS[@]}"; do
        run_eval "generated_output_finetuned/voxpopuli/${BASE}/${MODEL}/privacy/en.jsonl"
    done
done

# ── SIMILARITY ANALYSIS (FLEURS) ──────────────────────────────────────────────
echo "=== Similarity analysis ==="

run_sim_eval() {
    local results_path="$1"
    local prepared_path="$2"
    [ -f "$results_path" ] || return
    relative=$(echo "$results_path" | sed 's|^generated_output[^/]*/||')
    model_key="${relative%/privacy/en.jsonl}"   # e.g. fleurs/qwen_omni
    out_folder="generated_eval/similarity_analysis/${model_key}"
    echo "  $model_key"
    python evaluation/evaluate.py \
        --results_path "$results_path" \
        --out_folder "$out_folder" \
        --prepared_path "$prepared_path" \
        --similarity_groups
}

PREPARED_FLEURS="data/prepared/fleurs.jsonl"
run_sim_eval "generated_output/fleurs/qwen_omni/privacy/en.jsonl" "$PREPARED_FLEURS"
run_sim_eval "generated_output/fleurs/phi_multimodal/privacy/en.jsonl" "$PREPARED_FLEURS"
run_sim_eval "generated_output_finetuned/fleurs/qwen/fleurs_context_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_sim_eval "generated_output_finetuned/fleurs/phi/fleurs_context_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_sim_eval "generated_output_finetuned/fleurs/qwen/context_word_fleurs_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_sim_eval "generated_output_finetuned/fleurs/phi/context_word_fleurs_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"

# ── DISTANCE ANALYSIS (FLEURS) ────────────────────────────────────────────────
echo "=== Distance analysis ==="

run_dist_eval() {
    local results_path="$1"
    local prepared_path="$2"
    [ -f "$results_path" ] || return
    relative=$(echo "$results_path" | sed 's|^generated_output[^/]*/||')
    model_key="${relative%/privacy/en.jsonl}"
    out_folder="generated_eval/distance_analysis/${model_key}"
    echo "  $model_key"
    python evaluation/evaluate.py \
        --results_path "$results_path" \
        --out_folder "$out_folder" \
        --prepared_path "$prepared_path" \
        --distance_groups
}

run_dist_eval "generated_output/fleurs/qwen_omni/privacy/en.jsonl" "$PREPARED_FLEURS"
run_dist_eval "generated_output/fleurs/phi_multimodal/privacy/en.jsonl" "$PREPARED_FLEURS"
run_dist_eval "generated_output_finetuned/fleurs/qwen/fleurs_context_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_dist_eval "generated_output_finetuned/fleurs/phi/fleurs_context_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_dist_eval "generated_output_finetuned/fleurs/qwen/context_word_fleurs_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"
run_dist_eval "generated_output_finetuned/fleurs/phi/context_word_fleurs_mixed/privacy/en.jsonl" "$PREPARED_FLEURS"

# ── COMBINED average ──────────────────────────────────────────────────────────
echo "=== Combined average ==="
python evaluation/evaluate.py \
    --average_datasets generated_eval/fleurs generated_eval/acl6060 generated_eval/voxpopuli \
    --out_folder generated_eval/combined

echo "Done."
