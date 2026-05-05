#!/bin/bash

# Per-dataset plots
for DATASET in fleurs acl6060 voxpopuli; do
    [ -d "generated_eval/${DATASET}" ] || continue
    echo "Plotting ${DATASET}..."
    PREFIX=""
    [ "$DATASET" != "fleurs" ] && PREFIX="${DATASET}_"
    python evaluation/plot_results.py \
        --eval_root "generated_eval/${DATASET}" \
        --out_dir "generated_eval/${DATASET}/plots" \
        --dataset_prefix "$PREFIX"
done

# Average over all datasets
echo "Plotting combined average..."
python evaluation/plot_results.py \
    --eval_root generated_eval \
    --out_dir generated_eval/combined/plots \
    --average_datasets fleurs acl6060 voxpopuli

# Similarity analysis
echo "Plotting similarity analysis..."
python evaluation/plot_results.py \
    --similarity_analysis generated_eval/similarity_analysis \
    --out_dir generated_eval/similarity_analysis/plots

# Distance analysis
echo "Plotting distance analysis..."
python evaluation/plot_results.py \
    --distance_analysis generated_eval/distance_analysis \
    --out_dir generated_eval/distance_analysis/plots
