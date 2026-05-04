"""Same pipeline as prepare.py but for VoxPopuli English test split."""

import argparse
import json
import os

import spacy
import torch
from tqdm import tqdm
from transformers import pipeline as hf_pipeline, set_seed

from data.voxpopuli import load_asr
from data_preparation.prepare import (
    extract_candidate_words,
    find_similar_word,
    generate_context_sentence,
    generate_target_sentence,
)
from data_preparation.utils import _build_scenario, generate_filler_sentences

set_seed(42)


def main(out_path: str, max_distance: int, gemma_model_path: str, splits: tuple):
    nlp = spacy.load("en_core_web_trf")
    pipe = hf_pipeline(
        "text-generation",
        model=gemma_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "eager"},
    )

    data = load_asr(splits=splits)
    audio_paths = data["inputs"]
    references = data["references"]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    kept, dropped = 0, 0
    seen_pairs: set[tuple[str, str, str]] = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f_existing:
            for line in f_existing:
                if line.strip():
                    s = json.loads(line)
                    seen_pairs.add((s["audio_path"], s["target_word"], s["context_word"]))
        print(f"Resuming: {len(seen_pairs)} already-processed pairs found.")

    with open(out_path, "a", encoding="utf-8") as f_out:
        for audio_path, reference in tqdm(
            zip(audio_paths, references), total=len(audio_paths), desc="Preparing VoxPopuli"
        ):
            candidates = extract_candidate_words(reference, nlp)

            substitution_found = False
            for word in candidates:
                context_word, dist = find_similar_word(word, max_distance=max_distance)
                if context_word is not None:
                    if (audio_path, word, context_word) in seen_pairs:
                        continue
                    seen_pairs.add((audio_path, word, context_word))

                    ctx_sent  = generate_context_sentence(reference, word, context_word, pipe)
                    tgt_sent  = generate_target_sentence(reference, word, pipe)
                    fillers   = generate_filler_sentences(reference, word, pipe, n=9, context_word=context_word)

                    record = {
                        "audio_path": audio_path,
                        "reference":  reference,
                        "target_word":  word,
                        "context_word": context_word,
                        "phoneme_distance": dist,
                        "context_sentence":        ctx_sent,
                        "target_context_sentence": tgt_sent,
                        "mixed_sentences":         [ctx_sent, tgt_sent],
                        "context_sentences_5":        _build_scenario([ctx_sent], fillers, n=5),
                        "target_context_sentences_5": _build_scenario([tgt_sent], fillers, n=5),
                        "mixed_sentences_5":          _build_scenario([ctx_sent, tgt_sent], fillers, n=5),
                        "context_sentences_10":        _build_scenario([ctx_sent], fillers, n=10),
                        "target_context_sentences_10": _build_scenario([tgt_sent], fillers, n=10),
                        "mixed_sentences_10":          _build_scenario([ctx_sent, tgt_sent], fillers, n=10),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    substitution_found = True
                    break

            if substitution_found:
                kept += 1
            else:
                dropped += 1

    print(f"Done. Kept {kept} samples, dropped {dropped} (no substitution found).")
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="data/prepared/voxpopuli.jsonl")
    parser.add_argument("--max_distance", type=int, default=2)
    parser.add_argument("--gemma_model_path", default="google/gemma-3-12b-it")
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args()
    main(args.out_path, args.max_distance, args.gemma_model_path, tuple(args.splits))
