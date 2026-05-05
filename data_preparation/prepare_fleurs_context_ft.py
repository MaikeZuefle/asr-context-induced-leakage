import argparse
import json
import os
import random

import soundfile as sf
import spacy
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline, set_seed

from data.utils import FLEURS_LANG_MAP
from data_preparation.utils import (
    _SECRET_ENTITY_TYPES,
    _build_scenario,
    generate_filler_sentences,
    generate_sentence_containing,
)

# Prompt — only the target word (the NE itself) goes in the context sentence

_CONTEXT_SENTENCE_PROMPT = """\
Here is a sentence from a spoken transcript:
"{reference}"

The word "{target_word}" appears in this transcript. Write exactly one short, natural \
sentence that:
- fits the same topic and register as the transcript
- uses the word "{target_word}" in the same role and context as it is used above

Return only the sentence, no explanation."""


def generate_context_sentence(reference: str, target_word: str, pipe) -> str:
    prompt = _CONTEXT_SENTENCE_PROMPT.format(reference=reference, target_word=target_word)
    return generate_sentence_containing(prompt, target_word, pipe)


# NER

def extract_ne_candidates(text: str, nlp) -> list[str]:
    """Return named entity proper nouns from text."""
    doc = nlp(text)
    return [
        tok.text for tok in doc
        if tok.ent_type_ in _SECRET_ENTITY_TYPES and tok.pos_ == "PROPN"
    ]


# FLEURS loading — metadata only, no audio saved yet

def load_fleurs_metadata(language: str, splits: list[str]) -> list[dict]:
    """Load FLEURS transcript metadata from the given splits without writing audio."""
    fleurs_lang = FLEURS_LANG_MAP[language]
    entries = []
    for split in splits:
        dataset = load_dataset(
            "google/fleurs", fleurs_lang, split=split, trust_remote_code=True
        )
        for entry in dataset:
            fleurs_id = entry["id"]
            wav_path = os.path.join("data/asr", f"fleurs_{language}_{split}_{fleurs_id}.wav")
            entries.append({
                "audio_path": wav_path,
                "transcript": entry["transcription"],
                "audio_array": entry["audio"]["array"],
                "sampling_rate": entry["audio"]["sampling_rate"],
                "fleurs_split": split,
                "fleurs_id": fleurs_id,
            })
    return entries


def save_audio(entry: dict) -> None:
    """Write audio to disk if not already there."""
    os.makedirs("data/asr", exist_ok=True)
    if not os.path.exists(entry["audio_path"]):
        sf.write(entry["audio_path"], entry["audio_array"], entry["sampling_rate"])


# Main

def main(language: str, out_path: str, gemma_model_path: str,
         existing_prepared_path: str | None, splits: list[str]):
    # Step 1: build set of audio paths already used in the evaluation dataset
    used_audio_paths: set[str] = set()
    if existing_prepared_path and os.path.exists(existing_prepared_path):
        with open(existing_prepared_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    used_audio_paths.add(json.loads(line)["audio_path"])
        print(f"Loaded {len(used_audio_paths)} already-used audio paths from {existing_prepared_path}")

    # Step 2: load metadata (no audio written yet), filter out eval samples
    print(f"Loading FLEURS {language} splits: {splits}")
    all_entries = load_fleurs_metadata(language, splits)
    entries = [e for e in all_entries if e["audio_path"] not in used_audio_paths]
    print(f"{len(entries)} entries after removing eval overlap ({len(all_entries) - len(entries)} dropped)")

    # Step 3: run NER and keep only entries that have at least one valid NE
    print("Loading spaCy NER model...")
    nlp = spacy.load("en_core_web_trf")

    candidates_by_entry = []
    for entry in tqdm(entries, desc="NER filtering"):
        candidates = extract_ne_candidates(entry["transcript"], nlp)
        if candidates:
            candidates_by_entry.append((entry, candidates))

    print(
        f"{len(candidates_by_entry)} entries with NEs "
        f"({len(entries) - len(candidates_by_entry)} dropped — no NE found)"
    )

    # Step 4: save audio only for the entries we'll actually use
    for entry, _ in candidates_by_entry:
        save_audio(entry)

    # Step 5: generate context sentences
    print(f"Loading LLM from {gemma_model_path} ...")
    pipe = hf_pipeline(
        "text-generation",
        model=gemma_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "eager"},
    )

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    # Resume support: skip audio_paths already written
    existing_written: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_written.add(json.loads(line)["audio_path"])
        print(f"Resuming: {len(existing_written)} entries already written.")

    kept = skipped_existing = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        for entry, candidates in tqdm(candidates_by_entry, desc="Generating context sentences"):
            if entry["audio_path"] in existing_written:
                skipped_existing += 1
                continue

            target_word = random.choice(candidates)
            context_sentence = generate_context_sentence(
                entry["transcript"], target_word, pipe
            )
            fillers = generate_filler_sentences(
                entry["transcript"], target_word, pipe, n=9
            )

            record = {
                "audio_path": entry["audio_path"],
                "transcript": entry["transcript"],
                "target_word": target_word,
                # 1-sentence context
                "context_sentence": context_sentence,
                # 5-sentence context (key sentence at random position among fillers)
                "context_sentences_5": _build_scenario([context_sentence], fillers, n=5),
                # 10-sentence context
                "context_sentences_10": _build_scenario([context_sentence], fillers, n=10),
                "fleurs_split": entry["fleurs_split"],
                "fleurs_id": entry["fleurs_id"],
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            f_out.flush()
            kept += 1

    print(
        f"Done. Wrote {kept} entries"
        + (f", skipped {skipped_existing} already written." if skipped_existing else ".")
    )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en")
    parser.add_argument("--out_path", default="data/ft/fleurs_context/en.jsonl")
    parser.add_argument(
        "--existing_prepared_path",
        default="data/prepared/en.jsonl",
        help="Path to eval prepared data; those audio paths are excluded.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="FLEURS splits to use (default: train). Options: train, validation, test.",
    )
    parser.add_argument("--gemma_model_path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    main(args.lang, args.out_path, args.gemma_model_path,
         args.existing_prepared_path, args.splits)
