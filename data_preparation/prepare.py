"""
For each FLEURS sample, extract a named entity from the transcript,
find a phonetically similar word via CMU pronouncing dict, then use an LLM
to generate a context sentence containing that word. Samples with no suitable
substitution are dropped.

Output format (one JSON object per line):
    {
        "audio_path": "...",
        "reference": "...",
        "target_word": "byron",          # named entity actually spoken
        "context_word": "baron",         # phonetically similar substitute (not a morphological variant)
        "phoneme_distance": 1,
        # 1-sentence contexts
        "context_sentence": "The baron was known for his wit and poetry.",
        "target_context_sentence": "Lord Byron visited the area in 1809.",
        "mixed_sentences": ["Unlike the baron, Lord Byron preferred solitude.", "..."],  # context + target sentence
        # 5-sentence contexts (lists), one sentence contains the key word(s)
        "context_sentences_5": ["...", "...", "...", "...", "..."],
        "target_context_sentences_5": ["...", "...", "...", "...", "..."],
        "mixed_sentences_5": ["...", "...", "...", "...", "..."],
        # 10-sentence contexts (lists), one sentence contains the key word(s)
        "context_sentences_10": ["...", ...],
        "target_context_sentences_10": ["...", ...],
        "mixed_sentences_10": ["...", ...]
    }

Requirements:
    pip install nltk spacy torch transformers
    python -m nltk.downloader cmudict
    python -m spacy download en_core_web_trf
"""

import argparse
import json
import os
import random
from collections import defaultdict

import nltk
import spacy
import torch
from nltk.corpus import cmudict
from nltk.stem import PorterStemmer
from tqdm import tqdm
from transformers import pipeline as hf_pipeline, set_seed

from data.fleurs import load_asr
from data_preparation.utils import (
    _SECRET_ENTITY_TYPES,
    _build_scenario,
    generate_filler_sentences,
    generate_sentence_containing,
)

nltk.download("cmudict", quiet=True)

_STEMMER = PorterStemmer()


def _are_morphological_variants(word1: str, word2: str) -> bool:
    """Return True if the two words share the same Porter stem (e.g. mariana/marianas)."""
    return _STEMMER.stem(word1.lower()) == _STEMMER.stem(word2.lower())


# ---------------------------------------------------------------------------
# Phoneme index built once at import time
# ---------------------------------------------------------------------------

_CMU_DICT = cmudict.dict()

# Index: first_phoneme -> [(word, phoneme_list), ...]
# Restricting search to same first phoneme cuts the candidate space ~40x.
_PHONEME_INDEX: dict[str, list] = defaultdict(list)
for _word, _pronunciations in _CMU_DICT.items():
    _pron = _pronunciations[0]
    if _pron:
        _PHONEME_INDEX[_pron[0]].append((_word, _pron))


def _get_phonemes(word: str) -> list[str] | None:
    entries = _CMU_DICT.get(word.lower())
    return entries[0] if entries else None


def _edit_distance(p1: list[str], p2: list[str]) -> int:
    """Levenshtein distance on phoneme sequences."""
    m, n = len(p1), len(p2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if p1[i - 1] == p2[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def find_similar_word(word: str, max_distance: int = 2) -> tuple[str | None, int | None]:
    """Return (candidate_word, phoneme_distance) or (None, None)."""
    phonemes = _get_phonemes(word)
    if phonemes is None:
        return None, None

    first_ph = phonemes[0]
    candidates = _PHONEME_INDEX.get(first_ph, [])

    best_word, best_dist = None, max_distance + 1
    for candidate, pron in candidates:
        if candidate.lower() == word.lower():
            continue
        if _are_morphological_variants(candidate, word):
            continue
        # Skip words with very different lengths (they won't sound similar)
        if abs(len(pron) - len(phonemes)) > 2:
            continue
        dist = _edit_distance(phonemes, pron)
        if 1 <= dist < best_dist:
            best_dist = dist
            best_word = candidate

    if best_word is None:
        return None, None
    return best_word, best_dist


# ---------------------------------------------------------------------------
# Candidate word extraction
# ---------------------------------------------------------------------------

def extract_candidate_words(text: str, nlp) -> list[str]:
    """
    Return named entity tokens from text, restricted to secret-like entity types,
    filtered to words present in CMU dict. Samples with no such entities are dropped.
    """
    doc = nlp(text)
    ne_words = [tok.text for tok in doc if tok.ent_type_ in _SECRET_ENTITY_TYPES and tok.pos_ == "PROPN"]
    return [w for w in ne_words if _get_phonemes(w) is not None]


# ---------------------------------------------------------------------------
# Context sentence generation
# ---------------------------------------------------------------------------

_CONTEXT_SENTENCE_PROMPT = """\
Here is a sentence from a spoken transcript:
"{reference}"

The word "{target_word}" appears in this transcript. Write exactly one short, natural sentence that:
- fits the same topic and register as the transcript
- uses the word "{context_word}" in the same role and context as "{target_word}" is used above

Return only the sentence, no explanation."""

_TARGET_SENTENCE_PROMPT = """\
Here is a sentence from a spoken transcript:
"{reference}"

Write exactly one short, natural sentence that:
- fits the same topic and register as the transcript
- uses the word "{target_word}" in the same role and context as it is used above

Return only the sentence, no explanation."""


def generate_context_sentence(reference: str, target_word: str, context_word: str, pipe) -> str:
    prompt = _CONTEXT_SENTENCE_PROMPT.format(reference=reference, target_word=target_word, context_word=context_word)
    return generate_sentence_containing(prompt, context_word, pipe)


def generate_target_sentence(reference: str, target_word: str, pipe) -> str:
    prompt = _TARGET_SENTENCE_PROMPT.format(reference=reference, target_word=target_word)
    return generate_sentence_containing(prompt, target_word, pipe)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(language: str, out_path: str, max_distance: int, gemma_model_path: str):
    nlp = spacy.load("en_core_web_trf")
    pipe = hf_pipeline(
        "text-generation",
        model=gemma_model_path,
        device="cuda",
        torch_dtype=torch.bfloat16,
        model_kwargs={"attn_implementation": "eager"},
    )

    data = load_asr(language)
    audio_paths = data["inputs"]
    references = data["references"]

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    kept, dropped = 0, 0
    with open(out_path, "w", encoding="utf-8") as f_out:
        for audio_path, reference in tqdm(
            zip(audio_paths, references), total=len(audio_paths), desc="Preparing"
        ):
            candidates = extract_candidate_words(reference, nlp)

            substitution_found = False
            for word in candidates:
                context_word, dist = find_similar_word(word, max_distance=max_distance)
                if context_word is not None:
                    # 3 LLM calls; all multi-sentence scenarios are assembled from these
                    ctx_sent = generate_context_sentence(reference, word, context_word, pipe)
                    tgt_sent = generate_target_sentence(reference, word, pipe)
                    fillers = generate_filler_sentences(reference, word, pipe, n=9, context_word=context_word)

                    record = {
                        "audio_path": audio_path,
                        "reference": reference,
                        "target_word": word,
                        "context_word": context_word,
                        "phoneme_distance": dist,
                        # 1-sentence contexts
                        "context_sentence": ctx_sent,
                        "target_context_sentence": tgt_sent,
                        "mixed_sentences": [ctx_sent, tgt_sent],
                        # 5-sentence contexts (1 key + 4 fillers, key at random position)
                        "context_sentences_5": _build_scenario([ctx_sent], fillers, n=5),
                        "target_context_sentences_5": _build_scenario([tgt_sent], fillers, n=5),
                        "mixed_sentences_5": _build_scenario([ctx_sent, tgt_sent], fillers, n=5),
                        # 10-sentence contexts (1 or 2 keys + fillers, keys at random positions)
                        "context_sentences_10": _build_scenario([ctx_sent], fillers, n=10),
                        "target_context_sentences_10": _build_scenario([tgt_sent], fillers, n=10),
                        "mixed_sentences_10": _build_scenario([ctx_sent, tgt_sent], fillers, n=10),
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    substitution_found = True
                    break  # one substitution per sample
            if substitution_found:
                kept += 1
            else:
                dropped += 1

    print(f"Done. Kept {kept} samples, dropped {dropped} (no substitution found).")
    print(f"Output written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="en", help="Language code (currently English only)")
    parser.add_argument("--out_path", default="data_storage/prepared/en.jsonl")
    parser.add_argument("--max_distance", type=int, default=2, help="Max phoneme edit distance")
    parser.add_argument("--gemma_model_path", required=True, help="Path to Gemma 12B model")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for filler insertion positions")
    args = parser.parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    main(args.lang, args.out_path, args.max_distance, args.gemma_model_path)
