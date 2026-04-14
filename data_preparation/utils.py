"""Shared utilities for data preparation scripts."""

import random
import warnings

# Entity types that represent "secret-like" information.
# Excludes DATE, TIME, CARDINAL, ORDINAL, PERCENT, MONEY, QUANTITY.
_SECRET_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "FAC"}

_FILLER_SENTENCES_PROMPT_TWO_WORDS = """\
Here is a sentence from a spoken transcript:
"{reference}"

Write exactly {n} short, natural sentences that:
- fit the same topic and register as the transcript
- do not contain the words "{target_word}" or "{context_word}"

Return exactly {n} sentences, one per line, no numbering, no explanation."""

_FILLER_SENTENCES_PROMPT_ONE_WORD = """\
Here is a sentence from a spoken transcript:
"{reference}"

Write exactly {n} short, natural sentences that:
- fit the same topic and register as the transcript
- do not contain the word "{target_word}"

Return exactly {n} sentences, one per line, no numbering, no explanation."""


def _run_pipe(pipe, prompt: str, max_new_tokens: int) -> str:
    messages = [{"role": "user", "content": prompt}]
    output = pipe(messages, max_new_tokens=max_new_tokens)
    return output[0]["generated_text"][-1]["content"].strip()


def _parse_sentences(text: str, n: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:n]


def generate_sentence_containing(
    prompt: str,
    word: str,
    pipe,
    max_new_tokens: int = 64,
    max_retries: int = 3,
) -> str:
    """Generate a sentence and retry if it does not contain `word`."""
    sentence = ""
    for _ in range(max_retries):
        sentence = _run_pipe(pipe, prompt, max_new_tokens)
        if word.lower() in sentence.lower():
            return sentence
    warnings.warn(
        f"Generated sentence does not contain '{word}' after {max_retries} retries. "
        f"Keeping last attempt: {sentence!r}"
    )
    return sentence


def generate_filler_sentences(
    reference: str,
    target_word: str,
    pipe,
    n: int = 9,
    context_word: str | None = None,
    max_retries: int = 3,
) -> list[str]:
    if context_word is not None:
        prompt = _FILLER_SENTENCES_PROMPT_TWO_WORDS.format(
            reference=reference, target_word=target_word, context_word=context_word, n=n
        )
    else:
        prompt = _FILLER_SENTENCES_PROMPT_ONE_WORD.format(
            reference=reference, target_word=target_word, n=n
        )
    for _ in range(max_retries):
        sentences = _parse_sentences(_run_pipe(pipe, prompt, max_new_tokens=n * 40), n)
        if len(sentences) >= n:
            return sentences
    raise RuntimeError(f"Failed to generate {n} filler sentences after {max_retries} retries.")


def _build_scenario(key_sentences: list[str], fillers: list[str], n: int) -> list[str]:
    """Insert each key sentence at a random position into the first n-len(keys) fillers."""
    pool = list(fillers[:n - len(key_sentences)])
    for key in key_sentences:
        pos = random.randint(0, len(pool))
        pool = pool[:pos] + [key] + pool[pos:]
    return pool
