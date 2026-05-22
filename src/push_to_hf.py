"""Push the ASR privacy leakage dataset to HuggingFace Hub."""
import argparse
import json
import os
from difflib import SequenceMatcher

import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, Features, Sequence, Value

HF_REPO = "PLACEHOLDER/asr-privacy-leakage"  # set via --repo argument

FEATURES = Features({
    "audio":           Audio(sampling_rate=None),
    "reference":       Value("string"),
    "acoustic_word":   Value("string"),
    "context_word":    Value("string"),
    "phoneme_distance": Value("int32"),
    "dataset":         Value("string"),
    "similarity":      Value("float32"),
    "ctx_1_context":   Value("string"),
    "ctx_1_acoustic":  Value("string"),
    "ctx_2_mixed":     Sequence(Value("string")),
    "ctx_5_context":   Sequence(Value("string")),
    "ctx_5_acoustic":  Sequence(Value("string")),
    "ctx_5_mixed":     Sequence(Value("string")),
    "ctx_10_context":  Sequence(Value("string")),
    "ctx_10_acoustic": Sequence(Value("string")),
    "ctx_10_mixed":    Sequence(Value("string")),
})

EMPTY_ROW = {
    "audio":            None,
    "reference":        None,
    "acoustic_word":    None,
    "context_word":     None,
    "phoneme_distance": None,
    "dataset":          None,
    "similarity":       None,
    "ctx_1_context":    None,
    "ctx_1_acoustic":   None,
    "ctx_2_mixed":      [],
    "ctx_5_context":    [],
    "ctx_5_acoustic":   [],
    "ctx_5_mixed":      [],
    "ctx_10_context":   [],
    "ctx_10_acoustic":  [],
    "ctx_10_mixed":     [],
}


def load_audio(path):
    if not path or not os.path.exists(path):
        return None
    data, sr = sf.read(path)
    return {"path": path, "array": data, "sampling_rate": sr}


def load_test():
    rows = []
    for ds in ["fleurs", "acl6060", "voxpopuli"]:
        with open(f"data/prepared/{ds}.jsonl") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                sim = SequenceMatcher(
                    None, d["reference"].lower(), d["context_sentence"].lower()
                ).ratio()
                rows.append({
                    **EMPTY_ROW,
                    "audio":            load_audio(d["audio_path"]),
                    "reference":        d["reference"],
                    "acoustic_word":    d["target_word"],
                    "context_word":     d["context_word"],
                    "phoneme_distance": d["phoneme_distance"],
                    "dataset":          ds,
                    "similarity":       round(sim, 4),
                    "ctx_1_context":    d["context_sentence"],
                    "ctx_1_acoustic":   d["target_context_sentence"],
                    "ctx_2_mixed":      d["mixed_sentences"],
                    "ctx_5_context":    d["context_sentences_5"],
                    "ctx_5_acoustic":   d["target_context_sentences_5"],
                    "ctx_5_mixed":      d["mixed_sentences_5"],
                    "ctx_10_context":   d["context_sentences_10"],
                    "ctx_10_acoustic":  d["target_context_sentences_10"],
                    "ctx_10_mixed":     d["mixed_sentences_10"],
                })
    return rows


def load_prompt_adapt():
    rows = []
    with open("data/ft/fleurs_context/en.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rows.append({
                **EMPTY_ROW,
                "audio":          load_audio(d["audio_path"]),
                "reference":      d["transcript"],
                "acoustic_word":  d["target_word"],
                "dataset":        "fleurs",
                "ctx_1_acoustic": d["context_sentence"],
                "ctx_5_acoustic": d["context_sentences_5"],
                "ctx_10_acoustic": d["context_sentences_10"],
            })
    return rows


def load_tts_split(tts_filename_key, ctx_col):
    """Load a TTS-based fine-tuning split.

    tts_filename_key: 'context_sentence' or 'target_context_sentence'
    ctx_col: which ctx_* column to fill with the sentence text
    """
    rows = []
    for ds in ["fleurs", "acl6060", "voxpopuli"]:
        items = []
        with open(f"data/prepared/{ds}.jsonl") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
        for idx, d in enumerate(items):
            wav = f"data/tts/{ds}/{idx}_{tts_filename_key}.wav"
            audio = load_audio(wav)
            if audio is None:
                print(f"  Warning: missing {wav}")
                continue
            row = {
                **EMPTY_ROW,
                "audio":            audio,
                "reference":        d[tts_filename_key],
                "acoustic_word":    d["target_word"],
                "context_word":     d["context_word"],
                "phoneme_distance": d["phoneme_distance"],
                "dataset":          ds,
                ctx_col:            d[tts_filename_key],
            }
            rows.append(row)
    return rows


def load_both_words():
    return load_tts_split("context_sentence", "ctx_1_context") + \
           load_tts_split("target_context_sentence", "ctx_1_acoustic")


def build_dataset(rows):
    return Dataset.from_list(rows, features=FEATURES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=HF_REPO, help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    print("Loading test split...")
    test_rows = load_test()
    print(f"  {len(test_rows)} examples")

    print("Loading train_prompt_adapt split...")
    prompt_adapt_rows = load_prompt_adapt()
    print(f"  {len(prompt_adapt_rows)} examples")

    print("Loading train_context_word split...")
    context_word_rows = load_tts_split("context_sentence", "ctx_1_context")
    print(f"  {len(context_word_rows)} examples")

    print("Loading train_acoustic_word split...")
    acoustic_word_rows = load_tts_split("target_context_sentence", "ctx_1_acoustic")
    print(f"  {len(acoustic_word_rows)} examples")

    print("Loading train_both_words split...")
    both_words_rows = load_both_words()
    print(f"  {len(both_words_rows)} examples")

    print("Building DatasetDict...")
    dd = DatasetDict({
        "test":               build_dataset(test_rows),
        "train_prompt_adapt": build_dataset(prompt_adapt_rows),
        "train_context_word": build_dataset(context_word_rows),
        "train_acoustic_word": build_dataset(acoustic_word_rows),
        "train_both_words":   build_dataset(both_words_rows),
    })

    print(dd)
    print(f"Pushing to {args.repo}...")
    dd.push_to_hub(args.repo, private=args.private)
    print("Done.")
