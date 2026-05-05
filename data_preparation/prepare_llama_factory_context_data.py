import argparse
import json
import os
import random

DATASET_CONFIGS = {
    "privacy_fleurs_context_1":  "context_sentence",
    "privacy_fleurs_context_5":  "context_sentences_5",
    "privacy_fleurs_context_10": "context_sentences_10",
}

_MIXED_FIELDS = ["context_sentence", "context_sentences_5", "context_sentences_10"]

INSTRUCTION_TEMPLATE = "Context: {context}\n\nPlease transcribe the audio."


def format_context(value: str | list[str]) -> str:
    """Join a list of sentences into a single string, or return as-is if already a string."""
    if isinstance(value, list):
        return " ".join(value)
    return value


def build_entry(audio_path: str, transcript: str, context: str | list[str]) -> dict:
    instruction = INSTRUCTION_TEMPLATE.format(context=format_context(context))
    return {
        "conversations": [
            {"from": "human", "value": f"<audio>{instruction}"},
            {"from": "gpt",   "value": transcript},
        ],
        "audios": [os.path.relpath(audio_path)],
    }


def main(ft_jsonl: str, data_dir: str):
    os.makedirs(data_dir, exist_ok=True)

    with open(ft_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(samples)} entries from {ft_jsonl}")

    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    for dataset_name, field in DATASET_CONFIGS.items():
        entries = [
            build_entry(s["audio_path"], s["transcript"], s[field])
            for s in samples
        ]
        out_path = os.path.join(data_dir, f"{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(entries)} entries to {out_path}")

        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "audios": "audios",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    # mixed: one context length sampled uniformly at random per example (seeded for reproducibility)
    rng = random.Random(42)
    mixed_entries = [
        build_entry(s["audio_path"], s["transcript"], s[rng.choice(_MIXED_FIELDS)])
        for s in samples
    ]
    mixed_name = "privacy_fleurs_context_mixed"
    mixed_path = os.path.join(data_dir, f"{mixed_name}.json")
    with open(mixed_path, "w", encoding="utf-8") as f:
        json.dump(mixed_entries, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(mixed_entries)} entries to {mixed_path}")

    dataset_info[mixed_name] = {
        "file_name": f"{mixed_name}.json",
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "audios": "audios",
        },
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"Updated {dataset_info_path} — datasets: {list(dataset_info.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_jsonl", default="data/ft/fleurs_context/en.jsonl")
    parser.add_argument("--data_dir", default="data/llama_factory")
    args = parser.parse_args()
    main(args.ft_jsonl, args.data_dir)
