import argparse
import json
import os


INSTRUCTION = "Please transcribe the audio."

DATASET_NAMES = {
    "context_sentence":         "context_word",
    "target_context_sentence":  "target_word",
}


def build_entry(audio_path: str, transcript: str, data_dir: str) -> dict:
    rel_audio = os.path.relpath(audio_path)  # relative to repo root (cwd)
    return {
        "conversations": [
            {"from": "human", "value": f"<audio>{INSTRUCTION}"},
            {"from": "gpt",   "value": transcript},
        ],
        "audios": [rel_audio],
    }


def main(tts_jsonl: str, data_dir: str, prefix: str = "privacy"):
    os.makedirs(data_dir, exist_ok=True)

    with open(tts_jsonl, "r", encoding="utf-8") as f:
        tts_samples = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(tts_samples)} TTS entries from {tts_jsonl}")

    # Collect entries per source
    entries: dict[str, list] = {source: [] for source in DATASET_NAMES}
    for sample in tts_samples:
        source = sample["source"]
        if source not in entries:
            continue
        entries[source].append(build_entry(sample["audio_path"], sample["transcript"], data_dir))

    # Write dataset JSON files
    new_entries = {}
    for source, base_name in DATASET_NAMES.items():
        dataset_name = f"{prefix}_{base_name}"
        out_path = os.path.join(data_dir, f"{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries[source], f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(entries[source])} entries to {out_path}")

        new_entries[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "audios": "audios"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    # Merge into existing dataset_info.json
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    existing = {}
    if os.path.exists(dataset_info_path):
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.update(new_entries)
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
    print(f"Updated {dataset_info_path} — datasets: {list(existing.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_jsonl", default="data/tts/fleurs/en.jsonl")
    parser.add_argument("--data_dir", default="data/llama_factory")
    parser.add_argument("--prefix", default="privacy", help="Dataset name prefix (e.g. 'privacy' or 'acl6060')")
    args = parser.parse_args()
    main(args.tts_jsonl, args.data_dir, prefix=args.prefix)
