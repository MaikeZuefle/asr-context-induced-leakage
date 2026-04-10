"""
Convert TTS output JSONL into LLaMA Factory ShareGPT audio format and register
the datasets in LLaMA Factory's dataset_info.json.

Creates two datasets from data/tts/en.jsonl:
  - privacy_context_word: audio of context_sentence, transcript is the target
  - privacy_target_word:  audio of target_context_sentence, transcript is the target

Output format per entry:
    {
        "conversations": [
            {"from": "human", "value": "<audio>Please transcribe the audio."},
            {"from": "gpt",   "value": "<transcript>"}
        ],
        "audios": ["<audio_path>"]
    }

Audio paths in the JSON are stored relative to --data_dir so LLaMA Factory can
resolve them via its media_dir setting.

Also patches LlamaFactory/data/dataset_info.json with entries for both datasets.
"""

import argparse
import json
import os


INSTRUCTION = "Please transcribe the audio."

DATASET_NAMES = {
    "context_sentence":         "privacy_context_word",
    "target_context_sentence":  "privacy_target_word",
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


def main(tts_jsonl: str, data_dir: str):
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
    dataset_info_entries = {}
    for source, dataset_name in DATASET_NAMES.items():
        out_path = os.path.join(data_dir, f"{dataset_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entries[source], f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(entries[source])} entries to {out_path}")

        dataset_info_entries[dataset_name] = {
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

    # Write dataset_info.json into data_dir (where LLaMA Factory looks via dataset_dir)
    dataset_info_path = os.path.join(data_dir, "dataset_info.json")
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info_entries, f, ensure_ascii=False, indent=2)
    print(f"Written {dataset_info_path} with entries: {list(dataset_info_entries.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_jsonl", default="data/tts/en.jsonl", help="Path to TTS output JSONL")
    parser.add_argument("--data_dir", default="data/llama_factory", help="Output dir for dataset JSON files (stays in repo)")
    args = parser.parse_args()
    main(args.tts_jsonl, args.data_dir)
