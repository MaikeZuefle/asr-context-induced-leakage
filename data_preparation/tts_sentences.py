"""
Generate TTS audio for context_sentence and target_context_sentence fields
from the prepared privacy evaluation data, using Kokoro TTS with randomly
sampled English voices.

Output format (one JSON object per line):
    {
        "audio_path": "data/tts/42_context.wav",
        "transcript": "The baron was known for his wit and poetry.",
        "voice": "af_heart",
        "source": "context_sentence",   # or "target_context_sentence"
        "sample_idx": 42,
        "target_word": "byron",
        "context_word": "baron"
    }
"""

import argparse
import json
import os
import random

import numpy as np
import soundfile as sf
from kokoro import KPipeline
from tqdm import tqdm
from transformers import set_seed

SAMPLE_RATE = 24000

# American English voices only — consistent with CMU pronouncing dict phoneme distances
ENGLISH_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica",
    "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam",
    "am_michael", "am_onyx", "am_puck",
]

_PIPELINE = None


def _get_pipeline() -> KPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = KPipeline(lang_code="a")
    return _PIPELINE


def synthesize(text: str, voice: str) -> np.ndarray:
    pipe = _get_pipeline()
    chunks = [audio for _, _, audio in pipe(text, voice=voice)]
    return np.concatenate(chunks)


def main(prepared_path: str, out_dir: str, out_jsonl: str):
    os.makedirs(out_dir, exist_ok=True)
    jsonl_dir = os.path.dirname(out_jsonl)
    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    with open(prepared_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(samples)} samples from {prepared_path}")
    print("Loading Kokoro pipeline (American English)...")

    # Load existing outputs to resume interrupted runs
    existing = set()
    if os.path.exists(out_jsonl):
        with open(out_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["audio_path"])
        print(f"Resuming: {len(existing)} audio files already generated.")

    processed = skipped = 0
    with open(out_jsonl, "a", encoding="utf-8") as f_out:
        for idx, sample in enumerate(tqdm(samples, desc="TTS")):
            for source in ("context_sentence", "target_context_sentence"):
                audio_path = os.path.join(out_dir, f"{idx}_{source}.wav")
                if audio_path in existing:
                    skipped += 1
                    continue

                voice = random.choice(ENGLISH_VOICES)
                text = sample[source]

                audio = synthesize(text, voice)
                sf.write(audio_path, audio, SAMPLE_RATE)

                record = {
                    "audio_path":  audio_path,
                    "transcript":  text,
                    "voice":       voice,
                    "source":      source,
                    "sample_idx":  idx,
                    "target_word": sample["target_word"],
                    "context_word": sample["context_word"],
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()
                processed += 1

    print(f"Done. Generated {processed} audio files, skipped {skipped} existing.")
    print(f"Output written to {out_jsonl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared_path", default="data/prepared/en.jsonl")
    parser.add_argument("--out_dir", default="data/tts")
    parser.add_argument("--out_jsonl", default="data/tts/en.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    main(args.prepared_path, args.out_dir, args.out_jsonl)
