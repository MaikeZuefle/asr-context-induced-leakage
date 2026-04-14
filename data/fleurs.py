from datasets import load_dataset
import soundfile as sf
import os
from data.utils import FLEURS_LANG_MAP


def load_asr(language, splits=("test", "validation")):
    base_dir = "data/asr"
    os.makedirs(base_dir, exist_ok=True)

    audio_paths = []
    references = []

    for split in splits:
        fleurs_asr = load_dataset("google/fleurs", FLEURS_LANG_MAP[language], split=split, trust_remote_code=True)

        for entry in fleurs_asr:
            fleurs_idx = entry["id"]
            wav_path = os.path.join(base_dir, f"fleurs_{language}_{split}_{fleurs_idx}.wav")

            if not os.path.exists(wav_path):
                sf.write(wav_path, entry["audio"]["array"], entry["audio"]["sampling_rate"])

            audio_paths.append(wav_path)
            references.append(entry["transcription"])

    return {"inputs": audio_paths, "references": references}