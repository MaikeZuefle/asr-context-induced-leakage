from datasets import load_dataset
import soundfile as sf
import os
from data.utils import FLEURS_LANG_MAP

def load_asr(language):

    base_dir = "data/asr"
    os.makedirs(base_dir, exist_ok=True)

    fleurs_asr = load_dataset("google/fleurs", FLEURS_LANG_MAP[language], split="test", trust_remote_code=True)

    audio_paths = [];  references = []

    for idx, entry in enumerate(fleurs_asr):
        fleurs_idx = entry["id"]
        wav_path = os.path.join(
            base_dir, f"fleurs_{language}_{fleurs_idx}.wav"
        )

        if not os.path.exists(wav_path):
            audio_array = entry["audio"]["array"]
            sr = entry["audio"]["sampling_rate"]
            sf.write(wav_path, audio_array, sr)

        audio_paths.append(wav_path)
        references.append(entry["transcription"])

    return {"inputs" : audio_paths, "references": references}