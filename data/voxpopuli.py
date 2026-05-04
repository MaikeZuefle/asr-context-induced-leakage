from datasets import load_dataset
import soundfile as sf
import os


def load_asr(splits=("test",)):
    base_dir = "data/asr/voxpopuli"
    os.makedirs(base_dir, exist_ok=True)

    audio_paths = []
    references = []

    for split in splits:
        dataset = load_dataset("facebook/voxpopuli", "en", split=split, trust_remote_code=True, streaming=True)

        for entry in dataset:
            audio_id = entry["audio_id"]
            wav_path = os.path.join(base_dir, f"{audio_id.replace(':', '-')}.wav")

            if not os.path.exists(wav_path):
                sf.write(wav_path, entry["audio"]["array"], entry["audio"]["sampling_rate"])

            audio_paths.append(wav_path)
            references.append(entry["normalized_text"])

    return {"inputs": audio_paths, "references": references}
