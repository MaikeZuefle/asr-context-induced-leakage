"""Loader for the ACL6060 evaluation dataset."""
import os


REFERENCE_FILE = "data/asr/acl6060/ACL.6060.eval.en-xx.en.txt"
AUDIO_DIR = "data/asr/acl6060"


def load_asr():
    with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f if line.strip()]

    audio_paths = []
    for i, _ in enumerate(references, start=1):
        audio_paths.append(os.path.join(AUDIO_DIR, f"sent_{i}.wav"))

    return {"inputs": audio_paths, "references": references}
