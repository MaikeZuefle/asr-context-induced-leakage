import argparse
import json
import logging
import os
import random

from tqdm import tqdm
from transformers import set_seed

from models.phi_multimodal import generate as generate_phi_multimodal
from models.phi_multimodal import load_model as load_phi_multimodal
from models.qwen_omni import generate as generate_qwen_omni
from models.qwen_omni import load_model as load_qwen_omni
from src.utils import set_up_logging

set_seed(42)
random.seed(42)

PROMPT_NO_CONTEXT = "Please transcribe the audio."
PROMPT_WITH_CONTEXT_WORD = "Context: {context_word}\n\nPlease transcribe the audio."
PROMPT_WITH_CONTEXT_SENTENCE = "Context: {context_sentence}\n\nPlease transcribe the audio."


def load_model(model_name):
    if model_name == "phi_multimodal":
        return load_phi_multimodal(), generate_phi_multimodal
    elif model_name == "qwen_omni":
        return load_qwen_omni(), generate_qwen_omni
    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")


def make_prompt(text):
    return {"prompt_modality": "text", "prompt": text}


def main(out_folder, model_name, prepared_path):
    output_file_path = f"{out_folder}/{model_name}/privacy/en.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    set_up_logging(output_file_path)

    logging.info(f"Loading prepared data from {prepared_path}")
    with open(prepared_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Loaded {len(samples)} samples.")

    logging.info("Loading model.")
    model_instance, generate = load_model(model_name)

    # Load existing outputs to skip already processed samples
    existing_outputs = {}
    if os.path.exists(output_file_path):
        logging.info("Found existing output file. Loading to skip already processed samples.")
        with open(output_file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    existing_outputs[idx] = json.loads(line)
                except json.JSONDecodeError:
                    continue
        logging.info(f"Found {len(existing_outputs)} already processed samples.")

    logging.info("Starting generation.")
    skipped = processed = 0

    with open(output_file_path, "a", encoding="utf-8") as f_out:
        for idx, sample in enumerate(tqdm(samples, desc="Generating")):
            if idx in existing_outputs:
                skipped += 1
                continue

            no_context_pred = generate(
                model_instance,
                make_prompt(PROMPT_NO_CONTEXT),
                sample["audio_path"],
                modality="audio",
                output_modality="text",
                out_wav=None,
            )
            with_context_word_pred = generate(
                model_instance,
                make_prompt(PROMPT_WITH_CONTEXT_WORD.format(context_word=sample["context_word"])),
                sample["audio_path"],
                modality="audio",
                output_modality="text",
                out_wav=None,
            )
            with_context_sentence_pred = generate(
                model_instance,
                make_prompt(PROMPT_WITH_CONTEXT_SENTENCE.format(context_sentence=sample["context_sentence"])),
                sample["audio_path"],
                modality="audio",
                output_modality="text",
                out_wav=None,
            )

            out = {
                "reference":        sample["reference"],
                "target_word":      sample["target_word"],
                "context_word":     sample["context_word"],
                "context_sentence": sample["context_sentence"],
                "predicted": {
                    "no_context":            no_context_pred,
                    "with_context_word":     with_context_word_pred,
                    "with_context_sentence": with_context_sentence_pred,
                },
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()
            processed += 1

    logging.info(f"Skipped {skipped}, processed {processed} samples.")
    logging.info(f"Output written to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["phi_multimodal", "qwen_omni"], required=True)
    parser.add_argument("--prepared_path", default="data/prepared/en.jsonl")
    parser.add_argument("--out_folder", default="generated_output")
    args = parser.parse_args()
    main(args.out_folder, args.model, args.prepared_path)
