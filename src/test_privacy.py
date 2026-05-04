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
PROMPT_WITH_CONTEXT = "Context: {context}\n\nPlease transcribe the audio."


def load_model(model_name, model_path=None):
    if model_name == "phi_multimodal":
        return load_phi_multimodal(model_path=model_path), generate_phi_multimodal
    elif model_name == "qwen_omni":
        return load_qwen_omni(model_path=model_path), generate_qwen_omni
    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")


def make_prompt(text):
    return {"prompt_modality": "text", "prompt": text}


def _format_context(value) -> str:
    """Join a list of sentences into a single context string, or pass through a plain string."""
    if isinstance(value, list):
        return " ".join(value)
    return value


def _get_scenarios(sample: dict) -> list[tuple[str, str | None]]:
    """Return (output_key, context_string_or_None) for every scenario."""
    return [
        ("no_context",           None),
        # 1-word contexts
        ("word_context",         sample["context_word"]),
        ("word_target",          sample["target_word"]),
        ("word_mixed",           f"{sample['context_word']} {sample['target_word']}"),
        # 1-sentence contexts
        ("sentence_context",     sample["context_sentence"]),
        ("sentence_target",      sample["target_context_sentence"]),
        # 2-sentence mixed context (context_sentence + target_context_sentence)
        ("sentences_2_mixed",    _format_context(sample["mixed_sentences"])),
        # 5-sentence contexts
        ("sentences_5_context",  _format_context(sample["context_sentences_5"])),
        ("sentences_5_target",   _format_context(sample["target_context_sentences_5"])),
        ("sentences_5_mixed",    _format_context(sample["mixed_sentences_5"])),
        # 10-sentence contexts
        ("sentences_10_context", _format_context(sample["context_sentences_10"])),
        ("sentences_10_target",  _format_context(sample["target_context_sentences_10"])),
        ("sentences_10_mixed",   _format_context(sample["mixed_sentences_10"])),
    ]


def main(out_folder, model_name, prepared_path, model_path=None):
    model_tag = os.path.basename(model_path) if model_path else model_name
    output_file_path = f"{out_folder}/{model_tag}/privacy/en.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    set_up_logging(output_file_path)

    logging.info(f"Loading prepared data from {prepared_path}")
    with open(prepared_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    logging.info(f"Loaded {len(samples)} samples.")

    logging.info("Loading model.")
    model_instance, generate = load_model(model_name, model_path=model_path)

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
    skipped = processed = updated = 0
    all_scenario_keys = [key for key, _ in _get_scenarios(samples[0])]
    results = dict(existing_outputs)  # idx -> output dict

    for idx, sample in enumerate(tqdm(samples, desc="Generating")):
        existing = results.get(idx)
        scenarios = _get_scenarios(sample)

        if existing is not None:
            missing = [(k, c) for k, c in scenarios if k not in existing["predicted"]]
            if not missing:
                skipped += 1
                continue
            # run only missing scenarios and merge into existing record
            for scenario_key, context in missing:
                prompt_text = PROMPT_NO_CONTEXT if context is None else PROMPT_WITH_CONTEXT.format(context=context)
                existing["predicted"][scenario_key] = generate(
                    model_instance, make_prompt(prompt_text),
                    sample["audio_path"], modality="audio",
                    output_modality="text", out_wav=None,
                )
            results[idx] = existing
            updated += 1
        else:
            predictions = {}
            for scenario_key, context in scenarios:
                prompt_text = PROMPT_NO_CONTEXT if context is None else PROMPT_WITH_CONTEXT.format(context=context)
                predictions[scenario_key] = generate(
                    model_instance, make_prompt(prompt_text),
                    sample["audio_path"], modality="audio",
                    output_modality="text", out_wav=None,
                )
            results[idx] = {
                "reference":    sample["reference"],
                "target_word":  sample["target_word"],
                "context_word": sample["context_word"],
                "predicted":    predictions,
            }
            processed += 1

    # rewrite the full output file to capture any updates
    if updated > 0 or processed > 0:
        with open(output_file_path, "w", encoding="utf-8") as f_out:
            for idx in range(len(samples)):
                if idx in results:
                    f_out.write(json.dumps(results[idx], ensure_ascii=False) + "\n")

    logging.info(f"Skipped {skipped}, processed {processed} new, updated {updated} existing.")
    logging.info(f"Output written to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["phi_multimodal", "qwen_omni"], required=True)
    parser.add_argument("--prepared_path", default="data/prepared/fleurs.jsonl")
    parser.add_argument("--out_folder", default="generated_output")
    parser.add_argument("--model_path", default=None, help="Path to merged model (overrides default HF model)")
    args = parser.parse_args()
    main(args.out_folder, args.model, args.prepared_path, model_path=args.model_path)
