import argparse
import logging
import os
from tqdm import tqdm
from transformers import set_seed
import json

# import multimodal models
from models.qwen_omni import generate as generate_qwen_omni
from models.qwen_omni import load_model as load_qwen_omni
from models.phi_multimodal import generate as generate_phi_multimodal
from models.phi_multimodal import load_model as load_phi_multimodal

# import data
from data.fleurs import load_asr

# utils
from utils import set_up_logging

# setting seed for reproducibilty
import random
set_seed(42)
random.seed(42)


def load_model(model_name):
    if model_name == "phi_multimodal":
        model = load_phi_multimodal()
        generate_func = generate_phi_multimodal
    elif model_name == "qwen_omni":
        model = load_qwen_omni()
        generate_func = generate_qwen_omni
    else:
        raise NotImplementedError(f"Model {model_name} currently not supported!")
    return model, generate_func

def load_data(task, language):
    if task == "ASR":
        return load_asr(language)
    else:
        raise NotImplementedError(f"Task {task} currently not supported!")


def load_prompt(task, language):
    if task == "ASR":
        return {
            "standard": {
                "prompt_modality": "text",
                "prompt": "Please transcribe the audio.",
            }
        }
    else:
        raise NotImplementedError(f"Task {task} currently not supported!")



def main(out_folder, model, task, lang):

    # Setting output paths and inferring modalities
    output_file_path = f"{out_folder}/{model}/{task}/{lang}.jsonl"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    set_up_logging(output_file_path)

    # Logging
    logging.info("Welcome!")
    logging.info(
        f"Lang: {lang}, Task: {task}"
    )

    logging.info(f"Output Json: {output_file_path}")

    # Loading Data
    logging.info(f"Loading Data.")
    data = load_data(task=task, language=lang)
    input_data, references = data["inputs"], data["references"]

    # Loading Prompts
    logging.info(f"Loading Prompts.")
    prompt_dict = load_prompt(task=task, language=lang)

    # Loading Model
    logging.info(f"Loading Model.")
    model_instance, generate = load_model(model)

    # Starting Generation
    logging.info(f"Starting Output Generation.")

    # Load existing outputs to skip already processed samples
    existing_outputs = {}
    if os.path.exists(output_file_path):
        logging.info(f"Found existing output file. Loading to skip already processed samples.")
        try:
            with open(output_file_path, "r", encoding="utf-8") as f_in:
                idx = 0
                for line in f_in:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        existing_out = json.loads(line)
                        if "ref" in existing_out:
                            existing_outputs[idx] = existing_out["ref"]
                            idx += 1
                    except json.JSONDecodeError as e:
                        logging.warning(f"Could not parse line: {line[:100]}... Error: {e}")
                        continue
            logging.info(f"Found {len(existing_outputs)} already processed samples. Skipping them.")
        except Exception as e:
            logging.error(f"Error reading existing output file: {e}")
            logging.info("Will proceed without skipping any samples.")

    f_out = open(output_file_path, "a", encoding="utf-8")
    
    skipped_count = 0
    processed_count = 0

    for idx, (x, ref) in enumerate(
            tqdm(zip(input_data, references),
                desc="Generating Outputs",
                total=len(input_data))
        ):

        # Skip if already processed (check by index and verify reference)
        if idx in existing_outputs:
            if existing_outputs[idx] == ref:
                skipped_count += 1
                continue
            else:
                # Reference mismatch - warn and re-process
                logging.warning(f"Index {idx}: Reference mismatch! Expected: {ref[:50]}... Found: {existing_outputs[idx][:50]}...")
                logging.warning(f"Will re-process this sample.")

        out = {"ref": ref, "predicted": {}}
        for prompt_type, prompt in prompt_dict.items():
            out["predicted"][prompt_type] = generate(
                model_instance, prompt, x, modality="audio", output_modality="text", out_wav=None
            )

        f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
        f_out.flush()
        processed_count += 1

    f_out.close()

    logging.info(f"Skipped {skipped_count} already processed samples.")
    logging.info(f"Processed {processed_count} new samples.")
    logging.info(f"Writing Outputs to file {output_file_path}.")
    logging.info("All done.")


if __name__ == "__main__":
    LANGS = ["cs", "de", "en", "es", "fr", "hu", "it", "nl", "pt", "ru", "sv"]
    TASKS = ["ASR"]
    MODELS = ["phi_multimodal", "qwen_omni"]

    parser = argparse.ArgumentParser(description="Process MCIF data.")

    parser.add_argument(
        "--lang", choices=LANGS, default=LANGS[0], help="Language to process"
    )
    parser.add_argument(
        "--task", choices=TASKS, default=TASKS[0], help="Task"
    )
    parser.add_argument("--model", choices=MODELS, default=MODELS[0], help="Model type")
    parser.add_argument(
        "--out_folder", default="generated_output", help="Output data folder path"
    )

    args = parser.parse_args()
    main(
        out_folder=args.out_folder,
        model=args.model,
        task=args.task,
        lang=args.lang,
    )

