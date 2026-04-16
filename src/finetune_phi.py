"""
Fine-tune Phi-4-multimodal-instruct on TTS-generated privacy evaluation sentences
or FLEURS context FT data.

Uses the model's built-in speech LoRA adapter (model.set_lora_adapter('speech')).

Requirements:
    peft>=0.13.2
    transformers>=4.46.1
    accelerate>=1.3.0
    scipy, soundfile
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import soundfile as sf
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
    set_seed,
)

set_seed(42)

MODEL_PATH = "microsoft/Phi-4-multimodal-instruct"
INSTRUCTION = "Please transcribe the audio."
INSTRUCTION_WITH_CONTEXT = "Context: {context}\n\nPlease transcribe the audio."
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Stopping criteria (unchanged from original)
# ---------------------------------------------------------------------------

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens:].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]
        return torch.all(self.stop_tokens_idx)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TTSDataset(Dataset):
    """Loads TTS-generated audio samples from our privacy evaluation JSONL.

    Each entry has:
        audio_path, transcript, source (context_sentence / target_context_sentence), ...

    The `sources` argument controls which entries to include.
    """

    def __init__(self, processor, tts_jsonl: str, sources: list[str]):
        with open(tts_jsonl, "r", encoding="utf-8") as f:
            all_samples = [json.loads(line) for line in f if line.strip()]
        self.samples = [s for s in all_samples if s["source"] in sources]
        self.processor = processor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample["audio_path"])

        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{INSTRUCTION}",
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors="pt")

        answer = f"{sample['transcript']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1]:] = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


# ---------------------------------------------------------------------------
# FLEURS context dataset
# ---------------------------------------------------------------------------

_FLEURS_CONTEXT_FIELDS = {
    "fleurs_context_1":  "context_sentence",
    "fleurs_context_5":  "context_sentences_5",
    "fleurs_context_10": "context_sentences_10",
}
_FLEURS_MIXED_FIELDS = ["context_sentence", "context_sentences_5", "context_sentences_10"]


class FleursContextDataset(Dataset):
    """Loads FLEURS context FT data and injects context into the prompt,
    matching the inference-time format: 'Context: {context}\\n\\nPlease transcribe the audio.'
    """

    def __init__(self, processor, ft_jsonl: str, context_field: str | None = None, seed: int = 42):
        with open(ft_jsonl, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]
        self.processor = processor
        self.context_field = context_field  # None means mixed (random per sample)
        if context_field is None:
            rng = random.Random(seed)
            self.context_fields = [rng.choice(_FLEURS_MIXED_FIELDS) for _ in self.samples]

    def __len__(self):
        return len(self.samples)

    def _get_context(self, idx: int) -> str:
        field = self.context_fields[idx] if self.context_field is None else self.context_field
        value = self.samples[idx][field]
        return " ".join(value) if isinstance(value, list) else value

    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio, sr = sf.read(sample["audio_path"])
        context = self._get_context(idx)
        instruction = INSTRUCTION_WITH_CONTEXT.format(context=context)

        user_message = {
            "role": "user",
            "content": f"<|audio_1|>\n{instruction}",
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=prompt, audios=[(audio, sr)], return_tensors="pt")

        answer = f"{sample['transcript']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors="pt").input_ids

        input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        labels[:, -answer_ids.shape[1]:] = answer_ids

        return {
            "input_ids": input_ids,
            "labels": labels,
            "input_audio_embeds": inputs.input_audio_embeds,
            "audio_embed_sizes": inputs.audio_embed_sizes,
        }


# ---------------------------------------------------------------------------
# Collation (unchanged from original)
# ---------------------------------------------------------------------------

def pad_sequence(sequences, padding_side="right", padding_value=0):
    assert padding_side in ["right", "left"]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == "right":
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    ndim = tensors[0].dim()
    assert all(t.dim() == ndim for t in tensors[1:])
    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    index = 0
    for t in tensors:
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        slices[dim] = slice(index, index + t.shape[dim])
        output[slices] = t
        index += t.shape[dim]
    return output


def collate_fn(batch):
    input_ids_list = [b["input_ids"][0] for b in batch]
    labels_list = [b["labels"][0] for b in batch]
    input_audio_embeds_list = [b["input_audio_embeds"] for b in batch]
    audio_embed_sizes_list = [b["audio_embed_sizes"] for b in batch]
    audio_attention_mask_list = [
        b["input_audio_embeds"].new_full((b["input_audio_embeds"].size(1),), True, dtype=torch.bool)
        for b in batch
    ]

    input_ids = pad_sequence(input_ids_list, padding_side="left", padding_value=0)
    labels = pad_sequence(labels_list, padding_side="left", padding_value=_IGNORE_INDEX)
    audio_attention_mask = (
        pad_sequence(audio_attention_mask_list, padding_side="right", padding_value=False)
        if len(audio_attention_mask_list) > 1
        else None
    )
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature({
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "input_audio_embeds": input_audio_embeds,
        "audio_embed_sizes": audio_embed_sizes,
        "audio_attention_mask": audio_attention_mask,
        "input_mode": 2,  # speech mode
    })


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def create_model(use_flash_attention: bool):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation="flash_attention_2" if use_flash_attention else "sdpa",
        trust_remote_code=True,
    ).to("cuda")
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, processor, dataset, save_path=None, batch_size=1):
    from jiwer import wer
    from transformers import Compose, ToLowerCase, RemovePunctuation, RemoveMultipleSpaces, Strip

    normalizer = Compose([ToLowerCase(), RemovePunctuation(), RemoveMultipleSpaces(), Strip()])

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    model.eval()
    all_hyps, all_refs = [], []

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn,
        shuffle=False, drop_last=False, num_workers=4,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_token_ids = processor.tokenizer(
        stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt"
    )["input_ids"].to(f"cuda:{local_rank}")

    for inputs in tqdm(dataloader, desc="Evaluating", disable=rank != 0):
        stopping_criteria = StoppingCriteriaList([
            MultipleTokenBatchStoppingCriteria(stop_token_ids, batch_size=inputs.input_ids.size(0))
        ])
        inputs = inputs.to(f"cuda:{local_rank}")
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id,
            max_new_tokens=64, stopping_criteria=stopping_criteria,
        )
        stop_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]
        stop_idx = torch.where(stop_idx > 0, stop_idx - stop_token_ids.shape[-1], generated_ids.shape[-1])

        hyps = [
            processor.decode(generated_ids[i, inputs["input_ids"].shape[1]:stop_idx[i]],
                             skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for i in range(len(generated_ids))
        ]
        refs = [
            processor.decode(inputs["labels"][i][inputs["labels"][i] != _IGNORE_INDEX],
                             skip_special_tokens=True).rstrip(ANSWER_SUFFIX)
            for i in range(len(inputs["labels"]))
        ]
        all_hyps.extend(hyps)
        all_refs.extend(refs)

    all_hyps = gather_object(all_hyps)
    all_refs = gather_object(all_refs)

    if rank == 0:
        norm_hyps = [normalizer(h) for h in all_hyps]
        norm_refs = [normalizer(r) for r in all_refs]
        wer_score = round(wer(norm_refs, norm_hyps) * 100, 2)
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                for ref, hyp in zip(all_refs, all_hyps):
                    print(json.dumps({"ref": ref, "hyp": hyp}, ensure_ascii=False), file=f)
        return wer_score
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DATASET_SOURCES = {
    "context_word": ["context_sentence"],
    "target_word":  ["target_context_sentence"],
    "both":         ["context_sentence", "target_context_sentence"],
}

COMBINED_DATASETS = {
    "context_word_fleurs_mixed": ("context_word", "fleurs_context_mixed"),
    "target_word_fleurs_mixed":  ("target_word",  "fleurs_context_mixed"),
    "both_fleurs_mixed":         ("both",          "fleurs_context_mixed"),
}

_EVAL_DATASETS      = list(DATASET_SOURCES.keys())
_FLEURS_DATASETS    = list(_FLEURS_CONTEXT_FIELDS.keys()) + ["fleurs_context_mixed"]
_COMBINED_DATASETS  = list(COMBINED_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",
                        choices=_EVAL_DATASETS + _FLEURS_DATASETS + _COMBINED_DATASETS,
                        required=True)
    parser.add_argument("--tts_jsonl", default="data/tts/en.jsonl")
    parser.add_argument("--ft_jsonl",  default="data/ft/fleurs_context/en.jsonl")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=4.0e-5)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"saves/phi4-multimodal/lora/{args.dataset}"

    accelerator = Accelerator()
    num_gpus = accelerator.num_processes
    assert args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0, \
        "batch_size must be divisible by num_gpus * batch_size_per_gpu"
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model = create_model(args.use_flash_attention)

    model.set_lora_adapter("speech")

    if args.dataset in _EVAL_DATASETS:
        sources = DATASET_SOURCES[args.dataset]
        train_dataset = TTSDataset(processor, args.tts_jsonl, sources=sources)
        print(f"Training on {len(train_dataset)} samples (sources: {sources})")
    elif args.dataset in _COMBINED_DATASETS:
        eval_key, fleurs_key = COMBINED_DATASETS[args.dataset]
        sources = DATASET_SOURCES[eval_key]
        tts_dataset = TTSDataset(processor, args.tts_jsonl, sources=sources)
        context_field = _FLEURS_CONTEXT_FIELDS.get(fleurs_key)  # None for mixed
        fleurs_dataset = FleursContextDataset(processor, args.ft_jsonl, context_field=context_field)
        train_dataset = ConcatDataset([tts_dataset, fleurs_dataset])
        print(f"Training on {len(tts_dataset)} TTS + {len(fleurs_dataset)} FLEURS context samples")
    else:
        context_field = _FLEURS_CONTEXT_FIELDS.get(args.dataset)  # None for mixed
        train_dataset = FleursContextDataset(processor, args.ft_jsonl, context_field=context_field)
        print(f"Training on {len(train_dataset)} FLEURS context samples (dataset: {args.dataset})")

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy="no",
        save_only_model=True,
        bf16=args.use_flash_attention,
        fp16=not args.use_flash_attention,
        remove_unused_columns=False,
        report_to="wandb",
        dataloader_num_workers=1,
        ddp_find_unused_parameters=True,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_dataset,
    )
    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(args.output_dir)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
