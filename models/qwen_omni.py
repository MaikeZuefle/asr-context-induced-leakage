import os
import torch

DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

def load_model(model_path=None):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    model_path = model_path or DEFAULT_MODEL_PATH
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # Always load processor from base model — fine-tuned checkpoints only save the tokenizer
    processor_path = DEFAULT_MODEL_PATH if model_path != DEFAULT_MODEL_PATH else model_path
    processor = Qwen2_5OmniProcessor.from_pretrained(processor_path)
    return model, processor


def generate(model_processor, prompt, input_data, modality, output_modality, out_wav):
    import torch
    from qwen_omni_utils import process_mm_info
    import soundfile as sf

    # get (prompt-)modalities and model
    prompt_modality = prompt["prompt_modality"]
    orig_prompt = prompt["prompt"]
    model, processor = model_processor

    # Handle question answering tasks
    if isinstance(input_data, dict):
        example = input_data["audio_path"]
        is_q_task = True
    else:
        example = input_data
        is_q_task = False

    # prepare prompts
    if prompt_modality == "audio":
        prompt_dict = [{"type": "audio", "audio": orig_prompt}]
        if is_q_task:
            speech_q = input_data["question_speech"]
            prompt_dict.append({"type": "audio", "audio": speech_q})
    elif prompt_modality == "text":
        text_prompt = orig_prompt
        if is_q_task:
            text_q = input_data["question_text"]
            text_prompt += " " + text_q
        prompt_dict = [{"type": "text", "text": text_prompt}]

    # prepare inputs
    if modality == "audio":
        input_dict = [{"type": "audio", "audio": example}]
    elif modality == "text":
        input_dict = [{"type": "text", "text": example}]

    USE_AUDIO_IN_VIDEO = False
    RETURN_AUDIO = output_modality == "audio"


    user_conv_content = input_dict + prompt_dict

    if RETURN_AUDIO:
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
        user_conv_content.append({"type": "text", "text": "Only return the answer requested. Do not include any explanation or introductions."})
    else:
        system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech. Only return the answer requested. Do not include any explanation or introductions."

    system_conv = {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt}
        ],
    }
    
    user_conv = {"role": "user", "content": user_conv_content}

    conversation = [system_conv, user_conv]

    # Preparation for inference
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
    inputs = inputs.to(model.device).to(model.dtype)

    # Inference: Generation of the output text and audio
    if RETURN_AUDIO:
        _, audio  = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=RETURN_AUDIO)
        response = out_wav
        sf.write(
            out_wav,
            audio.reshape(-1).detach().cpu().numpy(),
            samplerate=24000,
        )
        
    else:
        text_ids  = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=RETURN_AUDIO)
        audio = None
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        # postprocess
        response = text[-1].split("\nassistant")[-1].strip()


    # Clear CUDA cache before returning
    torch.cuda.empty_cache()

    return response