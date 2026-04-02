def load_model():
    from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
    model_path = "microsoft/Phi-4-multimodal-instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True,
        _attn_implementation='flash_attention_2',
    ).cuda()
    generation_config = GenerationConfig.from_pretrained(model_path)
    return model, processor, generation_config

def generate(model_processor_config, prompt, input_data, modality, output_modality, out_wav=None):
    import soundfile as sf
    import torch

    if output_modality == "audio":
        raise NotImplementedError("Phi-4-multimodal-instruct does not support speech in output.")

    # get (prompt-)modalities and model
    prompt_modality = prompt["prompt_modality"]
    orig_prompt = prompt["prompt"]
    model, processor,  generation_config = model_processor_config

    # Handle question answering tasks
    if isinstance(input_data, dict):
        example = input_data["audio_path"]
        is_q_task = True
    else:
        example = input_data
        is_q_task = False

    # prompts
    user_prompt = "<|user|>"
    assistant_prompt = "<|assistant|>"
    prompt_suffix = "<|end|>"
    audios = []
    seperator_token = ""

    # prepare prompts
    if prompt_modality == "audio":
        prompt_audio, prompt_samplerate = sf.read(orig_prompt)
        audios.append((prompt_audio, prompt_samplerate))
        seperator_token += f"<|audio_{len(audios)}|>"
        if is_q_task:
            speech_q = input_data["question_speech"]
            speech_q_audio, speech_q_samplerate = sf.read(speech_q)
            audios.append((speech_q_audio, speech_q_samplerate))
            seperator_token += f"<|audio_{len(audios)}|>"
        prompt = ""
    elif prompt_modality == "text":
        prompt = orig_prompt
        if is_q_task:
            text_q = input_data["question_text"]
            prompt += " " + text_q

    # prepare inputs
    if modality == "audio":
        audio, samplerate = sf.read(example)
        audios.append((audio, samplerate))
        seperator_token += f"<|audio_{len(audios)}|>"
    elif modality == "text":
        seperator_token += f"\n"
        if prompt_modality == "text":
            audios = None

    final_prompt = f"{user_prompt}{example}{seperator_token}{prompt}{prompt_suffix}{assistant_prompt}"
    inputs = processor(text=final_prompt, audios=audios, return_tensors='pt').to('cuda:0')
    generate_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Clear CUDA cache before returning
    torch.cuda.empty_cache()

    return response