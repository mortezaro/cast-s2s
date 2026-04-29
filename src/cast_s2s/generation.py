from __future__ import annotations

from pathlib import Path

import torch
import torchaudio
from peft import PeftModel
from transformers import AutoTokenizer, Gemma3ForCausalLM

from .audio_tokens import codes_to_text, encode_audio, load_wavtokenizer, speech_tokens, text_to_codes


def load_generation_model(
    model_name_or_path: str,
    base_model_name_or_path: str | None,
    speech_token_count: int,
    token: str | None,
    dtype: torch.dtype,
):
    tokenizer_source = base_model_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=token)
    tokenizer.add_tokens(speech_tokens(speech_token_count))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model_name_or_path:
        model = Gemma3ForCausalLM.from_pretrained(base_model_name_or_path, token=token, torch_dtype=dtype)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, model_name_or_path)
    else:
        model = Gemma3ForCausalLM.from_pretrained(model_name_or_path, token=token, torch_dtype=dtype)

    return model.eval(), tokenizer


def audio_prompt(
    audio_path: str,
    wavtokenizer,
    wavtokenizer_repo_path: str,
    device: torch.device,
    sampling_rate: int,
) -> str:
    wav, sr = torchaudio.load(audio_path)
    wav = wav.mean(dim=0)
    if sr != sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, sampling_rate)
    codes = encode_audio(wav, wavtokenizer, wavtokenizer_repo_path, device, input_sr=sampling_rate)
    return "[Speech]" + codes_to_text(codes)


def read_prompt(path: str | None) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int, temperature: float, top_p: float):
    model = model.to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=False)


def load_inference_wavtokenizer(config, device: torch.device):
    return load_wavtokenizer(
        config.wavtokenizer_repo_path,
        config.wavtokenizer_config,
        config.wavtokenizer_checkpoint,
        device,
    )


def generated_speech_codes(text: str) -> list[int]:
    generated = text.split("[Speech]")[-1]
    return text_to_codes(generated)
