from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torchaudio
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

from .audio_tokens import (
    codes_to_text,
    decode_codes,
    load_wavtokenizer,
    normalize_codes,
    resolve_codec_files,
    speech_tokens,
)


@dataclass
class GenerationResult:
    text: str
    prompt_codes: list[int]
    generated_codes: list[int]
    recon_audio: torch.Tensor | None = None
    continuation_audio: torch.Tensor | None = None
    stitched_audio: torch.Tensor | None = None


class SpeechOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_ids: Iterable[int]):
        super().__init__()
        self.allowed_ids = list(allowed_ids)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float("-inf"))
        mask[..., self.allowed_ids] = 0.0
        return scores + mask


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
        model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, token=token, torch_dtype=dtype)
        model.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(model, model_name_or_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=token, torch_dtype=dtype)

    return model.eval(), tokenizer


def encode_prompt_audio(
    audio_path: str,
    wavtokenizer,
    wavtokenizer_repo_path: str | None,
    device: torch.device,
    input_sampling_rate: int,
    codec_sampling_rate: int,
) -> tuple[list[int], torch.Tensor]:
    wav, sr = torchaudio.load(audio_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != input_sampling_rate:
        wav = torchaudio.functional.resample(wav, sr, input_sampling_rate)

    wav24 = torchaudio.functional.resample(wav, input_sampling_rate, codec_sampling_rate).to(device)
    bandwidth_id = torch.tensor([0], device=device)
    with torch.no_grad():
        feats, codes = wavtokenizer.encode_infer(wav24, bandwidth_id=bandwidth_id)
        recon = wavtokenizer.decode(feats, bandwidth_id=bandwidth_id)
    if recon.dim() == 3:
        recon = recon.squeeze(0)
    return normalize_codes(codes), recon


def speech_token_table(tokenizer, codebook_size: int) -> tuple[list[int], dict[int, int]]:
    token_ids = []
    for index in range(1, codebook_size + 1):
        ids = tokenizer(f"[Sp{index}]", add_special_tokens=False)["input_ids"]
        if len(ids) != 1:
            raise RuntimeError(f"[Sp{index}] is not a single tokenizer token.")
        token_ids.append(ids[0])
    return token_ids, {token_id: index for index, token_id in enumerate(token_ids)}


def generate_speech_continuation(
    model,
    tokenizer,
    prompt_codes: list[int],
    device: torch.device,
    codebook_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
) -> tuple[str, list[int]]:
    model = model.to(device)
    prefix = "[Speech]" + codes_to_text(prompt_codes)
    inputs = tokenizer(prefix, return_tensors="pt").to(device)

    speech_token_ids, id_to_code = speech_token_table(tokenizer, codebook_size)
    eos_id = tokenizer.eos_token_id
    allowed_ids = set(speech_token_ids)
    if eos_id is not None:
        allowed_ids.add(eos_id)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max(1, max_new_tokens),
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            logits_processor=LogitsProcessorList([SpeechOnlyLogitsProcessor(allowed_ids)]),
        )

    tail = output[0][inputs["input_ids"].size(1) :].tolist()
    if eos_id is not None and eos_id in tail:
        tail = tail[: tail.index(eos_id)]
    codes = [id_to_code[token_id] for token_id in tail if token_id in id_to_code]
    return tokenizer.decode(output[0], skip_special_tokens=False), codes


def equal_power_crossfade(
    previous: torch.Tensor,
    continuation: torch.Tensor,
    fade_ms: int,
    sampling_rate: int,
    device: torch.device,
) -> torch.Tensor:
    previous = previous.to(device)
    continuation = continuation.to(device)
    fade = max(1, int(sampling_rate * fade_ms / 1000))
    if previous.size(1) < fade or continuation.size(1) < fade:
        return torch.cat([previous, continuation], dim=1)

    t = torch.linspace(0, 1, fade, device=device).view(1, -1)
    a = previous[:, -fade:]
    b = continuation[:, :fade]
    mix = torch.cos(t * 0.5 * math.pi) * a + torch.sin(t * 0.5 * math.pi) * b
    return torch.cat([previous[:, :-fade], mix, continuation[:, fade:]], dim=1)


def load_inference_wavtokenizer(config, device: torch.device, token: str | None = None):
    config_path, checkpoint_path = resolve_codec_files(
        config.codec_repo_id,
        config.wavtokenizer_config,
        config.wavtokenizer_checkpoint,
        token=token,
    )
    return load_wavtokenizer(
        config.wavtokenizer_repo_path,
        config_path,
        checkpoint_path,
        device,
    )


def build_speech_generation(
    model,
    tokenizer,
    wavtokenizer,
    audio_path: str,
    config,
    device: torch.device,
) -> GenerationResult:
    prompt_codes, recon = encode_prompt_audio(
        audio_path,
        wavtokenizer,
        config.wavtokenizer_repo_path,
        device,
        config.audio_sampling_rate,
        config.codec_sampling_rate,
    )
    max_new_tokens = int(round(config.seconds * config.codes_per_second))
    text, generated_codes = generate_speech_continuation(
        model,
        tokenizer,
        prompt_codes,
        device,
        codebook_size=config.speech_token_count,
        max_new_tokens=max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
    )

    keep = max(0, int(round(config.keep_prompt_seconds * config.codes_per_second)))
    decode_codes_input = prompt_codes[-keep:] + generated_codes if keep else generated_codes
    continuation = decode_codes(decode_codes_input, wavtokenizer, device)
    stitched = equal_power_crossfade(
        recon,
        continuation,
        fade_ms=config.crossfade_ms,
        sampling_rate=config.codec_sampling_rate,
        device=device,
    )
    return GenerationResult(
        text=text,
        prompt_codes=prompt_codes,
        generated_codes=generated_codes,
        recon_audio=recon,
        continuation_audio=continuation,
        stitched_audio=stitched,
    )
