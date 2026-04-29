from __future__ import annotations

import os
from typing import Any

import torch
import transformers as ts
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Gemma3ForCausalLM

from .audio_tokens import speech_tokens


def local_device_map() -> str | dict[str, int]:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return "auto"
    return {"": int(local_rank)}


def load_tokenizer(model_name_or_path: str, speech_token_count: int, token: str | None):
    tokenizer = ts.AutoTokenizer.from_pretrained(model_name_or_path, token=token)
    tokenizer.add_tokens(speech_tokens(speech_token_count))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_asr_pipeline(model_name_or_path: str, device_map: Any, dtype: torch.dtype, token: str | None):
    model = ts.AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device_map,
        token=token,
    )
    processor = ts.AutoProcessor.from_pretrained(model_name_or_path, token=token)
    return ts.pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
    )


def load_causal_model(
    model_name_or_path: str,
    vocab_size: int,
    device_map: Any,
    dtype: torch.dtype,
    token: str | None,
):
    model = Gemma3ForCausalLM.from_pretrained(
        model_name_or_path,
        token=token,
        device_map=device_map,
        torch_dtype=dtype,
    )
    model.resize_token_embeddings(_pad_to_multiple(vocab_size, 8))
    return model


def apply_lora(
    model,
    r: int,
    alpha: int,
    dropout: float,
    train_embeddings: bool = False,
    train_lm_head: bool = False,
):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, config)

    for name, param in model.named_parameters():
        if train_lm_head and "lm_head" in name:
            param.requires_grad = True
        if train_embeddings and "embed" in name:
            param.requires_grad = True

    return model


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            print(name)
    print(f"Total parameters: {total / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable / 1e6:.2f}M")


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
