#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os

import torch

from cast_s2s.config import load_inference_config
from cast_s2s.generation import (
    audio_prompt,
    generate,
    generated_speech_codes,
    load_generation_model,
    load_inference_wavtokenizer,
    read_prompt,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--prompt")
    parser.add_argument("--output-codes")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_inference_config(args.config)
    token = os.environ.get("HF_TOKEN")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_generation_model(
        cfg.model_name_or_path,
        cfg.base_model_name_or_path,
        cfg.speech_token_count,
        token,
        dtype,
    )

    parts = []
    text_prompt = read_prompt(args.prompt)
    if text_prompt:
        parts.append(text_prompt)
    if args.audio:
        wavtokenizer = load_inference_wavtokenizer(cfg, device)
        parts.append(
            audio_prompt(
                args.audio,
                wavtokenizer,
                cfg.wavtokenizer_repo_path,
                device,
                cfg.audio_sampling_rate,
            )
        )

    prompt = "".join(parts) or "[Speech]"
    text = generate(
        model,
        tokenizer,
        prompt,
        device,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    print(text)

    if args.output_codes:
        codes = generated_speech_codes(text)
        with open(args.output_codes, "w", encoding="utf-8") as handle:
            json.dump({"codes": codes}, handle)


if __name__ == "__main__":
    main()
