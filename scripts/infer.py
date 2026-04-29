#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch

from cast_s2s.config import load_inference_config
from cast_s2s.generation import (
    build_speech_generation,
    load_generation_model,
    load_inference_wavtokenizer,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output-dir", default="generated")
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

    wavtokenizer = load_inference_wavtokenizer(cfg, device, token=token)
    result = build_speech_generation(
        model,
        tokenizer,
        wavtokenizer,
        args.audio,
        cfg,
        device,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        output_dir / "recon_24k.wav",
        result.recon_audio.squeeze(0).float().cpu().numpy(),
        cfg.codec_sampling_rate,
    )
    sf.write(
        output_dir / "continuation.wav",
        result.continuation_audio.squeeze(0).float().cpu().numpy(),
        cfg.codec_sampling_rate,
    )
    sf.write(
        output_dir / "stitched_24k.wav",
        result.stitched_audio.squeeze(0).float().cpu().numpy(),
        cfg.codec_sampling_rate,
    )

    print(result.text)
    print(f"[INFO] Wrote {output_dir / 'recon_24k.wav'}")
    print(f"[INFO] Wrote {output_dir / 'continuation.wav'}")
    print(f"[INFO] Wrote {output_dir / 'stitched_24k.wav'}")

    output_codes = Path(args.output_codes) if args.output_codes else output_dir / "codes.json"
    output_codes.parent.mkdir(parents=True, exist_ok=True)
    with output_codes.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "prompt_codes": result.prompt_codes,
                "generated_codes": result.generated_codes,
            },
            handle,
        )


if __name__ == "__main__":
    main()
