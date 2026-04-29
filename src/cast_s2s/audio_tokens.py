from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


def add_wavtokenizer_to_path(repo_path: str | None) -> None:
    if not repo_path:
        return
    path = str(Path(repo_path).expanduser().resolve())
    if path not in sys.path:
        sys.path.append(path)


def resolve_codec_files(
    codec_repo_id: str | None,
    config_path: str | None,
    checkpoint_path: str | None,
    token: str | None = None,
) -> tuple[str | None, str]:
    if checkpoint_path:
        return config_path, checkpoint_path
    if not codec_repo_id:
        raise ValueError("Set wavtokenizer_checkpoint or codec_repo_id.")

    checkpoint_path = hf_hub_download(
        codec_repo_id,
        filename="wavtokenizer_large_unify_600_24k.ckpt",
        token=token,
    )
    if config_path:
        return config_path, checkpoint_path

    try:
        config_path = hf_hub_download(codec_repo_id, filename="config.yaml", token=token)
    except Exception:
        config_path = None
    return config_path, checkpoint_path


def load_wavtokenizer(
    repo_path: str | None,
    config_path: str | None,
    checkpoint_path: str,
    device: torch.device,
):
    add_wavtokenizer_to_path(repo_path)
    module = importlib.import_module("decoder.pretrained")
    wavtokenizer_cls = getattr(module, "WavTokenizer")
    wavtokenizer = wavtokenizer_cls.from_pretrained0802(config_path, checkpoint_path)
    return wavtokenizer.to(device).eval()


def convert_audio_for_wavtokenizer(
    repo_path: str | None,
    wav: torch.Tensor,
    input_sr: int,
    target_sr: int,
    device: torch.device,
) -> torch.Tensor:
    add_wavtokenizer_to_path(repo_path)
    convert_audio = importlib.import_module("encoder.utils").convert_audio
    wav = wav.view(1, -1).cpu().float()
    return convert_audio(wav, input_sr, target_sr, 1).to(device)


def speech_tokens(count: int = 4096) -> list[str]:
    return [f"[Sp{index + 1}]" for index in range(count)]


def encode_audio(
    audio: np.ndarray | torch.Tensor,
    wavtokenizer,
    repo_path: str | None,
    device: torch.device,
    input_sr: int = 16000,
    target_sr: int = 24000,
) -> list[int]:
    wav = torch.as_tensor(audio).squeeze()
    wav = convert_audio_for_wavtokenizer(repo_path, wav, input_sr, target_sr, device)
    bandwidth_id = torch.tensor([0], device=device)
    with torch.no_grad():
        _, discrete = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return normalize_codes(discrete)


def normalize_codes(codes: torch.Tensor) -> list[int]:
    if codes.dim() == 3:
        codes = codes.squeeze(0)
        codes = codes[0] if codes.size(0) > 1 else codes.squeeze(0)
    elif codes.dim() == 2:
        codes = codes.squeeze(0)
    return codes.long().detach().cpu().tolist()


def decode_codes(
    codes: list[int],
    wavtokenizer,
    device: torch.device,
) -> torch.Tensor:
    token_tensor = torch.tensor(codes, dtype=torch.long, device=device).view(1, 1, -1)
    bandwidth_id = torch.tensor([0], device=device)
    with torch.no_grad():
        audio = wavtokenizer.decode(
            wavtokenizer.codes_to_features(token_tensor),
            bandwidth_id=bandwidth_id,
        )
    if audio.dim() == 3:
        audio = audio.squeeze(0)
    return audio


def codes_to_text(codes: list[int]) -> str:
    return "".join(f"[Sp{code + 1}]" for code in codes)


def text_to_codes(text: str) -> list[int]:
    codes = []
    for piece in text.replace("[Sp", " [Sp").split():
        if piece.startswith("[Sp") and piece.endswith("]"):
            value = piece[3:-1]
            if value.isdigit():
                codes.append(int(value) - 1)
    return codes
