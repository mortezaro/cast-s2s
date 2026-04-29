from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import torch


def add_wavtokenizer_to_path(repo_path: str) -> None:
    path = str(Path(repo_path).expanduser().resolve())
    if path not in sys.path:
        sys.path.append(path)


def load_wavtokenizer(repo_path: str, config_path: str, checkpoint_path: str, device: torch.device):
    add_wavtokenizer_to_path(repo_path)
    module = importlib.import_module("decoder.pretrained")
    wavtokenizer_cls = getattr(module, "WavTokenizer")
    wavtokenizer = wavtokenizer_cls.from_pretrained0802(config_path, checkpoint_path)
    return wavtokenizer.to(device).eval()


def convert_audio_for_wavtokenizer(
    repo_path: str,
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
    repo_path: str,
    device: torch.device,
    input_sr: int = 16000,
    target_sr: int = 24000,
) -> list[int]:
    wav = torch.as_tensor(audio).squeeze()
    wav = convert_audio_for_wavtokenizer(repo_path, wav, input_sr, target_sr, device)
    bandwidth_id = torch.tensor([0], device=device)
    with torch.no_grad():
        _, discrete = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    return discrete.squeeze().detach().cpu().numpy().astype(int).tolist()


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
