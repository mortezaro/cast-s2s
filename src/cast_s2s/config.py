from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TrainConfig:
    model_name_or_path: str
    dataset_path: str
    output_dir: str
    checkpoint_dir: str
    wavtokenizer_repo_path: str
    wavtokenizer_config: str
    wavtokenizer_checkpoint: str
    asr_model_name_or_path: str = "openai/whisper-large-v3"
    hub_repo_id: str | None = None
    audio_sampling_rate: int = 16000
    speech_token_count: int = 4096
    max_length: int = 4096
    train_test_split: float = 0.05
    seed: int = 42
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 1000
    save_total_limit: int = 2
    bf16: bool = True
    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.01
    train_embeddings: bool = False
    train_lm_head: bool = False


@dataclass
class InferenceConfig:
    model_name_or_path: str
    base_model_name_or_path: str | None = None
    codec_repo_id: str | None = "KrauthammerLab/cast-wavtokenizer-24k-40tps"
    wavtokenizer_repo_path: str | None = None
    wavtokenizer_config: str | None = None
    wavtokenizer_checkpoint: str | None = None
    audio_sampling_rate: int = 16000
    codec_sampling_rate: int = 24000
    speech_token_count: int = 4096
    codes_per_second: int = 40
    seconds: float = 3.0
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    keep_prompt_seconds: float = 1.0
    crossfade_ms: int = 60


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def load_train_config(path: str | Path) -> TrainConfig:
    return TrainConfig(**load_yaml(path))


def load_inference_config(path: str | Path) -> InferenceConfig:
    return InferenceConfig(**load_yaml(path))
