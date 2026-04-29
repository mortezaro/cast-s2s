from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

import datasets as ds
import numpy as np
import torch

from .codec import codes_to_text, encode_audio


def load_audio_dataset(path: str, sampling_rate: int, split_ratio: float, seed: int):
    ds.disable_caching()
    dataset = ds.load_from_disk(path)

    def flag_valid(example):
        audio = example.get("audio")
        path = audio.get("path") if isinstance(audio, dict) else None
        return {"keep": bool(path is None or os.path.exists(path))}

    dataset = dataset.map(flag_valid, num_proc=min(os.cpu_count() or 1, 8), load_from_cache_file=False)
    dataset = dataset.filter(lambda row: row["keep"])
    dataset = dataset.remove_columns(["keep"])
    dataset = dataset.cast_column("audio", ds.Audio(sampling_rate=sampling_rate, decode=True))
    dataset = dataset.shuffle(seed=seed)

    if split_ratio <= 0:
        return dataset, None
    split = dataset.train_test_split(test_size=split_ratio, seed=seed)
    return split["train"], split["test"]


def random_text_segment(
    audio_length: int,
    min_text_portion: float = 0.35,
    max_text_portion: float = 0.55,
    position_probs: tuple[float, float, float] = (0.2, 0.6, 0.2),
) -> tuple[int, int]:
    lower = max(1, int(min_text_portion * audio_length))
    upper = max(lower, int(max_text_portion * audio_length))
    seg_len = random.randint(lower, upper)
    pos = random.choices([0, 1, 2], weights=position_probs, k=1)[0]

    if pos == 0:
        return 0, seg_len
    if pos == 2:
        return audio_length - seg_len, audio_length

    start = random.randint(0, audio_length - seg_len)
    return start, start + seg_len


def split_audio(
    audio_array: np.ndarray | torch.Tensor,
    min_text_portion: float = 0.35,
    max_text_portion: float = 0.55,
    position_probs: tuple[float, float, float] = (0.2, 0.6, 0.2),
) -> list[np.ndarray]:
    if isinstance(audio_array, torch.Tensor):
        audio_array = audio_array.squeeze().cpu().numpy()

    audio_array = np.asarray(audio_array).squeeze()
    if audio_array.size < 2:
        return [audio_array]

    start, end = random_text_segment(
        len(audio_array),
        min_text_portion=min_text_portion,
        max_text_portion=max_text_portion,
        position_probs=position_probs,
    )
    return [audio_array[:start], audio_array[start:end], audio_array[end:]]


@dataclass
class InterleavedCollator:
    tokenizer: Any
    wavtokenizer: Any
    asr_pipeline: Any
    wavtokenizer_repo_path: str
    device: torch.device
    sampling_rate: int = 16000
    max_length: int = 4096
    min_speech_samples: int = 16000

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded_strings = self.encode_batch(batch)
        tokenized = self.tokenizer(
            encoded_strings,
            truncation=True,
            max_length=self.max_length,
            padding="longest",
            return_tensors="pt",
        )
        labels = tokenized["input_ids"].clone()
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        tokenized["labels"] = labels
        return tokenized

    def encode_batch(self, batch: list[dict[str, Any]]) -> list[str]:
        return [self.encode_item(item) for item in batch]

    def encode_item(self, item: dict[str, Any]) -> str:
        audio = item["audio"]["array"]
        splits = split_audio(audio)
        eos = self.tokenizer.eos_token or ""

        if len(splits) == 1:
            return self._speech_text(splits[0]) + eos

        before, text_span, after = splits
        text = self._transcribe(text_span)
        before_text = self._speech_text(before) if len(before) >= self.min_speech_samples else ""
        after_text = self._speech_text(after) if len(after) >= self.min_speech_samples else ""

        parts = []
        if before_text:
            parts.append(f"[Speech]{before_text}")
        if text:
            parts.append(f"[Text]{text.strip()}")
        if after_text:
            parts.append(f"[Speech]{after_text}")

        return "".join(parts or ["[Text]"]) + eos

    def _speech_text(self, audio: np.ndarray) -> str:
        codes = encode_audio(
            audio,
            self.wavtokenizer,
            self.wavtokenizer_repo_path,
            self.device,
            input_sr=self.sampling_rate,
        )
        return codes_to_text(codes)

    def _transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""
        return self.asr_pipeline(audio)["text"]
