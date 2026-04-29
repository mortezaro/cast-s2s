#!/usr/bin/env python
from __future__ import annotations

import argparse
import os

import torch
from transformers import Trainer, TrainingArguments

from cast_s2s.audio_tokens import load_wavtokenizer
from cast_s2s.callbacks import HubCheckpointCallback, SafeCheckpointCallback
from cast_s2s.config import load_train_config
from cast_s2s.data import InterleavedCollator, load_audio_dataset
from cast_s2s.model import (
    apply_lora,
    load_asr_pipeline,
    load_causal_model,
    load_tokenizer,
    local_device_map,
    print_trainable_parameters,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_train_config(args.config)
    token = os.environ.get("HF_TOKEN")
    dtype = torch.bfloat16 if cfg.bf16 else torch.float16
    device_map = local_device_map()
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}" if torch.cuda.is_available() else "cpu")

    train_dataset, eval_dataset = load_audio_dataset(
        cfg.dataset_path,
        sampling_rate=cfg.audio_sampling_rate,
        split_ratio=cfg.train_test_split,
        seed=cfg.seed,
    )

    wavtokenizer = load_wavtokenizer(
        cfg.wavtokenizer_repo_path,
        cfg.wavtokenizer_config,
        cfg.wavtokenizer_checkpoint,
        device,
    )
    asr_pipeline = load_asr_pipeline(cfg.asr_model_name_or_path, device_map, dtype, token)
    tokenizer = load_tokenizer(cfg.model_name_or_path, cfg.speech_token_count, token)
    model = load_causal_model(cfg.model_name_or_path, len(tokenizer), device_map, dtype, token)

    if cfg.use_lora:
        model = apply_lora(
            model,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            train_embeddings=cfg.train_embeddings,
            train_lm_head=cfg.train_lm_head,
        )

    print_trainable_parameters(model)

    collator = InterleavedCollator(
        tokenizer=tokenizer,
        wavtokenizer=wavtokenizer,
        asr_pipeline=asr_pipeline,
        wavtokenizer_repo_path=cfg.wavtokenizer_repo_path,
        device=device,
        sampling_rate=cfg.audio_sampling_rate,
        max_length=cfg.max_length,
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        lr_scheduler_type="linear",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=cfg.eval_steps,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        run_name="cast-s2s",
        bf16=cfg.bf16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[
            SafeCheckpointCallback(save_steps=cfg.save_steps, output_dir=cfg.checkpoint_dir),
            HubCheckpointCallback(repo_id=cfg.hub_repo_id, save_steps=cfg.save_steps),
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(cfg.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"[INFO] Saved final artifacts to {final_dir}")


if __name__ == "__main__":
    main()
