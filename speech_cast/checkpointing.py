from __future__ import annotations

import os

from transformers import TrainerCallback, TrainerState


class SafeCheckpointCallback(TrainerCallback):
    def __init__(self, save_steps: int, output_dir: str):
        self.save_steps = save_steps
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def on_step_end(self, args, state: TrainerState, control, **kwargs):
        if state.global_step == 0 or state.global_step % self.save_steps != 0:
            return control

        model = kwargs["model"]
        tokenizer = kwargs.get("tokenizer")
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        if tokenizer:
            tokenizer.save_pretrained(checkpoint_dir)
        state.save_to_json(os.path.join(checkpoint_dir, "trainer_state.json"))

        print(f"[INFO] Saved checkpoint to {checkpoint_dir}")
        return control


class HubCheckpointCallback(TrainerCallback):
    def __init__(self, repo_id: str | None, save_steps: int):
        self.repo_id = repo_id
        self.save_steps = save_steps

    def on_step_end(self, args, state, control, **kwargs):
        if not self.repo_id or state.global_step == 0 or state.global_step % self.save_steps != 0:
            return control

        model = kwargs["model"]
        tokenizer = kwargs.get("tokenizer")
        model.push_to_hub(self.repo_id)
        if tokenizer:
            tokenizer.push_to_hub(self.repo_id)
        print(f"[INFO] Pushed checkpoint {state.global_step} to {self.repo_id}")
        return control
