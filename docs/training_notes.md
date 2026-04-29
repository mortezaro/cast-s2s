# Training Notes

The original notebook mixed environment setup, data loading, model creation, checkpointing, and inference logic in one file. This repository separates those responsibilities so runs can be reproduced from a config file.

Recommended workflow:

1. Save your dataset with `datasets.save_to_disk`.
2. Point `configs/train.local.yaml` at the dataset and WavTokenizer files.
3. Set `HF_TOKEN` only in the shell if the base model or Hub repo is private.
4. Train locally or with `accelerate launch`.
5. Keep merged-model exports separate from adapter checkpoints.

For multi-GPU training, launch with Accelerate or torchrun. The script honors `LOCAL_RANK` and maps each process to its assigned GPU.

The collator performs ASR and speech tokenization during training. That is flexible but expensive. For large runs, precomputing interleaved examples is often faster.
