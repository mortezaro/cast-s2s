# CAST-S2S

CAST-S2S is a compact speech-to-speech language-model training stack. It interleaves discrete speech tokens with transcribed text spans, then fine-tunes a causal language model so the same sequence format can represent speech-only, text-only, and mixed speech/text examples.

The code is intentionally model- and dataset-agnostic. Credentials, local paths, dataset names, and organization-specific details are passed through config files or environment variables.

## What Is Included

- LoRA fine-tuning for Gemma-style causal language models
- WavTokenizer-based speech tokenization
- Whisper-based text span transcription during collation
- Train/validation splitting for Hugging Face datasets saved on disk
- Safe checkpointing for adapters, tokenizer files, and trainer state
- Inference utilities for audio-to-token, prompt construction, generation, and token extraction
- Minimal scripts that keep experiments reproducible without notebook state

## Repository Layout

```text
.
├── configs/
│   ├── train.example.yaml
│   └── inference.example.yaml
├── docs/
│   ├── data_format.md
│   └── training_notes.md
├── examples/
│   └── prompt.txt
├── scripts/
│   ├── train.py
│   └── infer.py
├── src/
│   └── cast_s2s/
│       ├── audio_tokens.py
│       ├── callbacks.py
│       ├── config.py
│       ├── data.py
│       ├── generation.py
│       └── model.py
└── tests/
    └── test_sequence_format.py
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[train]"
```

WavTokenizer is loaded from a local checkout because upstream installations vary. Set the path in the config:

```yaml
wavtokenizer_repo_path: /path/to/WavTokenizer
```

The Hugging Face token is read from `HF_TOKEN` when required:

```bash
export HF_TOKEN=...
```

Do not put tokens in config files or commits.

## Training

Copy the example config and adjust paths:

```bash
cp configs/train.example.yaml configs/train.local.yaml
python scripts/train.py --config configs/train.local.yaml
```

The dataset should be a Hugging Face dataset saved with `datasets.save_to_disk`. It needs an `audio` column that can be cast to `datasets.Audio`.

The trainer builds examples on the fly:

```text
[Speech][Sp12][Sp2048]...[Text]transcribed span[Speech][Sp99]...</s>
```

This keeps storage light and allows randomized text/speech interleaving across epochs.

## Inference

```bash
python scripts/infer.py \
  --config configs/inference.example.yaml \
  --audio path/to/input.wav \
  --output-dir generated/sample
```

The inference CLI follows the published speech-to-speech path:

1. Download or load the companion WavTokenizer codec.
2. Encode a 16 kHz prompt into `[Sp1]` ... `[Sp4096]` speech tokens.
3. Constrain generation so the LM can sample only speech tokens and EOS.
4. Decode the generated codes at 24 kHz.
5. Save `recon_24k.wav`, `continuation.wav`, `stitched_24k.wav`, and generated code IDs.

The default example config points to the public CAST 0.7B checkpoint and companion codec. For local models, set `model_name_or_path`, `base_model_name_or_path`, and the codec fields in a local config.

## Checkpoints

By default, training saves:

- Trainer checkpoints under `output_dir`
- Adapter checkpoints under `checkpoint_dir`
- Final model/adapters under `output_dir/final`

If `hub_repo_id` is set, checkpoints can also be pushed to the Hub. The token is still read only from `HF_TOKEN`.

## Notes

This repo is a cleaned implementation of the training and inference workflow only. It intentionally excludes private dataset names, local infrastructure paths, access tokens, personal identifiers, and notebook-specific scratch code.
