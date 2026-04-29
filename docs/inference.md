# Inference

The speech-to-speech checkpoint generates codec tokens, not waveform samples directly.

The runtime path is:

```text
16 kHz prompt audio
-> WavTokenizer encode at 24 kHz
-> [Speech][Sp...][Sp...] language-model prefix
-> speech-only constrained generation
-> WavTokenizer decode
-> optional crossfade with the reconstructed prompt
```

The companion codec defaults to `KrauthammerLab/cast-wavtokenizer-24k-40tps`. If you already have WavTokenizer files locally, set `wavtokenizer_checkpoint` and optionally `wavtokenizer_config` in the config.

Run:

```bash
python inference/generate_continuation.py \
  --config configs/inference.example.yaml \
  --audio prompt_16k.wav \
  --output-dir generated/prompt_001
```

Outputs:

```text
generated/prompt_001/recon_24k.wav
generated/prompt_001/continuation.wav
generated/prompt_001/stitched_24k.wav
generated/prompt_001/codes.json
```

`recon_24k.wav` is the codec round trip of the prompt. `continuation.wav` is the decoded generated continuation, including the retained prompt tail used for smoother decoding. `stitched_24k.wav` crossfades the prompt reconstruction into the continuation.
