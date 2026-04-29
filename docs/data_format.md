# Data Format

Training expects a Hugging Face dataset saved to disk.

Required column:

```text
audio
```

The `audio` column may contain file paths or decoded arrays, as long as it can be cast with:

```python
dataset.cast_column("audio", Audio(sampling_rate=16000, decode=True))
```

Rows with missing audio paths are filtered before decoding. The training collator then samples one span for transcription and keeps the surrounding spans as discrete speech tokens.

Canonical sequence forms:

```text
[Speech]<speech tokens>[Text]<transcription>[Speech]<speech tokens></s>
[Speech]<speech tokens>[Text]<transcription></s>
[Text]<transcription>[Speech]<speech tokens></s>
<speech tokens></s>
```

Speech tokens are represented as `[Sp1]` through `[Sp4096]`.
