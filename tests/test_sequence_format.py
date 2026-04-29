from speech_cast.codec import codes_to_text, speech_tokens, text_to_codes
from speech_cast.datasets import split_audio


def test_speech_token_names_are_one_indexed():
    assert speech_tokens(3) == ["[Sp1]", "[Sp2]", "[Sp3]"]


def test_code_roundtrip():
    codes = [0, 17, 4095]
    assert text_to_codes(codes_to_text(codes)) == codes


def test_split_audio_returns_three_segments_for_normal_audio():
    splits = split_audio([0.0] * 16000)
    assert len(splits) == 3
    assert sum(len(part) for part in splits) == 16000
