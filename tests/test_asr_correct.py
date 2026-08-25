"""Tests for the deterministic ASR homophone post-filter."""

import asr_correct
from asr_correct import correct_transcript, load_map


def setup_function(fn):
    asr_correct._MAP = None  # force reload per test


def test_known_homophones_are_canonicalized():
    out = correct_transcript("chú quậy ngồi dưới gốc cây gì vậy")
    assert "chú cuội" in out.lower()
    assert "múa lân" in correct_transcript("ông ơi múa lan là gì vậy").lower()


def test_longest_match_wins():
    out = correct_transcript("xem lân sư rông đi")
    assert "lân sư rồng" in out
    assert "sư rông" not in out


def test_clean_text_untouched():
    src = "Đèn ông sao làm bằng tre và giấy."
    assert correct_transcript(src) == src


def test_sentence_capitalization_restored():
    out = correct_transcript("chú quậy là ai")
    assert out[0].isupper()


def test_csv_override(tmp_path, monkeypatch):
    csv_path = tmp_path / "asr_homophones.csv"
    csv_path.write_text("sai,dung\nem bé siêu nhân,em nhí\n", encoding="utf-8")
    monkeypatch.setattr(asr_correct, "_CSV_PATH", csv_path)
    mapping = load_map()
    assert mapping["em bé siêu nhân"] == "em nhí"
    assert "chú quậy" in mapping  # defaults preserved
    out = correct_transcript("em bé siêu nhân ơi")
    assert "siêu nhân" not in out and "Em nhí" in out
