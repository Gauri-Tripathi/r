"""Source annotations stay source annotations; no invented correction labels."""

import pytest

from duplex.config import load_config
from duplex.model import ROOT
from duplex.openyap import CONVERSATION_SYSTEM, split_speakers, transcript_words
from duplex.replay import ReplayRecord


def test_full_data_boundary_words_are_masked_not_invented():
    from duplex.openyap_full import window_words

    record = {
        "transcripts": {
            "a": {
                "words": [
                    {"word": "partial", "start": 0.5, "end": 1.2, "type": "word"},
                    {"word": "whole", "start": 1.3, "end": 1.6, "type": "word"},
                ]
            }
        }
    }
    words, regions = window_words(record, "a", 1, 2)
    assert [w.text for w in words] == ["whole"]
    assert regions[0][0] == 0 and regions[0][1] == pytest.approx(0.2)


def test_full_data_causal_text_queue_is_bounded():
    from duplex.codec import causal_align_text
    from duplex.fixtures import FixtureTokenizer
    from duplex.schema import Word

    words = [
        Word(text="one two", start_s=0, end_s=0.2),
        Word(text="three", start_s=0.08, end_s=0.3),
    ]
    _, stats = causal_align_text(words, 20, FixtureTokenizer())
    assert 0 < stats["max_piece_queue_delay_s"] <= 0.24
    with pytest.raises(ValueError, match="bounded"):
        causal_align_text(
            [
                Word(text="one two three four five six seven", start_s=0, end_s=0.5),
                Word(text="eight", start_s=0.08, end_s=0.7),
            ],
            20,
            FixtureTokenizer(),
        )


def test_full_training_is_data_sized_not_the_pilot_cap():
    from math import ceil

    cfg = load_config(ROOT / "configs/openyap-full.yaml")
    assert cfg.experiment_kind == "conversation_full"
    assert cfg.max_effective_epochs == 1
    assert ceil(544 * cfg.max_effective_epochs / cfg.train.accumulation) == 136
    assert cfg.train.max_steps == 500


def test_connected_speakers_cannot_cross_splits():
    pairs = [("a", "b"), ("b", "c"), ("d", "e"), ("f", "g"), ("g", "h")]
    rows = [
        {
            "key": str(i),
            "record": {
                "speakers": {
                    side: {"speaker_id": speaker}
                    for side, speaker in zip(("a", "b"), pair, strict=True)
                }
            },
        }
        for i, pair in enumerate(pairs)
    ]
    splits = split_speakers(rows)
    seen = {}
    for row, pair in zip(rows, pairs, strict=True):
        for speaker in pair:
            if speaker in seen:
                assert seen[speaker] == splits[row["key"]]
            seen[speaker] = splits[row["key"]]
    assert set(splits.values()) == {"train", "dev", "test"}
    with pytest.raises(ValueError, match="Too few"):
        split_speakers(rows[:2])


def test_source_timestamps_and_filler_preserved_without_repair():
    record = {
        "transcripts": {
            "a": {
                "words": [
                    {"word": "hello", "start": 1.0, "end": 1.3, "type": "word"},
                    {"word": "um", "start": 1.5, "end": 1.7, "type": "filler"},
                    {"word": "laugh", "start": 2.0, "end": 2.5, "type": "laugh"},
                ]
            }
        }
    }
    words = transcript_words(record, "a", 0.8, 3)
    assert [w.text for w in words] == ["hello", "um"]
    assert words[0].start_s == pytest.approx(0.2)
    with pytest.raises(ValueError, match="cuts"):
        transcript_words(record, "a", 1.2, 3)


def test_pilot_recipe_preserves_selective_lora_and_has_no_teacher():
    cfg = load_config(ROOT / "configs/openyap.yaml")
    assert cfg.experiment_kind == "conversation_pilot"
    assert cfg.train.layers == [28, 29, 30, 31] and cfg.train.rank == 8
    assert cfg.max_effective_epochs == 2
    assert "appointment" not in CONVERSATION_SYSTEM
    assert "supervision" not in ReplayRecord.model_fields
