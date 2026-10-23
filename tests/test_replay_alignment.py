from __future__ import annotations

from pathlib import Path

import pytest

from duplex.codec import align_text
from duplex.fixtures import FixtureTokenizer
from duplex.io import atomic_json
from duplex.replay import mix_frames
from duplex.schema import Word


def test_text_alignment_uses_measured_boundaries_and_rejects_collision() -> None:
    words = [
        Word(text="Hello", start_s=0.4, end_s=0.6),
        Word(text="Ananya", start_s=1.2, end_s=1.6),
    ]
    stream = align_text(words, 30, FixtureTokenizer())
    assert stream[0, 0, 5] != 3 and stream[0, 0, 15] != 3
    assert stream[0, 0, 4] == 0
    with pytest.raises(ValueError, match="collision"):
        align_text(
            [
                Word(text="one two three", start_s=0, end_s=1),
                Word(text="four", start_s=0.08, end_s=1),
            ],
            20,
            FixtureTokenizer(),
        )


def test_bounded_replay_fraction(tmp_path: Path) -> None:
    paths = []
    for index in range(11):
        path = tmp_path / f"{index}.safetensors"
        path.touch()
        atomic_json(
            path.with_suffix(".json"),
            {"audio_frames": 100 if index < 10 else 50, "replay": index == 10},
        )
        paths.append(path)
    mixed, report = mix_frames(paths, 0.15)
    assert 0.1 <= report["replay_fraction"] <= 0.2
    assert len(mixed) > 10
    _, absent = mix_frames(paths[:10], 0.15)
    assert absent["replay_fraction"] == 0
