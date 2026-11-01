"""WAV replay bridge for Full-Duplex-Bench v1/v1.5 inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import soundfile as sf

from .audio import load_wav
from .budget import Deadline
from .inference import generate
from .io import atomic_json, file_hash
from .schema import ModelInput

BENCH_COMMIT = "3e799c45a045256f47d5f1c9cda90157e2d2ec9e"


def limited_replay(
    lm: Any,
    mimi: Any,
    tokenizer: Any,
    input_root: Path,
    out: Path,
    limit: int,
    deadline: Deadline,
    voice: Path | None = None,
) -> dict[str, Any]:
    if not 1 <= limit <= 8:
        raise ValueError("Limited benchmark bridge supports 1..8 input WAVs")
    inputs = sorted(input_root.glob("**/input.wav"))[:limit]
    if not inputs:
        raise ValueError("Supply a licensed v1/v1.5 user_interruption subset with input.wav files")
    results = []
    for index, path in enumerate(inputs):
        if path.is_symlink() or not path.resolve().is_relative_to(input_root.resolve()):
            raise ValueError("Benchmark input escapes supplied root")
        audio = load_wav(path, max_s=60)
        duration = len(audio) / 24000
        if not deadline.can_step(duration * 3):
            raise ValueError("Insufficient benchmark reserve")
        dest = out / f"item-{index:03d}"
        dest.mkdir(parents=True, exist_ok=True)
        sf.write(dest / "input.wav", audio, 24000, subtype="FLOAT")
        visible = ModelInput(
            request_id=f"bench-{index}", user_audio="input.wav", events=[], duration_s=duration
        )
        result = generate(lm, mimi, tokenizer, visible, dest, dest, voice=voice)
        (dest / "generated.wav").replace(dest / "output.wav")
        result.update(
            input_sha256=file_hash(path), subset="user_interruption", upstream_commit=BENCH_COMMIT
        )
        results.append(result)
    report = {
        "kind": "system replay bridge, NOT official benchmark score",
        "subset": "v1_v1.5 user_interruption",
        "version": BENCH_COMMIT,
        "modifications": "24kHz mono conversion, bounded subset, same appointment role prompt (not official task prompt); no upstream scoring run",
        "results": results,
        "cost": "included in reservation ledger; no external ASR or judge request",
    }
    atomic_json(out / "benchmark-bridge.json", report)
    return report
