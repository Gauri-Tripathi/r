"""Score a locked base speech run using externally cached ASR; no teacher/judge API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .evaluate import audible_text, import_asr, score, summarize
from .io import atomic_json, digest, file_hash, read_json
from .schema import Example


def score_baseline(
    rows: list[Example], root: Path, generations: Path, asr: Path, out: Path
) -> dict[str, Any]:
    if not rows or any(r.split != "dev" for r in rows):
        raise ValueError(
            "Use the locked development subset for baseline difficulty checks; final test stays untouched"
        )
    lock = read_json(root / "evaluation-lock.json")
    if lock["rendered_fingerprint"] != digest([r.model_dump() for r in rows]):
        raise ValueError("Baseline subset changed after locking")
    transcripts = import_asr(asr)
    results = []
    settings = None
    for row in rows:
        directory = generations / row.id
        generation = read_json(directory / "generation.json")
        current = {
            key: generation[key]
            for key in ("base_revision", "runtime_fingerprint", "decoding", "voice_sha256")
        }
        if settings is not None and current != settings:
            raise ValueError("Baseline runtime/decoding/voice changed within the locked subset")
        settings = current
        if generation.get("adapter_sha256") is not None or generation["truncated"]:
            raise ValueError("Baseline must be a complete, adapter-disabled generation")
        if generation["input_fingerprint"] != digest(row.visible.model_dump()):
            raise ValueError("Baseline used different user audio/context inputs")
        if generation.get("user_audio_sha256") != file_hash(root / str(row.visible.user_audio)):
            raise ValueError("Baseline user waveform hash is missing or changed")
        if row.id not in transcripts:
            raise ValueError("Missing hash-matched cached ASR for a baseline example")
        text, words, _ = audible_text(generation, directory / "generated.wav", transcripts[row.id])
        timeline = read_json(root / "audio" / row.id / "manifest.json")
        latest = max(row.visible.events, key=lambda e: e.version) if row.visible.events else None
        available = max(timeline["correction_end_s"], latest.arrival_s if latest else 0)
        item = score(row, text, words, available)
        audio, sr = sf.read(directory / "generated.wav", dtype="float32")
        segment = audio[
            round(timeline["correction_start_s"] * sr) : round(timeline["correction_end_s"] * sr)
        ]
        item.update(
            actual_audible_output_during_correction=bool(
                len(segment) and np.any(np.abs(segment) > 0.003)
            ),
            asr_uncertainty=transcripts[row.id].uncertainty,
            rtf=generation["rtf"],
            silence_fraction=generation["silence_fraction"],
            truncation=False,
            listening_audio=str(directory / "generated.wav"),
        )
        results.append(item)
    aggregate = summarize(results)
    report = {
        "kind": "locked_base_baseline",
        "evidence": "audible_asr",
        "count": len(rows),
        "subset_fingerprint": lock["rendered_fingerprint"],
        "lock_sha256": file_hash(root / "evaluation-lock.json"),
        "summary": aggregate,
        "rows": results,
        "all_cases_solved": all(r["joint_success"] for r in results),
        "runtime_fingerprint": generation["runtime_fingerprint"],
        "base_revision": generation["base_revision"],
        "settings": settings,
        "limitations": "Lexical scorer and ASR uncertainty require listening review. No acoustic quality or universal-generalization claim.",
    }
    atomic_json(out, report)
    return report
