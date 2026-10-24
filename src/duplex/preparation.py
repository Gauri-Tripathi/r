"""Separate audio preparation for checked teacher training data and user-only evaluation."""

from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .audio import SAMPLE_RATE, SpeechRenderer, _rendered_row, render_example
from .codec import CodecMeasurements, build_sample, save_sample
from .data import save_examples
from .io import atomic_json, digest, file_hash, read_json
from .research import select_groups
from .runtime import schedule
from .schema import Example
from .supervision import require_teacher


def prepare_evaluation(
    rows: list[Example],
    root: Path,
    renderer: SpeechRenderer,
    tokenizer: Any,
    response_window_s: float = 15,
) -> dict[str, Any]:
    """Render only the scripted user. No target continuation/audio is a model input.

    Reference prefix duration establishes an intended interruption time. Whether
    the generated assistant actually speaks then is measured in evaluation.
    """
    if any(r.split == "train" for r in rows) or not rows:
        raise ValueError("Evaluation preparation requires a locked dev/test subset")
    lock = digest([r.model_dump() for r in rows])
    if (root / "evaluation-lock.json").exists() and read_json(root / "evaluation-lock.json")[
        "source_fingerprint"
    ] != lock:
        raise ValueError("Evaluation source changed; choose a new versioned directory")
    grouped: dict[str, list[Example]] = defaultdict(list)
    for row in rows:
        grouped[row.group_id].append(row)
    rendered = []
    started = time.perf_counter()
    for group in grouped.values():
        first = group[0]
        request, _ = renderer.render(first.script[0].text)
        prefix, _ = renderer.render(first.script[1].text)
        correction = next((u for u in first.script if u.kind == "correction"), None)
        correction_audio, _ = (
            renderer.render(correction.text) if correction else (np.zeros(0, dtype=np.float32), [])
        )
        prefix_start = len(request) / SAMPLE_RATE + 0.32
        prefix_end = prefix_start + len(prefix) / SAMPLE_RATE
        correction_start = prefix_start + (
            0.48 if first.timing == "overlap" else len(prefix) / SAMPLE_RATE + 0.32
        )
        correction_end = correction_start + len(correction_audio) / SAMPLE_RATE
        arrival = max(prefix_end, correction_end) + 0.24
        versions = []
        for row in group:
            events = [
                e.model_copy(
                    update={
                        "arrival_s": prefix_start + 0.48
                        if e.arrival_s < 10
                        else arrival + (1 if e.version == 1 else 0)
                    }
                )
                for e in row.visible.events
            ]
            context, audit = schedule(
                events, math.ceil((arrival + 40) * 12.5), 12.5, row.group_id, tokenizer
            )
            delivered = (
                max((i + 1 for i, token in enumerate(context) if token is not None), default=0)
                / 12.5
            )
            versions.append((row, events, audit, delivered))
        # One duration per counterfactual group, independent of the desired answer.
        duration = max(arrival, max(v[3] for v in versions)) + response_window_s
        n = math.ceil(duration * SAMPLE_RATE / 1920) * 1920
        user = np.zeros(n, dtype=np.float32)
        user[: len(request)] = request
        offset = round(correction_start * SAMPLE_RATE)
        user[offset : offset + len(correction_audio)] = correction_audio
        for row, events, audit, delivered in versions:
            directory = root / "audio" / row.id
            directory.mkdir(parents=True, exist_ok=True)
            sf.write(directory / "user.wav", user, SAMPLE_RATE, subtype="FLOAT")
            payload = {
                "backend": renderer.backend,
                "renderer_version": renderer.version,
                "voice": renderer.voice,
                "user": f"audio/{row.id}/user.wav",
                "user_sha256": file_hash(directory / "user.wav"),
                "events": [e.model_dump() for e in events],
                "context": audit,
                "duration_s": n / SAMPLE_RATE,
                "sample_rate": SAMPLE_RATE,
                "prefix_start_s": prefix_start,
                "prefix_end_s": prefix_end,
                "correction_start_s": correction_start,
                "correction_end_s": correction_end,
                "response_start_s": max(correction_end, delivered),
                "words": [],
                "evaluation_only": True,
                "assistant_continuation_supplied": False,
                "alignment": "Measured user construction boundaries; prefix duration is a timing reference, not model input",
            }
            atomic_json(directory / "manifest.json", payload)
            rendered.append(_rendered_row(row.model_copy(update={"supervision": None}), payload))
    save_examples(root / "rendered.jsonl", rendered)
    report = {
        "kind": "locked_user_audio_evaluation",
        "source_fingerprint": lock,
        "examples": len(rendered),
        "groups": len(grouped),
        "audio_s": sum(r.visible.duration_s for r in rendered),
        "wall_s": time.perf_counter() - started,
        "model_visible": "user WAV and arrived facts only",
        "rendered_fingerprint": digest([r.model_dump() for r in rendered]),
        "teacher_required": False,
    }
    atomic_json(root / "evaluation-lock.json", report)
    return report


def prepare_training(
    rows: list[Example],
    root: Path,
    renderer: SpeechRenderer,
    tokenizer: Any,
    mimi: Any,
    cache: Path,
    max_audio_s: float,
    measure_groups: int | None = None,
) -> dict[str, Any]:
    require_teacher(rows)
    if any(r.split == "test" for r in rows) or not rows:
        raise ValueError("Training preparation accepts checked train/dev data only")
    if max_audio_s <= 0 or max_audio_s > 10800:
        raise ValueError("Choose a bounded actual-audio cap <=10800 seconds")
    source = digest([r.model_dump() for r in rows])
    recipe = digest(
        {
            "source": source,
            "cap": max_audio_s,
            "measurement_groups": measure_groups,
            "tts": renderer.version,
            "voice": renderer.voice,
            "schema": "prepared-teacher-audio-v1",
        }
    )
    if (root / "preparation-input.json").exists() and read_json(root / "preparation-input.json")[
        "recipe"
    ] != recipe:
        raise ValueError("Preparation recipe changed; use a fresh dataset directory")
    atomic_json(root / "preparation-input.json", {"recipe": recipe, "source": source})
    if (root / "prepared.json").exists() and read_json(root / "prepared.json")[
        "source_fingerprint"
    ] != source:
        raise ValueError("Prepared input changed; create a new versioned dataset directory")
    chosen: list[Example] = []
    exclusions: list[dict[str, Any]] = []
    measurements = CodecMeasurements()
    started = time.perf_counter()
    seconds = 0.0
    for split in ("train", "dev"):
        split_seconds = 0.0
        split_limit = (
            max_audio_s if measure_groups else max_audio_s * (0.8 if split == "train" else 0.2)
        )
        ordered = select_groups(rows, split, measure_groups or 200)
        grouped: dict[str, list[Example]] = defaultdict(list)
        for row in ordered:
            grouped[row.group_id].append(row)
        for group_id, group in grouped.items():
            rendered = [render_example(row, root, renderer, tokenizer) for row in group]
            duration = sum(row.visible.duration_s for row in rendered)
            if seconds + duration > max_audio_s or split_seconds + duration > split_limit:
                exclusions.append(
                    {"group": group_id, "reason": "actual_duration_cap", "audio_s": duration}
                )
                break
            try:
                samples = [
                    build_sample(
                        row, root, mimi, tokenizer, cache, "cpu", measurements=measurements
                    )
                    for row in rendered
                ]
                if any(s["codes"].shape[-1] > 750 for s in samples):
                    raise ValueError("complete_session_exceeds_750_frames")
            except ValueError as exc:
                exclusions.append(
                    {"group": group_id, "reason": type(exc).__name__, "detail": str(exc)[:180]}
                )
                continue
            for row, sample in zip(rendered, samples, strict=True):
                save_sample(root / "encoded" / f"{row.id}.safetensors", sample, row)
            chosen.extend(rendered)
            seconds += duration
            split_seconds += duration
            save_examples(root / "rendered.jsonl", chosen)
            print(
                f"Prepared {len(chosen)} examples / {seconds:.1f} unique session-seconds",
                flush=True,
            )
            # Stop selecting once the remaining duration cannot hold even one short group.
            if max_audio_s - seconds < 15:
                break
    encoded = {p.name: file_hash(p) for p in sorted((root / "encoded").glob("*.safetensors"))}
    report = {
        "kind": "preparation_measurement_only" if measure_groups else "checked_teacher_audio",
        "source_fingerprint": source,
        "examples": len(chosen),
        "audio_s": seconds,
        "splits": {split: sum(r.split == split for r in chosen) for split in ("train", "dev")},
        "split_audio_s": {
            split: sum(r.visible.duration_s for r in chosen if r.split == split)
            for split in ("train", "dev")
        },
        "wall_s": time.perf_counter() - started,
        "codec": measurements.report(),
        "exclusions": exclusions,
        "encoded": encoded,
        "stronger_teacher_claim_verified": False,
        "teacher_provenance": sorted(
            {(r.provenance["teacher_model"], r.provenance["teacher_revision"]) for r in chosen}
        ),
    }
    atomic_json(root / "prepared.json", report)
    if not chosen:
        raise ValueError(
            "No complete groups survived audio/alignment checks; inspect prepared.json exclusions"
        )
    return report
