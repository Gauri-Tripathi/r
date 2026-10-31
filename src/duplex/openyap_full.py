"""Full public-sample preparation, with explicit coverage and resumable conversations."""

from __future__ import annotations

import math
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .codec import causal_align_text
from .io import atomic_json, digest, file_hash, read_json, write_jsonl
from .openyap import REPO, REVISION, inventory, split_speakers
from .schema import Word


def conversation_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Approximately 80/10/10 by duration, whole conversations; speakers may overlap."""
    total = sum(r["record"]["duration_seconds"] for r in rows)
    pairs = [(a, b) for a in rows for b in rows if a["key"] != b["key"]]
    dev, test = min(
        pairs,
        key=lambda pair: (
            sum(abs(r["record"]["duration_seconds"] - total * 0.1) for r in pair),
            digest([r["key"] for r in pair]),
        ),
    )
    return {
        r["key"]: "dev"
        if r["key"] == dev["key"]
        else "test"
        if r["key"] == test["key"]
        else "train"
        for r in rows
    }


def window_words(
    record: dict[str, Any], side: str, start: float, end: float
) -> tuple[list[Word], list[tuple[float, float]]]:
    words, partial = [], []
    for item in record["transcripts"][side]["words"]:
        if item["type"] not in ("word", "filler"):
            continue
        a, b = float(item["start"]), float(item["end"])
        if b <= start or a >= end:
            continue
        if b <= a or "<" in item["word"] or ">" in item["word"]:
            raise ValueError("Invalid lexical annotation")
        if a < start or b > end:
            partial.append((max(0, a - start), min(end - start, b - start)))
        else:
            words.append(Word(text=item["word"], start_s=a - start, end_s=b - start))
    return words, partial


def full_prepare(
    source: Path,
    root: Path,
    tokenizer: Any,
    split_mode: str = "conversation",
    plan_only: bool = False,
) -> dict[str, Any]:
    rows = inventory(source)
    splits = conversation_splits(rows) if split_mode == "conversation" else split_speakers(rows)
    recipe = digest(
        {
            "revision": REVISION,
            "splits": splits,
            "window_s": 48,
            "tail_s": 0.64,
            "alignment": "bounded-causal-text-v1",
            "version": 1,
        }
    )
    if (root / "prepared-input.json").exists() and read_json(root / "prepared-input.json")[
        "recipe"
    ] != recipe:
        raise ValueError("Full-data recipe changed; use a new directory")
    atomic_json(
        root / "prepared-input.json",
        {
            "recipe": recipe,
            "conversation_splits": splits,
            "split_mode": split_mode,
            "revision": REVISION,
        },
    )
    chosen: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    coverage: dict[str, float] = defaultdict(float)
    for item in rows:
        key, record = item["key"], item["record"]
        split = splits[key]
        marker = root / "progress" / f"{key}.json"
        if marker.exists() and not plan_only:
            prior = read_json(marker)
            if prior["recipe"] != recipe:
                raise ValueError("Resume recipe mismatch")
            for row in prior["rows"]:
                timeline = read_json(root / "audio" / row["id"] / "manifest.json")
                if any(
                    file_hash(root / row[c]) != timeline[c + "_sha256"]
                    for c in ("user", "assistant")
                ):
                    raise ValueError("Completed preparation waveform changed")
            chosen.extend(prior["rows"])
            exclusions.update(prior["exclusions"])
            coverage[split] += prior["source_audio_s"]
            print(f"Verified/resumed {key}", flush=True)
            continue
        selected = []
        omitted: Counter[str] = Counter()
        covered = 0.0
        with tarfile.open(source / item["shard"]) as archive:
            streams = [archive.extractfile(f"{key}.{side}.flac") for side in ("a", "b")]
            if any(s is None for s in streams):
                raise ValueError("Missing source tracks")
            with sf.SoundFile(streams[0]) as a_file, sf.SoundFile(streams[1]) as b_file:
                if (
                    a_file.channels != 1
                    or b_file.channels != 1
                    or a_file.samplerate != b_file.samplerate
                    or len(a_file) != len(b_file)
                ):
                    raise ValueError("Invalid shared timeline")
                sr = a_file.samplerate
                duration = min(record["duration_seconds"], len(a_file) / sr)
                for index, start in enumerate(np.arange(0, duration, 48)):
                    start = float(start)
                    end = min(start + 48, duration)
                    if end - start < 2:
                        omitted["final_fragment_under_2s"] += 1
                        continue
                    assistant = "a" if (int(digest(key)[0], 16) + index) % 2 else "b"
                    user = "b" if assistant == "a" else "a"
                    session_s = math.ceil((end - start + 0.64) * 12.5) / 12.5
                    try:
                        words, partial = window_words(record, assistant, start, end)
                        _, alignment = causal_align_text(words, round(session_s * 12.5), tokenizer)
                    except ValueError as exc:
                        omitted[str(exc)] += 1
                        continue
                    identifier = f"{key}-full-{index:05d}-{assistant}"
                    row = {
                        "id": identifier,
                        "user": f"audio/{identifier}/user.wav",
                        "assistant": f"audio/{identifier}/assistant.wav",
                        "words": [w.model_dump() for w in words],
                        "source": f"hf://datasets/{REPO}@{REVISION}/{key}",
                        "license": "CC-BY-4.0",
                        "isolated_channels_verified": True,
                        "duration_s": session_s,
                        "split": split,
                        "conversation_id": key,
                        "alignment_mode": "causal_queue",
                        "partial_text_regions": partial,
                    }
                    if not plan_only:
                        channels = {}
                        for side, file in (("a", a_file), ("b", b_file)):
                            file.seek(round(start * sr))
                            channels[side] = file.read(round((end - start) * sr), dtype="float32")
                        if any(not np.isfinite(x).all() for x in channels.values()):
                            omitted["nonfinite_audio"] += 1
                            continue
                        clipped = max(float(np.mean(np.abs(x) >= 0.999)) for x in channels.values())
                        if clipped > 0.001:
                            omitted["source_clipping_over_0.1_percent"] += 1
                            continue
                        divisor = math.gcd(sr, 24000)
                        channels = {
                            s: resample_poly(x, 24000 // divisor, sr // divisor).astype(np.float32)
                            for s, x in channels.items()
                        }
                        gain = min(
                            1.0,
                            0.95
                            / max(1e-9, max(float(np.max(np.abs(x))) for x in channels.values())),
                        )
                        directory = root / "audio" / identifier
                        directory.mkdir(parents=True, exist_ok=True)
                        for name, side in (("user", user), ("assistant", assistant)):
                            pcm = np.zeros(round(session_s * 24000), dtype=np.float32)
                            x = channels[side] * gain
                            pcm[: min(len(x), len(pcm))] = x[: len(pcm)]
                            sf.write(directory / f"{name}.wav", pcm, 24000, subtype="FLOAT")
                        atomic_json(
                            directory / "manifest.json",
                            {
                                **row,
                                **alignment,
                                "source_start_s": start,
                                "source_end_s": end,
                                "source_clipped_fraction": clipped,
                                "joint_peak_gain": gain,
                                "padding_s": session_s - (end - start),
                                "sample_rate": 24000,
                                "user_sha256": file_hash(directory / "user.wav"),
                                "assistant_sha256": file_hash(directory / "assistant.wav"),
                                "transcripts": "machine generated; boundary-crossing words masked from text loss",
                            },
                        )
                    selected.append(row)
                    covered += end - start
        for stream in streams:
            if stream is not None:
                stream.close()
        chosen.extend(selected)
        exclusions.update(omitted)
        coverage[split] += covered
        if not plan_only:
            atomic_json(
                marker,
                {
                    "recipe": recipe,
                    "rows": selected,
                    "exclusions": dict(omitted),
                    "source_audio_s": covered,
                },
            )
        print(
            f"{key}: {len(selected)} windows, {covered / 3600:.3f} source hours ({split})",
            flush=True,
        )
    report = {
        "kind": "openyap_full_plan" if plan_only else "openyap_full",
        "revision": REVISION,
        "source_hours": sum(r["record"]["duration_seconds"] for r in rows) / 3600,
        "split_mode": split_mode,
        "speaker_disjoint": split_mode == "speaker",
        "selected": dict(Counter(r["split"] for r in chosen)),
        "source_audio_s": dict(coverage),
        "audio_s": {
            s: sum(r["duration_s"] for r in chosen if r["split"] == s)
            for s in ("train", "dev", "test")
        },
        "exclusions": dict(exclusions),
        "recipe": recipe,
        "teacher_or_tts_used": False,
        "all_audio_validated": not plan_only,
        "limitations": "Conversation splits share speakers; no unseen-speaker claim. Causal text token serialization allows <=240ms queue delay, with original word timestamps retained. Partial words at window edges have no text loss.",
    }
    path = root / ("plan.json" if plan_only else "prepared.json")
    if not plan_only:
        write_jsonl(root / "replay.jsonl", chosen)
        report["manifest_sha256"] = file_hash(root / "replay.jsonl")
    atomic_json(path, report)
    return report
