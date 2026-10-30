"""Pinned real OpenYap pilot data. No teacher, TTS, task labels or speaker cloning."""

from __future__ import annotations

import json
import math
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .audio import validate_alignment
from .codec import align_text
from .io import atomic_json, digest, file_hash, read_json, write_jsonl
from .schema import Word

REPO = "TheAgenticDataCompany/open-yap-1k"
REVISION = "a08b1c80e50159d462ddcbbabfdea5bf7ccfc626"
CONVERSATION_SYSTEM = (
    "You are a friendly conversational assistant. Listen and respond naturally in English."
)


def inventory(source: Path) -> list[dict[str, Any]]:
    verification = read_json(source / "verification.json")
    result = []
    for path in sorted(source.glob("shard-*.tar")):
        expected = verification["files"][path.name]
        if path.stat().st_size != expected["byteLength"] or file_hash(path) != expected["sha256"]:
            raise ValueError("OpenYap shard checksum mismatch")
        with tarfile.open(path) as archive:
            members = {m.name: m for m in archive if m.isfile()}
            for name in sorted(members):
                if not name.endswith(".json"):
                    continue
                member = members[name]
                if member.size > 8 * 1024**2:
                    raise ValueError("Oversized conversation metadata")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Missing metadata stream")
                with stream:
                    record = json.load(stream)
                key = name[:-5]
                if not all(key + suffix in members for suffix in (".a.flac", ".b.flac")):
                    raise ValueError(
                        "Separate speaker tracks missing; preview audio cannot substitute"
                    )
                result.append({"key": key, "shard": path.name, "record": record})
    if len(result) != 16:
        raise ValueError("The complete pinned sample must contain 16 conversations")
    return result


def split_speakers(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Connected speaker components cannot cross train/development/test."""
    components: list[set[str]] = []
    for row in rows:
        speakers = {row["record"]["speakers"][s]["speaker_id"] for s in ("a", "b")}
        touched = [c for c in components if c & speakers]
        for item in touched:
            components.remove(item)
            speakers |= item
        components.append(speakers)
    ordered = sorted(components, key=lambda c: digest(sorted(c)))
    if len(ordered) < 3:
        raise ValueError("Too few disconnected speaker groups for a speaker-disjoint pilot")
    assignment = {
        speaker: ("test" if i == len(ordered) - 1 else "dev" if i == len(ordered) - 2 else "train")
        for i, group in enumerate(ordered)
        for speaker in group
    }
    return {row["key"]: assignment[row["record"]["speakers"]["a"]["speaker_id"]] for row in rows}


def transcript_words(record: dict[str, Any], side: str, start: float, end: float) -> list[Word]:
    result = []
    for item in record["transcripts"][side]["words"]:
        if item["type"] not in ("word", "filler"):
            continue
        a, b = float(item["start"]), float(item["end"])
        if a < start < b or a < end < b:
            raise ValueError("Window cuts a transcribed word")
        if start <= a and b <= end:
            text = item["word"]
            if "<" in text or ">" in text:
                raise ValueError("Transcript contains control-like text")
            result.append(Word(text=text, start_s=a - start, end_s=b - start))
    validate_alignment(result, end - start)
    return result


def prepare(
    source: Path, root: Path, tokenizer: Any, train_per_conversation: int = 4
) -> dict[str, Any]:
    rows = inventory(source)
    splits = split_speakers(rows)
    recipe = digest(
        {
            "revision": REVISION,
            "splits": splits,
            "window_s": 16,
            "train_per_conversation": train_per_conversation,
            "version": 1,
        }
    )
    if (root / "prepared-input.json").exists() and read_json(root / "prepared-input.json")[
        "recipe"
    ] != recipe:
        raise ValueError("Pilot recipe changed; preserve prior data and use a new directory")
    atomic_json(
        root / "prepared-input.json",
        {"recipe": recipe, "conversation_splits": splits, "revision": REVISION},
    )
    chosen = []
    exclusions: Counter[str] = Counter()
    stats = []
    for item in rows:
        key, record, split = item["key"], item["record"], splits[item["key"]]
        assistant = "a" if int(digest(key)[0], 16) % 2 else "b"
        user = "b" if assistant == "a" else "a"
        limit = train_per_conversation if split == "train" else 2
        accepted = 0
        previous_end = -1.0
        with tarfile.open(source / item["shard"]) as archive:
            a_stream = archive.extractfile(f"{key}.{assistant}.flac")
            u_stream = archive.extractfile(f"{key}.{user}.flac")
            if a_stream is None or u_stream is None:
                raise ValueError("Missing isolated audio")
            with (
                a_stream,
                u_stream,
                sf.SoundFile(a_stream) as a_file,
                sf.SoundFile(u_stream) as u_file,
            ):
                if (
                    a_file.channels != 1
                    or u_file.channels != 1
                    or a_file.samplerate != u_file.samplerate
                    or len(a_file) != len(u_file)
                ):
                    raise ValueError("Source channels lack a shared mono timeline")
                sr = a_file.samplerate
                for start in np.arange(0, min(record["duration_seconds"], 1200) - 16, 0.8):
                    start = round(float(start), 2)
                    if start < previous_end:
                        continue
                    try:
                        words = transcript_words(record, assistant, start, start + 16)
                        user_words = transcript_words(record, user, start, start + 16)
                        if len(words) < 8 or len(user_words) < 4:
                            raise ValueError("Insufficient speech on both channels")
                        # Keep exact source word boundaries; no synthetic timing repair.
                        align_text(words, 200, tokenizer)
                        a_file.seek(round(start * sr))
                        u_file.seek(round(start * sr))
                        audio = [f.read(round(16 * sr), dtype="float32") for f in (u_file, a_file)]
                        if any(
                            len(x) != round(16 * sr)
                            or not np.isfinite(x).all()
                            or np.max(np.abs(x)) >= 0.999
                            for x in audio
                        ):
                            raise ValueError("Clipped, nonfinite or short audio")
                        if any(float(np.sqrt(np.mean(x * x))) < 0.002 for x in audio):
                            raise ValueError("Very low audio level")
                    except ValueError as exc:
                        exclusions[str(exc)] += 1
                        continue
                    identifier = f"{key}-{round(start * 100):07d}-{assistant}"
                    directory = root / "audio" / identifier
                    directory.mkdir(parents=True, exist_ok=True)
                    for name, x in zip(("user", "assistant"), audio, strict=True):
                        divisor = math.gcd(sr, 24000)
                        pcm = resample_poly(x, 24000 // divisor, sr // divisor).astype(np.float32)
                        if np.max(np.abs(pcm)) >= 0.999:
                            raise ValueError("Resampling clipped; no silent gain alteration")
                        sf.write(directory / f"{name}.wav", pcm, 24000, subtype="FLOAT")
                    selected = {
                        "id": identifier,
                        "user": f"audio/{identifier}/user.wav",
                        "assistant": f"audio/{identifier}/assistant.wav",
                        "words": [w.model_dump() for w in words],
                        "source": f"hf://datasets/{REPO}@{REVISION}/{key}",
                        "license": "CC-BY-4.0",
                        "isolated_channels_verified": True,
                        "duration_s": 16.0,
                        "split": split,
                        "conversation_id": key,
                    }
                    chosen.append(selected)
                    atomic_json(
                        directory / "manifest.json",
                        {
                            **selected,
                            "source_start_s": start,
                            "assistant_side": assistant,
                            "sample_rate": 24000,
                            "user_sha256": file_hash(directory / "user.wav"),
                            "assistant_sha256": file_hash(directory / "assistant.wav"),
                            "transcripts": "machine generated, not human verified",
                            "assistant_voice_prompt_supplied": False,
                        },
                    )
                    accepted += 1
                    previous_end = start + 16
                    if accepted >= limit:
                        break
        stats.append({"conversation": key, "split": split, "selected": accepted})
        print(f"Prepared {key}: {split}, {accepted} non-overlapping windows", flush=True)
    write_jsonl(root / "replay.jsonl", chosen)
    report = {
        "kind": "openyap_conversation_pilot",
        "revision": REVISION,
        "source_hours": sum(r["record"]["duration_seconds"] for r in rows) / 3600,
        "source_conversations": len(rows),
        "source_speakers": len(
            {r["record"]["speakers"][s]["speaker_id"] for r in rows for s in ("a", "b")}
        ),
        "selected": dict(Counter(r["split"] for r in chosen)),
        "audio_s": dict(
            Counter(
                {
                    split: sum(r["duration_s"] for r in chosen if r["split"] == split)
                    for split in ("train", "dev", "test")
                }
            )
        ),
        "conversations": stats,
        "exclusions": dict(exclusions),
        "recipe": recipe,
        "manifest_sha256": file_hash(root / "replay.jsonl"),
        "teacher_or_tts_used": False,
        "limitations": "Strict exact-timestamp/subword filtering biases selection toward slower speech. Human-human targets test conversational modeling, not tool grounding or instruction following.",
    }
    atomic_json(root / "prepared.json", report)
    if any(report["selected"].get(split, 0) < 4 for split in ("train", "dev", "test")):
        raise ValueError(
            "Too few valid windows; inspect exclusions rather than substituting fixtures"
        )
    return report
