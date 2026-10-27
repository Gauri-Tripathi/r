"""Bounded licensed ordinary-conversation replay; no grounding labels inferred from OpenYap."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import Field

from .audio import load_wav, validate_alignment
from .codec import CodecMeasurements, align_text, causal_align_text, encode_channel
from .io import atomic_json, confined, digest, file_hash, read_jsonl
from .model import PIN, native
from .runtime import SYSTEM
from .schema import Strict, Word


class ReplayRecord(Strict):
    id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")
    user: str
    assistant: str
    words: list[Word] = Field(max_length=600)
    source: str = Field(min_length=1, max_length=200)
    license: str = Field(min_length=1, max_length=200)
    isolated_channels_verified: bool
    duration_s: float = Field(gt=0, le=60)
    split: Literal["train", "dev", "test"] = "train"
    conversation_id: str | None = None
    alignment_mode: Literal["strict", "causal_queue"] = "strict"
    partial_text_regions: list[tuple[float, float]] = Field(default_factory=list)


def encode_replay(
    manifest: Path,
    source_root: Path,
    out: Path,
    mimi: Any,
    tokenizer: Any,
    cache: Path,
    device: str,
    max_audio_s: float = 180,
    system_prompt: str = SYSTEM,
    chunk_frames: int = 1,
) -> dict[str, Any]:
    records = [ReplayRecord.model_validate(row) for row in read_jsonl(manifest)]
    if (
        len({r.id for r in records}) != len(records)
        or sum(r.duration_s for r in records) > max_audio_s
    ):
        raise ValueError("Replay IDs duplicate or duration exceeds explicit subset cap")
    native()
    measurements = CodecMeasurements()
    from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS

    for row in records:
        if not row.isolated_channels_verified:
            raise ValueError(
                "Verify isolated a/b channel assignment; mixed preview audio is excluded"
            )
        user_path, assistant_path = (
            confined(source_root, row.user),
            confined(source_root, row.assistant),
        )
        u, a = load_wav(user_path, max_s=60), load_wav(assistant_path, max_s=60)
        if len(u) != len(a) or abs(len(u) / 24000 - row.duration_s) > 0.08:
            raise ValueError("Replay channels do not share the declared timeline")
        if row.alignment_mode == "strict":
            validate_alignment(row.words, row.duration_s)
        user = encode_channel(mimi, user_path, "user", cache, device, measurements, chunk_frames)
        assistant = encode_channel(
            mimi, assistant_path, "assistant", cache, device, measurements, chunk_frames
        )
        alignment: dict[str, float] = {}
        if row.alignment_mode == "causal_queue":
            text, alignment = causal_align_text(row.words, user.shape[-1], tokenizer)
        else:
            text = align_text(row.words, user.shape[-1], tokenizer)
        codes = torch.cat([text, assistant, user], dim=1)
        valid = torch.ones_like(codes, dtype=torch.bool)
        valid[:, 9:] = False
        for start, end in row.partial_text_regions:
            if not 0 <= start <= end <= row.duration_s:
                raise ValueError("Invalid partial-word text mask")
            valid[:, 0, int(start * 12.5) : math.ceil(end * 12.5)] = False
        prompt_ids = tokenizer.encode(f"<system> {system_prompt} <system>")
        length = 12 + len(prompt_ids)
        prompt = torch.cat(
            [
                torch.full((1, 1, length), 3, dtype=torch.long),
                torch.as_tensor(SILENCE_TOKENS).view(1, 8, 1).expand(1, 8, length),
                torch.as_tensor(SINE_TOKENS).view(1, 8, 1).expand(1, 8, length),
            ],
            dim=1,
        )
        prompt[:, 0, 6 : 6 + len(prompt_ids)] = torch.tensor(prompt_ids)
        all_codes = torch.cat([prompt, codes], dim=-1)
        from safetensors.torch import save_file

        out.mkdir(parents=True, exist_ok=True)
        path = out / f"replay-{row.id}.safetensors"
        tmp = path.with_suffix(".tmp")
        save_file(
            {
                "codes": all_codes,
                "valid": torch.cat([torch.zeros_like(prompt, dtype=torch.bool), valid], dim=-1),
                "weights": torch.ones_like(all_codes, dtype=torch.float32),
                "post": torch.zeros(all_codes.shape[-1], dtype=torch.bool),
                "prompt_frames": torch.tensor(length),
            },
            str(tmp),
        )
        tmp.replace(path)
        atomic_json(
            path.with_suffix(".json"),
            {
                "id": row.id,
                "split": row.split,
                "group_id": row.conversation_id,
                "system_prompt": system_prompt,
                "alignment": alignment,
                "chunk_frames": chunk_frames,
                "replay": True,
                "source": row.source,
                "license": row.license,
                "sha256": file_hash(path),
                "codec_revision": PIN,
                "frames": all_codes.shape[-1],
                "audio_frames": codes.shape[-1],
                "fingerprint": digest(row.model_dump()),
            },
        )
    result = {
        "selected": len(records),
        "audio_s": sum(r.duration_s for r in records),
        "max_audio_s": max_audio_s,
        "codec": measurements.report(),
        "exclusions": "Mixed/unverified channels, missing rights, oversized/mismatched timelines fail closed",
    }
    atomic_json(out / "replay-manifest.json", result)
    return result


def mix_frames(paths: list[Path], fraction: float) -> tuple[list[Path], dict[str, Any]]:
    from .io import read_json

    task = [p for p in paths if not read_json(p.with_suffix(".json")).get("replay")]
    ordinary = [p for p in paths if read_json(p.with_suffix(".json")).get("replay")]
    if not task:
        raise ValueError("No scheduling examples; replay cannot replace the experiment")
    if not ordinary:
        return task, {"replay_fraction": 0.0, "reason": "No validated licensed replay supplied"}
    target_frames = sum(read_json(p.with_suffix(".json"))["audio_frames"] for p in task)
    target_replay = int(target_frames * fraction / (1 - fraction))
    mixed = task.copy()
    used = 0
    index = 0
    while used < target_replay:
        path = ordinary[index % len(ordinary)]
        frames = read_json(path.with_suffix(".json"))["audio_frames"]
        if used + frames > target_replay * 1.3:
            break
        mixed.insert(
            min(len(mixed), (index + 1) * max(1, len(task) // max(1, len(ordinary)))), path
        )
        used += frames
        index += 1
    actual = used / (used + target_frames)
    if not 0.1 <= actual <= 0.2:
        raise ValueError(
            "Replay subset cannot achieve 10-20% frames; provide shorter replay segments"
        )
    return mixed, {"replay_fraction": actual, "replay_frames": used, "task_frames": target_frames}
