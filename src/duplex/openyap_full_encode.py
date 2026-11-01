"""Resumable Mimi encoding with causal chunk verification."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audio import load_wav
from .budget import Deadline
from .io import atomic_json, digest, file_hash, read_json, read_jsonl, write_jsonl
from .model import FILES, PIN, native, tokenizer, verify_assets
from .openyap import CONVERSATION_SYSTEM
from .replay import ReplayRecord, encode_replay


def verify_chunks(mimi: Any, wav: Path, chunk_frames: int, device: str) -> dict[str, Any]:
    audio = load_wav(wav)
    n = 1920 * 50
    if len(audio) < n:
        raise ValueError("Need four real audio seconds to verify causal encoding")
    offsets = list(range(0, len(audio) - n + 1, n))
    offset = max(offsets, key=lambda i: float(np.mean(audio[i : i + n] ** 2)))
    audio = audio[offset : offset + n].copy()

    def codes(pcm: np.ndarray, chunk: int) -> torch.Tensor:
        pieces = []
        backend = torch.backends.nnpack.flags(enabled=False) if device == "cpu" else nullcontext()
        with backend, torch.no_grad(), mimi.streaming(1):
            for index in range(0, len(pcm), 1920 * chunk):
                pieces.append(
                    mimi.encode(
                        torch.from_numpy(pcm[index : index + 1920 * chunk].copy())
                        .to(device)
                        .reshape(1, 1, -1)
                    ).cpu()
                )
        return torch.cat(pieces, dim=-1)

    started = time.perf_counter()
    reference = codes(audio, 1)
    reference_s = time.perf_counter() - started
    started = time.perf_counter()
    batched = codes(audio, chunk_frames)
    batched_s = time.perf_counter() - started
    altered = audio.copy()
    cut = 1920 * 12
    altered[cut:] = np.random.default_rng(101).uniform(-0.1, 0.1, n - cut).astype(np.float32)
    counterfactual = codes(altered, chunk_frames)
    result = {
        "chunk_frames": chunk_frames,
        "device": device,
        "codec_revision": PIN,
        "source_wav_sha256": file_hash(wav),
        "reference_s": reference_s,
        "batched_s": batched_s,
        "codes_identical": torch.equal(reference, batched),
        "future_audio_does_not_change_prefix": torch.equal(
            batched[..., :12], counterfactual[..., :12]
        ),
    }
    if not result["codes_identical"] or not result["future_audio_does_not_change_prefix"]:
        raise ValueError(f"Causal chunk verification failed: {result}; use --chunk-frames 1")
    return result


def full_encode(
    root: Path,
    assets: Path,
    device: str = "cpu",
    chunk_frames: int = 25,
    max_wall_s: int = 2400,
    limit: int | None = None,
) -> dict[str, Any]:
    verify_assets(assets, include_base=False)
    prepared = read_json(root / "prepared.json")
    if prepared["kind"] != "openyap_full" or prepared["manifest_sha256"] != file_hash(
        root / "replay.jsonl"
    ):
        raise ValueError("Need completed full-data preparation with an unchanged manifest")
    rows = read_jsonl(root / "replay.jsonl")
    out = root / "encoded"
    out.mkdir(parents=True, exist_ok=True)
    completed = {}
    pending = []
    for row in rows:
        path = out / f"replay-{row['id']}.safetensors"
        if path.exists() and path.with_suffix(".json").exists():
            meta = read_json(path.with_suffix(".json"))
            if (
                meta["sha256"] != file_hash(path)
                or meta["fingerprint"] != digest(ReplayRecord.model_validate(row).model_dump())
                or meta["codec_revision"] != PIN
            ):
                raise ValueError("Existing encoded sample changed; do not overwrite it")
            completed[path.name] = meta["sha256"]
        else:
            pending.append(row)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no CPU fallback")
    import os

    deadline = (
        Deadline.from_env(reserve_s=180)
        if os.environ.get("DUPLEX_MODAL") == "1"
        else Deadline(time.time() + max_wall_s, reserve_s=30)
    )
    started = time.perf_counter()
    timings: list[float] = []
    verification: dict[str, Any] = {}
    if pending:
        torch.set_num_threads(2)
        loaders, _, _ = native()
        mimi = loaders.get_mimi(assets / FILES["codec"], device=device).requires_grad_(False)
        tok = tokenizer(assets)
        if chunk_frames != 1:
            verification = verify_chunks(mimi, root / pending[0]["user"], chunk_frames, device)
            atomic_json(out / "chunk-verification.json", verification)
        for row in pending:
            if (
                (limit is not None and len(timings) >= limit)
                or time.perf_counter() - started >= max_wall_s
                or not deadline.can_step(max(30, max(timings, default=0) * 1.5))
            ):
                break
            timeline = read_json(root / "audio" / row["id"] / "manifest.json")
            if any(
                file_hash(root / row[c]) != timeline[c + "_sha256"] for c in ("user", "assistant")
            ):
                raise ValueError("Prepared source waveform changed")
            write_jsonl(out / "selection-current.jsonl", [row])
            before = time.perf_counter()
            encode_replay(
                out / "selection-current.jsonl",
                root,
                out,
                mimi,
                tok,
                root / "mimi-cache",
                "cuda" if device == "cuda" else "cpu",
                row["duration_s"],
                system_prompt=CONVERSATION_SYSTEM,
                chunk_frames=chunk_frames,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - before)
            path = out / f"replay-{row['id']}.safetensors"
            completed[path.name] = file_hash(path)
            print(f"Encoded {len(completed)}/{len(rows)}: {timings[-1]:.2f}s", flush=True)
            atomic_json(
                out / "progress.json",
                {"completed": len(completed), "total": len(rows), "last_sample_s": timings[-1]},
            )
    previous = read_json(out / "encoding.json") if (out / "encoding.json").exists() else {}
    done = len(completed) == len(rows)
    result = {
        "kind": "openyap_full_encoded" if done else "openyap_full_encoding_partial",
        "source_manifest_sha256": prepared["manifest_sha256"],
        "completed": len(completed),
        "total": len(rows),
        "encoded": completed,
        "chunk_frames": chunk_frames,
        "device": device,
        "wall_s": time.perf_counter() - started,
        "cumulative_wall_s": previous.get("cumulative_wall_s", 0) + time.perf_counter() - started,
        "mean_window_s": sum(timings) / len(timings) if timings else previous.get("mean_window_s"),
        "verification": verification,
        "complete": done,
    }
    atomic_json(out / "encoding.json", result)
    return result
