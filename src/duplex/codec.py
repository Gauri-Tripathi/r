"""Session-isolated causal Mimi codes and native text/audio supervision layout."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from .audio import PREPROCESS_VERSION, SAMPLE_RATE, load_wav, validate_alignment
from .io import atomic_json, confined, digest, file_hash, read_json
from .model import PIN, native
from .runtime import SYSTEM, schedule
from .schema import Example, Word


@dataclass
class CodecMeasurements:
    cache_hits: int = 0
    cache_misses: int = 0
    cached_audio_s: float = 0
    fresh_audio_s: float = 0
    fresh_wall_s: float = 0

    def report(self) -> dict[str, Any]:
        return {
            **vars(self),
            "cold_audio_s_per_s": self.fresh_audio_s / self.fresh_wall_s
            if self.fresh_wall_s
            else None,
        }


def encode_channel(
    mimi: Any,
    wav: Path,
    assignment: str,
    cache: Path,
    device: str,
    measurements: CodecMeasurements | None = None,
    chunk_frames: int = 1,
) -> torch.Tensor:
    if not 1 <= chunk_frames <= 100:
        raise ValueError("Codec chunk must be 1..100 causal frames")
    settings = {
        "audio": file_hash(wav),
        "channel": assignment,
        "sample_rate": SAMPLE_RATE,
        "codec_revision": PIN,
        "n_codebooks": mimi.num_codebooks,
        "frame_rate": mimi.frame_rate,
        "preprocessing": PREPROCESS_VERSION,
        "mode": f"causal-stream-{1920 * chunk_frames}-v1",
    }
    key = digest(settings)
    dest = cache / f"{key}.safetensors"
    marker = cache / f"{key}.json"
    if dest.exists() and marker.exists():
        if read_json(marker)["sha256"] != file_hash(dest):
            raise ValueError("Corrupt codec cache")
        cached = load_file(str(dest))["codes"]
        if measurements:
            measurements.cache_hits += 1
            measurements.cached_audio_s += cached.shape[-1] / mimi.frame_rate
        return cached
    started = time.perf_counter()
    audio = load_wav(wav)
    if len(audio) % 1920:
        audio = np.pad(audio, (0, 1920 - len(audio) % 1920))
    pieces = []
    mimi.eval()
    # Avoid repeated NNPACK initialization failures on unsupported CPUs.
    cpu_backend = (
        torch.backends.nnpack.flags(enabled=False)
        if torch.device(device).type == "cpu"
        else nullcontext()
    )
    # Reset all encoder/cache state between channels/examples, preserve within-session causality.
    with cpu_backend, torch.no_grad(), mimi.streaming(1):
        for offset in range(0, len(audio), 1920 * chunk_frames):
            frame = audio[offset : offset + 1920 * chunk_frames]
            piece = mimi.encode(torch.from_numpy(frame.copy()).to(device).view(1, 1, -1))
            pieces.append(piece.cpu())
    codes = torch.cat(pieces, dim=-1)
    if codes.shape != (1, 8, len(audio) // 1920) or codes.min() < 0 or codes.max() >= 2048:
        raise ValueError("Codec IDs, frame count, or codebook count mismatch")
    cache.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    save_file({"codes": codes.contiguous()}, str(tmp))
    tmp.replace(dest)
    atomic_json(marker, {"settings": settings, "sha256": file_hash(dest)})
    if measurements:
        measurements.cache_misses += 1
        measurements.fresh_audio_s += codes.shape[-1] / mimi.frame_rate
        measurements.fresh_wall_s += time.perf_counter() - started
    return codes


def align_text(words: list[Word], frames: int, tokenizer: Any) -> torch.Tensor:
    """Native one-piece-per-frame convention from audited interleaver.

    Words start at measured boundaries. Subword tokens are contiguous, not
    uniformly stretched across speech. Reject collisions instead of dropping words.
    """
    tokens = torch.full((1, 1, frames), 3, dtype=torch.long)
    cursor = 0
    for word in words:
        encoded = tokenizer.encode(word.text)
        start = math.floor(word.start_s * 12.5)
        if start < cursor or start + len(encoded) > frames:
            raise ValueError(
                "Alignment/subword collision: slow down speech or supply revised checked alignment"
            )
        if start > 0 and tokens[0, 0, start - 1] == 3:
            tokens[0, 0, start - 1] = 0  # native end_of_text_padding_id
        tokens[0, 0, start : start + len(encoded)] = torch.tensor(encoded)
        cursor = start + len(encoded)
    return tokens


def causal_align_text(
    words: list[Word], frames: int, tokenizer: Any, max_delay_s: float = 0.24
) -> tuple[torch.Tensor, dict[str, float]]:
    """Place word pieces on the frame grid and record serialization delay."""
    tokens = torch.full((1, 1, frames), 3, dtype=torch.long)
    cursor, previous = 0, -1.0
    maximum = 0.0
    for word in words:
        if word.start_s < previous or word.end_s > frames / 12.5:
            raise ValueError("Invalid source word ordering/bounds")
        previous = word.start_s
        encoded = tokenizer.encode(word.text)
        requested = math.floor(word.start_s * 12.5)
        start = max(requested, cursor)
        delay = (start - requested) / 12.5
        if delay > max_delay_s + 1e-8 or start + len(encoded) > frames:
            raise ValueError("Native text serialization exceeds bounded delay/window")
        maximum = max(maximum, delay)
        if start > 0 and tokens[0, 0, start - 1] == 3:
            tokens[0, 0, start - 1] = 0
        tokens[0, 0, start : start + len(encoded)] = torch.tensor(encoded)
        cursor = start + len(encoded)
    return tokens, {"max_piece_queue_delay_s": maximum}


def build_sample(
    row: Example,
    root: Path,
    mimi: Any,
    tokenizer: Any,
    cache: Path,
    device: str,
    voice_codes: torch.Tensor | None = None,
    measurements: CodecMeasurements | None = None,
) -> dict[str, torch.Tensor]:
    if row.visible.user_audio is None:
        raise ValueError("Render/import real audio before codec encoding")
    metadata = read_json(root / "audio" / row.id / "manifest.json")
    if metadata["backend"] == "tone":
        raise ValueError("Tone fixtures cannot enter real model training/evaluation")
    words = [Word.model_validate(w) for w in metadata["words"]]
    validate_alignment(words, row.visible.duration_s)
    user = encode_channel(
        mimi, confined(root, row.visible.user_audio), "user", cache, device, measurements
    )
    assistant = encode_channel(
        mimi, confined(root, metadata["assistant"]), "assistant", cache, device, measurements
    )
    if user.shape != assistant.shape:
        raise ValueError("Channel timelines differ")
    frames = user.shape[-1]
    text = align_text(words, frames, tokenizer)
    context, _ = schedule(row.visible.events, frames, 12.5, row.visible.request_id, tokenizer)
    silence = encode_channel(mimi, _silence_wav(cache), "assistant-silence", cache, device)[..., :1]
    for i, token in enumerate(context):
        if token is not None:
            if any(w.start_s < (i + 1) / 12.5 and w.end_s > i / 12.5 for w in words):
                raise ValueError("Assistant supervision overlaps forced context silence")
            text[0, 0, i] = token
            assistant[..., i : i + 1] = silence
    codes = torch.cat([text, assistant, user], dim=1)
    valid = torch.zeros_like(codes, dtype=torch.bool)
    valid[:, :9] = True
    weights = torch.ones_like(codes, dtype=torch.float32)
    for i, token in enumerate(context):
        if token is not None:
            valid[..., i] = False  # context, forced silence, user all excluded from target loss
    post = math.ceil(metadata["response_start_s"] * 12.5)
    response_region = torch.zeros(frames, dtype=torch.bool)
    response_region[post:] = True
    # Match native role prompt order: licensed voice -> silence -> text -> silence.
    prompt_tokens = tokenizer.encode(f"<system> {SYSTEM} <system>")
    voice_frames = voice_codes.shape[-1] if voice_codes is not None else 0
    length = voice_frames + 6 + len(prompt_tokens) + 6
    native()
    from moshi.models.lm import SILENCE_TOKENS, SINE_TOKENS

    prompt_silence = torch.as_tensor(SILENCE_TOKENS).view(1, 8, 1)
    prompt_sine = torch.as_tensor(SINE_TOKENS).view(1, 8, 1)
    prompt = torch.cat(
        [
            torch.full((1, 1, length), 3, dtype=torch.long),
            prompt_silence.expand(1, 8, length),
            prompt_sine.expand(1, 8, length),
        ],
        dim=1,
    )
    if voice_codes is not None:
        prompt[:, 1:9, :voice_frames] = voice_codes
    prompt[:, 0, voice_frames + 6 : voice_frames + 6 + len(prompt_tokens)] = torch.tensor(
        prompt_tokens
    )
    return {
        "codes": torch.cat([prompt, codes], dim=-1),
        "valid": torch.cat([torch.zeros_like(prompt, dtype=torch.bool), valid], dim=-1),
        "weights": torch.cat([torch.ones_like(prompt, dtype=torch.float32), weights], dim=-1),
        "post": torch.cat([torch.zeros(length, dtype=torch.bool), response_region]),
        "prompt_frames": torch.tensor(length),
    }


def _silence_wav(cache: Path) -> Path:
    import soundfile as sf

    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "silence.wav"
    if not path.exists():
        sf.write(path, np.zeros(1920 * 8, dtype=np.float32), SAMPLE_RATE, subtype="FLOAT")
    return path


def save_sample(path: Path, sample: dict[str, torch.Tensor], row: Example) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.contiguous() for k, v in sample.items()}, str(tmp))
    tmp.replace(path)
    atomic_json(
        path.with_suffix(".json"),
        {
            "id": row.id,
            "split": row.split,
            "group_id": row.group_id,
            "supervision_provenance": row.provenance,
            "fingerprint": digest(row.model_dump()),
            "sha256": file_hash(path),
            "codec_revision": PIN,
            "channels": "text,assistant[8],user[8]",
            "frames": sample["codes"].shape[-1],
            "audio_frames": sample["codes"].shape[-1] - int(sample["prompt_frames"]),
        },
    )
