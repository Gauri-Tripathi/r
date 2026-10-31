"""Recorded-audio inference with a validated adapter and an answer-time silence tail."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import click
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .budget import Deadline
from .checkpoint import adapter_metadata
from .config import TrainConfig
from .inference import generate
from .io import atomic_json, file_hash, read_json, redact
from .model import PIN, attach, load_adapter, load_real
from .openyap import CONVERSATION_SYSTEM
from .schema import ModelInput


def prepare_input(source: Path, output: Path, response_s: float) -> dict[str, Any]:
    info = sf.info(source)
    if (
        not 0 < info.duration <= 60
        or info.channels not in (1, 2)
        or not 8000 <= info.samplerate <= 96000
    ):
        raise ValueError("Input must be mono/stereo audio, 8–96 kHz, at most 60 seconds")
    if not math.isfinite(response_s) or not 0 <= response_s <= 60:
        raise ValueError("Response tail must be 0–60 seconds")
    audio, sr = sf.read(source, dtype="float32", always_2d=True)
    if not np.isfinite(audio).all():
        raise ValueError("Input contains nonfinite samples")
    mono = audio.mean(axis=1)
    divisor = math.gcd(sr, 24000)
    mono = resample_poly(mono, 24000 // divisor, sr // divisor).astype(np.float32)
    peak = float(np.max(np.abs(mono)))
    gain = min(1.0, 0.95 / max(peak, 1e-9))
    input_samples = len(mono)
    length = math.ceil((input_samples + round(response_s * 24000)) / 1920) * 1920
    user = np.zeros(length, dtype=np.float32)
    user[:input_samples] = mono * gain
    path = output / "input/user.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, user, 24000, subtype="FLOAT")
    return {
        "source_sha256": file_hash(source),
        "original_input_s": info.duration,
        "response_tail_s": response_s,
        "timeline_s": length / 24000,
        "input_peak_gain": gain,
        "prepared_user_sha256": file_hash(path),
    }


def resolve_adapter(path: Path) -> tuple[Path, dict[str, Any]]:
    if path.is_file():
        raise ValueError(
            "Pass the export/checkpoint directory, not the safetensors file; exact LoRA configuration is required"
        )
    if (path / "best.json").exists() and not (path / "adapter.safetensors").exists():
        selected = read_json(path / "best.json")["checkpoint"]
        candidate = path / "checkpoints" / selected
        if candidate.resolve().parent != (path / "checkpoints").resolve():
            raise ValueError("Invalid checkpoint pointer")
        path = candidate
    metadata = adapter_metadata(path)
    if not metadata.get("real_checkpoint") or metadata["identity"]["base_revision"] != PIN:
        raise ValueError("Adapter must be a real checkpoint for the pinned PersonaPlex base")
    TrainConfig.model_validate(metadata["train_config"])
    return path, metadata


def infer_file(
    assets: Path,
    adapter: Path | None,
    source: Path,
    output: Path,
    response_s: float = 12,
    seed: int = 101,
) -> dict[str, Any]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            "Output directory is nonempty; choose a fresh name to preserve earlier inference"
        )
    metadata = None
    if adapter is not None:
        adapter, metadata = resolve_adapter(adapter)
    prepared = prepare_input(source, output, response_s)
    deadline = Deadline.from_env(reserve_s=60) if os.environ.get("DUPLEX_MODAL") == "1" else None
    if deadline is not None and not deadline.can_step(120 + prepared["timeline_s"] * 4):
        raise ValueError(
            "Not enough reservation for loading and inference; use a fresh budgeted shell"
        )
    started = time.perf_counter()
    lm, mimi, tok = load_real(assets)
    if deadline is not None and not deadline.can_step(prepared["timeline_s"] * 4):
        raise ValueError("Model loading consumed the remaining reservation; no generation started")
    adapter_hash = None
    prompt = CONVERSATION_SYSTEM
    if adapter is not None and metadata is not None:
        attach(lm, TrainConfig.model_validate(metadata["train_config"]))
        load_adapter(lm, adapter / "adapter.safetensors")
        adapter_hash = file_hash(adapter / "adapter.safetensors")
        prompt = metadata["identity"].get("system_prompt", CONVERSATION_SYSTEM)
    visible = ModelInput(
        request_id="user-inference",
        user_audio="input/user.wav",
        events=[],
        duration_s=prepared["timeline_s"],
    )
    generated = generate(
        lm,
        mimi,
        tok,
        visible,
        output,
        output,
        seed=seed,
        adapter_sha256=adapter_hash,
        system_prompt=prompt,
    )
    result = {
        **generated,
        **prepared,
        "total_with_loading_s": time.perf_counter() - started,
        "adapter_directory": str(adapter) if adapter is not None else None,
        "teacher_used": False,
        "mode": "recorded WAV through causal duplex runtime; silence tail permits a response",
    }
    atomic_json(output / "inference.json", result)
    return {
        "audio": str(output / "generated.wav"),
        "metadata": str(output / "inference.json"),
        "internal_generated_text_not_asr": generated["internal_text_proxy"],
        "generation_s": generated["generation_s"],
        "total_with_loading_s": result["total_with_loading_s"],
        "adapter_sha256": adapter_hash,
    }


@click.command("infer")
@click.option("--assets", type=click.Path(path_type=Path, exists=True), default="/cache/assets")
@click.option("--adapter", type=click.Path(path_type=Path))
@click.option(
    "--base-model", is_flag=True, help="Explicitly run without an adapter for comparison."
)
@click.option("--input", "source", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--out", type=click.Path(path_type=Path), required=True)
@click.option("--response-seconds", type=click.FloatRange(0, 60), default=12, show_default=True)
@click.option("--seed", type=int, default=101, show_default=True)
def infer_command(
    assets: Path,
    adapter: Path | None,
    base_model: bool,
    source: Path,
    out: Path,
    response_seconds: float,
    seed: int,
) -> None:
    """Generate assistant speech from your recorded WAV using PersonaPlex + LoRA."""
    if base_model and adapter is not None:
        raise click.UsageError("Choose --adapter or --base-model")
    if not base_model and adapter is None:
        adapter = Path("/runs/openyap/full-v1/export")
    click.echo(
        json.dumps(
            redact(infer_file(assets, adapter, source, out, response_seconds, seed)), indent=2
        )
    )
