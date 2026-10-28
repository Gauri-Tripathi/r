"""Local allowlisted metrics always; bounded W&B in a separate worker process."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Any, Literal, cast

from .io import atomic_json, read_json, redact

ALLOWED = {
    "step",
    "text_loss",
    "audio_loss",
    "valid_audio_tokens",
    "valid_text_tokens",
    "real_text_tokens",
    "loss",
    "lr",
    "grad_norm",
    "trainable",
    "audio_seconds",
    "examples",
    "steps_per_s",
    "peak_memory_bytes",
    "gpu_utilization",
    "stage_s",
    "estimated_spend_usd",
    "deadline_remaining_s",
    "checkpoint_sha256",
    "data_fingerprint",
    "dev_loss",
    "run_id",
    "event",
}


def _worker(
    channel: Any, directory: str, mode: str, run_id: str, project: str | None, entity: str | None
) -> None:
    # W&B never sees the raw config, environment, audio, transcripts, source, or weights.
    cache = Path(directory).resolve() / "private-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.update(
        WANDB_DISABLE_GIT="true",
        WANDB_DISABLE_CODE="true",
        WANDB_CONSOLE="off",
        WANDB_SILENT="true",
        WANDB_CACHE_DIR=str(cache),
        WANDB_CONFIG_DIR=str(cache / "config"),
        WANDB_DATA_DIR=str(cache / "data"),
        XDG_CACHE_HOME=str(cache),
        LOCALAPPDATA=str(cache),
    )
    try:
        import wandb

        settings = wandb.Settings(
            console="off",
            disable_git=True,
            disable_code=True,
            x_disable_meta=True,
            x_disable_stats=True,
            x_disable_machine_info=True,
            init_timeout=8,
            finish_timeout=4,
        )
        run = wandb.init(
            id=run_id,
            project=project or "duplex-local",
            entity=entity,
            mode=cast(Literal["offline", "online", "disabled"], mode),
            dir=directory,
            config={"task": "appointment_sft", "schema": "metrics-v1"},
            settings=settings,
        )
        while True:
            item = channel.get()
            if item is None:
                break
            run.log(item, step=item["step"])
        run.finish()
    except Exception as exc:
        # Isolated optional telemetry boundary: record only exception class, never payload.
        Path(directory, "tracking-error.txt").write_text(type(exc).__name__, encoding="utf-8")


class Tracker:
    def __init__(
        self,
        directory: Path,
        mode: str = "offline",
        project: str | None = None,
        entity: str | None = None,
    ):
        if mode not in ("offline", "online", "disabled"):
            raise ValueError("Invalid tracking mode")
        if mode == "online" and (not project or not entity or not os.environ.get("WANDB_API_KEY")):
            raise ValueError(
                "Online W&B requires explicit project/entity and existing private access settings plus WANDB_API_KEY"
            )
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        identity = directory / "tracking.json"
        import uuid

        self.run_id = read_json(identity)["run_id"] if identity.exists() else uuid.uuid4().hex[:16]
        atomic_json(identity, {"run_id": self.run_id})
        self.last_step = -1
        history = directory / "metrics.jsonl"
        if history.exists():
            with history.open(encoding="utf-8") as stream:
                for line in stream:
                    self.last_step = max(self.last_step, json.loads(line)["step"])
        self.channel: Any = mp.get_context("spawn").Queue(maxsize=64)
        self.process: Any = None
        if mode != "disabled":
            self.process = mp.get_context("spawn").Process(
                target=_worker,
                args=(self.channel, str(directory), mode, self.run_id, project, entity),
                daemon=True,
            )
            self.process.start()

    def log(self, metrics: dict[str, Any]) -> None:
        if set(metrics) - ALLOWED:
            raise ValueError("Tracking fields are not allowlisted")
        if metrics["step"] < self.last_step:
            raise ValueError("Nonmonotonic optimizer step on resume")
        self.last_step = metrics["step"]
        safe = redact(metrics)
        with (self.directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, allow_nan=False) + "\n")
        if self.process and self.process.is_alive():
            try:
                self.channel.put_nowait(safe)
            except queue.Full:
                pass  # optional telemetry is bounded; local metrics are already durable

    def close(self) -> None:
        if self.process:
            try:
                self.channel.put_nowait(None)
            except queue.Full:
                pass
            self.process.join(timeout=6)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
        self.channel.close()
