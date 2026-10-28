"""Verify real W&B offline files with environment sentinels and bounded worker lifetime."""

from __future__ import annotations

import os
import time
from pathlib import Path

from duplex.io import atomic_json
from duplex.tracking import Tracker


def main() -> None:
    parent = Path("runs/wandb-check")
    root = parent / str(time.time_ns())
    sentinel = "duplex-sentinel-must-stay-private-987654321"
    os.environ["TEACHER_API_KEY"] = sentinel
    os.environ["HF_TOKEN"] = sentinel
    tracker = Tracker(root, "offline")
    tracker.log(
        {
            "step": max(1, tracker.last_step + 1),
            "loss": 1.0,
            "valid_text_tokens": 8,
            "event": "offline_verification",
        }
    )
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        if (
            list(root.glob("wandb/offline-run-*/run-*.wandb"))
            or (root / "tracking-error.txt").exists()
        ):
            break
        time.sleep(0.5)
    tracker.close()
    files = list(root.rglob("*"))
    if (root / "tracking-error.txt").exists():
        raise ValueError("W&B worker failed: " + (root / "tracking-error.txt").read_text())
    if not list(root.glob("wandb/offline-run-*/run-*.wandb")):
        raise ValueError("No genuine W&B offline run file produced within bounded time")
    for path in files:
        if path.is_file() and sentinel.encode() in path.read_bytes():
            raise ValueError("Environment sentinel leaked into offline tracking files")
    atomic_json(
        root / "verification.json",
        {
            "wandb_offline_verified": True,
            "environment_sentinels_absent": True,
            "scanned_files": sum(p.is_file() for p in files),
            "no_media_uploaded": True,
        },
    )
    atomic_json(parent / "latest.json", {"verification": str(root / "verification.json")})
    print("Genuine W&B offline files verified; environment sentinels absent.")


if __name__ == "__main__":
    main()
