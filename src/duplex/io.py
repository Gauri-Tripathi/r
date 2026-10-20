"""Bounded input, atomic manifests, content addressing, and safe logging."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

MAX_FILE = 32 * 1024 * 1024


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def read_json(path: Path, limit: int = MAX_FILE) -> Any:
    if path.stat().st_size > limit:
        raise ValueError(f"Input exceeds {limit} bytes: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> Any:
    if path.stat().st_size > 65536:
        raise ValueError("Configuration too large")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.stat().st_size > MAX_FILE:
        raise ValueError("JSONL exceeds size limit; shard it")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if len(line) > 262144:
                raise ValueError("Oversized JSONL row")
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def confined(root: Path, relative: str) -> Path:
    if "\\" in relative or ":" in relative or Path(relative).is_absolute():
        raise ValueError("Expected portable relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path escapes dataset root")
    return path


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (
                "[REDACTED]"
                if re.fullmatch(
                    r"(?i)(?:.*_)?(?:api_key|secret|password|authorization|access_token|refresh_token|hf_token|token)",
                    k,
                )
                else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r"https?://[^\s\"<>]+", "[URL]", value)
        value = re.sub(r"\b(?:hf_|sk-|wandb_)[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        for name, secret in os.environ.items():
            if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY", name) and len(secret) >= 6:
                value = value.replace(secret, "[REDACTED]")
    return value
