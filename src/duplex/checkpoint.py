"""Atomic checkpoints and adapter exports."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .io import atomic_json, file_hash, read_json
from .model import adapter_state, load_adapter


def complete(path: Path) -> dict[str, Any]:
    manifest = read_json(path / "complete.json")
    if set(manifest["files"]) != {"adapter.safetensors", "state.pt", "metadata.json"}:
        raise ValueError("Unexpected checkpoint file set")
    for name, checksum in manifest["files"].items():
        if file_hash(path / name) != checksum:
            raise ValueError("Checkpoint checksum mismatch")
    return read_json(path / "metadata.json")


def adapter_metadata(path: Path) -> dict[str, Any]:
    """Load either a complete training checkpoint or a validated local export."""
    if (path / "complete.json").exists():
        return complete(path)
    manifest = read_json(path / "release.json")
    expected = {
        "adapter.safetensors",
        "adapter.json",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "pyproject.toml",
        "requirements-modal.txt",
        "upstream.lock.json",
        "runtime-requirements.json",
    }
    if set(manifest["files"]) != expected:
        raise ValueError("Unexpected adapter export file set")
    if any(file_hash(path / name) != checksum for name, checksum in manifest["files"].items()):
        raise ValueError("Adapter export checksum mismatch")
    return read_json(path / "adapter.json")


def save(
    root: Path,
    lm: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    sampler: int,
) -> Path:
    step = metadata["step"]
    root.mkdir(parents=True, exist_ok=True)
    dest = root / f"step-{step:06d}"
    if dest.exists():
        if complete(dest) != metadata:
            raise ValueError("Checkpoint step already exists with different metadata")
        existing = load_file(str(dest / "adapter.safetensors"))
        current = adapter_state(lm)
        if set(existing) != set(current) or any(
            not torch.equal(existing[n], t) for n, t in current.items()
        ):
            raise ValueError("Refuse to reuse a completed step with different adapter weights")
        return dest
    staging = root / f".step-{step:06d}-{os.getpid()}.incomplete"
    staging.mkdir(exist_ok=False)
    save_file(adapter_state(lm), str(staging / "adapter.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "sampler": sampler,
        },
        staging / "state.pt",
    )
    for tensor_file in (staging / "adapter.safetensors", staging / "state.pt"):
        with tensor_file.open("r+b") as stream:
            os.fsync(stream.fileno())
    atomic_json(staging / "metadata.json", metadata)
    atomic_json(
        staging / "complete.json",
        {
            "files": {
                name: file_hash(staging / name)
                for name in ("adapter.safetensors", "state.pt", "metadata.json")
            }
        },
    )
    staging.replace(dest)
    return dest


def restore(
    path: Path, lm: torch.nn.Module, optimizer: torch.optim.Optimizer, identity: dict[str, Any]
) -> tuple[int, int]:
    metadata = complete(path)
    if metadata["identity"] != identity:
        raise ValueError("Incompatible resume: architecture/config/data/revisions differ")
    load_adapter(lm, path / "adapter.safetensors")
    # Only locally created, checksum-validated resume state. Never accept external pickle.
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return metadata["step"], state["sampler"]


def retain(root: Path, best: Path | None = None) -> None:
    checkpoints = sorted(p for p in root.glob("step-*") if (p / "complete.json").exists())
    for path in checkpoints[:-2]:
        if best is not None and path.resolve() == best.resolve():
            continue
        # Delete only validated direct children of the explicit checkpoints directory.
        if path.resolve().parent != root.resolve() or path.is_symlink():
            raise ValueError("Unsafe checkpoint retention path")
        complete(path)
        shutil.rmtree(path)


def export(checkpoint: Path, dest: Path, notices: Path) -> None:
    metadata = complete(checkpoint)
    metadata = {
        **metadata,
        "experiment_kind": metadata.get("experiment_kind", "smoke_test"),
        "capability_improvement": "unverified",
    }
    dest.mkdir(parents=True, exist_ok=False)
    state = load_file(str(checkpoint / "adapter.safetensors"))
    save_file(state, str(dest / "adapter.safetensors"))
    atomic_json(dest / "adapter.json", metadata)
    shutil.copyfile(notices, dest / "THIRD_PARTY_NOTICES.md")
    repository = Path(__file__).resolve().parents[2]
    required = ("pyproject.toml", "requirements-modal.txt", "upstream.lock.json")
    for name in required:
        shutil.copyfile(repository / name, dest / name)
    atomic_json(
        dest / "runtime-requirements.json",
        {
            "runtime_patch": metadata["identity"].get("runtime_patch"),
            "source_sha256": {
                str(p.relative_to(repository)): file_hash(p)
                for p in sorted((repository / "src" / "duplex").glob("*.py"))
            },
            "dependency_files": {name: file_hash(repository / name) for name in required},
            "execution": "Linux, Python 3.11, pinned vendored Moshi and PersonaPlex assets; no teacher service at inference",
            "historical_runtime_hash_verified": "runtime_source_sha256" in metadata["identity"],
            "limitations": "Source hashes describe this export runtime. Older smoke checkpoints did not record exact executed wrapper source; do not infer historical source equality.",
        },
    )
    conversational = metadata["experiment_kind"].startswith("conversation_")
    title = (
        "OpenYap conversational feasibility adapter"
        if conversational
        else "Appointment correction/grounding adapter"
    )
    data_scope = (
        "English recorded human conversations; machine transcripts. No tool-grounding claim.\n"
        if conversational
        else "Context forces silence and costs frames. English synthetic scheduling only.\n"
    )
    (dest / "README.md").write_text(
        f"# {title}\n\nStatus: {metadata['experiment_kind']}; capability improvement unverified.\nExperimental sequence SFT LoRA for pinned PersonaPlex.\n"
        "No guaranteed improvement. See adapter.json for exact modules, revisions and data fingerprints.\n"
        "Use the identical duplex runtime for base/adapted evaluation. No text teacher at inference.\n"
        + data_scope
        + "GPU compatibility, audible quality, held-out improvement and redistribution rights require separate verification.\n"
        "Base: NVIDIA Open Model License, with Moshi attribution. Do not upload without checking voice/data terms.\n\n"
        "Loading in the repo: load_real(assets); attach(lm, TrainConfig.model_validate(metadata['train_config'])); load_adapter(lm, export/'adapter.safetensors').\n",
        encoding="utf-8",
    )
    atomic_json(
        dest / "release.json",
        {"files": {p.name: file_hash(p) for p in dest.iterdir() if p.is_file()}},
    )
