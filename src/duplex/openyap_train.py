"""Matched base and adapter evaluation for OpenYap training."""

from __future__ import annotations

import math
import random
import shutil
import signal
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from . import checkpoint
from .budget import Deadline, Ledger
from .config import Config, load_config
from .inference import generate
from .io import atomic_json, file_hash, read_json, read_jsonl, write_jsonl
from .model import (
    ROOT,
    adapter_state,
    attach,
    enable_adapters,
    load_adapter,
    load_real,
    native_loss,
)
from .openyap import CONVERSATION_SYSTEM, REVISION
from .replay import encode_replay
from .schema import ModelInput
from .train import _step, batch, commit_volumes, dataset_paths, identity, modality_gradients


def losses(lm: Any, paths: list[Path], cfg: Config, deadline: Deadline) -> list[dict[str, Any]]:
    result = []
    lm.eval()
    with torch.no_grad():
        for path in paths:
            if not deadline.can_step(5):
                raise ValueError("Insufficient evaluation reservation")
            sample = batch(path, str(lm.device), 1)
            _, stats = native_loss(
                lm.forward_train(sample["codes"]),
                sample["codes"],
                sample["valid"],
                sample["weights"],
                cfg.train.text_pad_weight,
            )
            metadata = read_json(path.with_suffix(".json"))
            result.append({"id": metadata["id"], "conversation": metadata["group_id"], **stats})
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: sum(r[key] for r in rows) / len(rows) for key in ("audio_loss", "text_loss")}


def encode_gpu(root: Path, mimi: Any, tok: Any, deadline: Deadline) -> dict[str, Any]:
    """Measure cold GPU encoding first; stop if the bounded pilot cannot fit."""
    rows = read_jsonl(root / "replay.jsonl")
    out = root / "encoded"
    elapsed = 0.0
    cold_per_window = []
    for index, row in enumerate(rows):
        if deadline.remaining() < 900:
            raise ValueError("Encoding cannot consume reserved training/evaluation/shutdown time")
        timeline = read_json(root / "audio" / row["id"] / "manifest.json")
        for channel in ("user", "assistant"):
            if file_hash(root / row[channel]) != timeline[channel + "_sha256"]:
                raise ValueError("Source waveform changed since preparation")
        write_jsonl(out / "selection-current.jsonl", [row])
        started = time.perf_counter()
        measured = encode_replay(
            out / "selection-current.jsonl",
            root,
            out,
            mimi,
            tok,
            root / "mimi-cache",
            "cuda",
            row["duration_s"],
            system_prompt=CONVERSATION_SYSTEM,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        elapsed += seconds
        if measured["codec"]["cache_misses"]:
            cold_per_window.append(seconds)
        if cold_per_window and max(cold_per_window) * (len(rows) - index - 1) > min(
            300, deadline.remaining() - 900
        ):
            raise ValueError(
                "Measured GPU encoding exceeds allocation; partial cache retained, no retry"
            )
        atomic_json(
            out / "encoding-progress.json",
            {
                "completed": index + 1,
                "total": len(rows),
                "wall_s": elapsed,
                "cold_window_s": cold_per_window,
            },
        )
        print(f"Mimi encoding {index + 1}/{len(rows)}: {seconds:.2f}s", flush=True)
    result = {
        "kind": "openyap_encoded_pilot",
        "source_manifest_sha256": file_hash(root / "replay.jsonl"),
        "selected": len(rows),
        "wall_s": elapsed,
        "cold_window_s": cold_per_window,
        "encoded": {p.name: file_hash(p) for p in out.glob("*.safetensors")},
    }
    atomic_json(out / "encoding.json", result)
    return result


def pilot(
    root: Path,
    assets: Path,
    run: Path,
    ledger_path: Path,
    config_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    cfg = load_config(config_path or ROOT / "configs/openyap.yaml")
    full = cfg.experiment_kind == "conversation_full"
    deadline = Deadline.from_env(reserve_s=90)
    ledger = Ledger(ledger_path, cfg.budget)
    if (run / "results.json").exists() or (
        not resume and any((run / "checkpoints").glob("step-*"))
    ):
        raise ValueError(
            "Pilot artifacts already exist; preserve them and inspect before another attempt"
        )
    prepared = read_json(root / "prepared.json")
    expected_kind = "openyap_full" if full else "openyap_conversation_pilot"
    if prepared["kind"] != expected_kind or prepared["manifest_sha256"] != file_hash(
        root / "replay.jsonl"
    ):
        raise ValueError("Prepared real-data manifest changed")
    run.mkdir(parents=True, exist_ok=True)
    if full:
        if (
            not (root / "encoded" / "encoding.json").exists()
            or read_json(root / "encoded" / "encoding.json").get("kind") != "openyap_full_encoded"
        ):
            raise ValueError("Complete/resume full-encode before training")
        if deadline.remaining() < 900:
            raise ValueError(
                "Exit and reconcile this session, then start training in a fresh budgeted shell"
            )
    started = time.perf_counter()
    lm, mimi, tok = load_real(assets)
    load_s = time.perf_counter() - started
    encoded = root / "encoded"
    encoding = (
        read_json(encoded / "encoding.json")
        if (encoded / "encoding.json").exists()
        else encode_gpu(root, mimi, tok, deadline)
    )
    if (
        prepared["kind"] != expected_kind
        or encoding["kind"] != ("openyap_full_encoded" if full else "openyap_encoded_pilot")
        or prepared["manifest_sha256"] != file_hash(root / "replay.jsonl")
        or encoding["source_manifest_sha256"] != prepared["manifest_sha256"]
    ):
        raise ValueError("OpenYap preparation/encoding provenance differs")
    if encoding["encoded"] != {p.name: file_hash(p) for p in encoded.glob("*.safetensors")}:
        raise ValueError("Encoded pilot hashes changed")
    paths = {
        split: dataset_paths(encoded, split, cfg.train.max_frames)
        for split in ("train", "dev", "test")
    }
    if min(map(len, paths.values())) < 4:
        raise ValueError("At least four actual windows per split are required")
    if full:
        if len(paths["train"]) < 64:
            raise ValueError("Full training requires at least 64 actual train windows")
        random.Random(cfg.train.seed).shuffle(paths["train"])
    seen: dict[str, str] = {}
    for split, files in paths.items():
        for path in files:
            meta = read_json(path.with_suffix(".json"))
            if meta["system_prompt"] != CONVERSATION_SYSTEM:
                raise ValueError("Conversation data must use the matched conversation prompt")
            group = meta["group_id"]
            if group in seen and seen[group] != split:
                raise ValueError("Conversation leakage")
            seen[group] = split
    torch.manual_seed(cfg.train.seed)
    modules = attach(lm, cfg.train)
    run_identity = identity(paths["train"], cfg, modules)
    run_identity.update(
        system_prompt=CONVERSATION_SYSTEM,
        source_dataset_revision=REVISION,
        source_dataset_manifest_sha256=prepared["manifest_sha256"],
        planned_epochs=cfg.max_effective_epochs,
    )
    if resume:
        if read_json(run / "run-identity.json") != run_identity:
            raise ValueError("Resume source/config/data identity changed")
        initial = load_file(str(run / "initial-adapter.safetensors"))
    else:
        initial = adapter_state(lm)
        save_file(initial, str(run / "initial-adapter.safetensors"))
        atomic_json(run / "run-identity.json", run_identity)
    enable_adapters(lm, False)
    baseline_started = time.perf_counter()
    if resume and (run / "base-losses.json").exists():
        base_saved = read_json(run / "base-losses.json")
        base_dev, base_test = base_saved["dev"], base_saved["test"]
    else:
        base_dev = losses(lm, paths["dev"], cfg, deadline)
        base_test = losses(lm, paths["test"], cfg, deadline)
        atomic_json(run / "base-losses.json", {"dev": base_dev, "test": base_test})
    baseline_loss_s = time.perf_counter() - baseline_started
    rows = read_jsonl(root / "replay.jsonl")
    generation_rows = [r for r in rows if r["split"] == "test"][:3]
    write_jsonl(run / "locked-generation-inputs.jsonl", generation_rows)

    def replay(destination: str, adapter_hash: str | None) -> list[dict[str, Any]]:
        outputs = []
        for row in generation_rows:
            destination_path = run / destination / row["id"]
            if (destination_path / "generation.json").exists():
                prior = read_json(destination_path / "generation.json")
                if (
                    prior.get("adapter_sha256") != adapter_hash
                    or prior["user_audio_sha256"] != file_hash(root / row["user"])
                    or prior["generated_audio_sha256"]
                    != file_hash(destination_path / "generated.wav")
                ):
                    raise ValueError("Saved speech generation identity changed")
                outputs.append(prior)
                continue
            if not deadline.can_step(row["duration_s"] * 5 + 30):
                raise ValueError("Not enough reserved time for matched speech generation")
            visible = ModelInput(
                request_id=row["id"],
                user_audio=row["user"],
                duration_s=row["duration_s"],
                events=[],
            )
            outputs.append(
                generate(
                    lm,
                    mimi,
                    tok,
                    visible,
                    root,
                    run / destination / row["id"],
                    seed=cfg.train.seed,
                    adapter_sha256=adapter_hash,
                    system_prompt=CONVERSATION_SYSTEM,
                )
            )
        return outputs

    base_audio = replay("base-audio", None)
    enable_adapters(lm, True)
    lm.train()
    optimizer = torch.optim.AdamW(
        [p for p in lm.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    target_examples = math.ceil(len(paths["train"]) * cfg.max_effective_epochs)
    cap = min(
        cfg.train.max_steps or 64,
        math.ceil(target_examples / cfg.train.accumulation)
        if full
        else math.floor(target_examples / cfg.train.accumulation),
    )
    sampler = 0
    history = []
    best = None
    best_dev = float("inf")
    first_step = 0
    if resume:
        candidates = sorted(
            p for p in (run / "checkpoints").glob("step-*") if (p / "complete.json").exists()
        )
        if not candidates:
            raise ValueError("No complete checkpoint exists to resume")
        first_step, sampler = checkpoint.restore(candidates[-1], lm, optimizer, run_identity)
        if (run / "training.jsonl").exists():
            original_history = read_jsonl(run / "training.jsonl")
            history = [row for row in original_history if row["step"] <= first_step]
            if len(history) != len(original_history):
                shutil.copyfile(
                    run / "training.jsonl", run / f"training-before-resume-{time.time_ns()}.jsonl"
                )
        if (run / "best.json").exists():
            selected_best = read_json(run / "best.json")
            best = run / "checkpoints" / selected_best["checkpoint"]
            checkpoint.complete(best)
            best_dev = selected_best["dev_loss"]
    frozen_versions = {name: p._version for name, p in lm.named_parameters() if not p.requires_grad}
    step_s = 0.0
    if full:
        deadline.reserve_s = max(
            300, sum(r["generation_s"] for r in base_audio) * 1.35 + baseline_loss_s * 1.3 + 180
        )
    previous_handlers = {}
    if full:

        def request_stop(signum: int, frame: Any) -> None:
            deadline.stop_requested = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[sig] = signal.signal(sig, request_stop)
    last_step = first_step
    for step in range(first_step + 1, cap + 1):
        if (
            not full
            and (
                time.perf_counter() - started > 1200
                or deadline.remaining() < 600 + max(5, step_s * 1.5)
            )
        ) or (full and not deadline.can_step(max(5, step_s * 1.5))):
            break
        lm.train()
        before = time.perf_counter()
        step_config = cfg
        if full:
            micro_count = min(cfg.train.accumulation, target_examples - sampler)
            step_config = cfg.model_copy(
                update={"train": cfg.train.model_copy(update={"accumulation": micro_count})}
            )
        metrics, sampler = _step(lm, optimizer, paths["train"], sampler, step_config)
        torch.cuda.synchronize()
        step_s = time.perf_counter() - before
        history.append({"step": step, "optimizer_step_s": step_s, **metrics})
        last_step = step
        write_jsonl(run / "training.jsonl", history)
        print(
            f"OpenYap step {step}/{cap}: audio={metrics['audio_loss']:.5f}, text={metrics['text_loss']:.5f}, {step_s:.2f}s",
            flush=True,
        )
        if step % cfg.train.checkpoint_every == 0 or step == cap:
            saved = checkpoint.save(
                run / "checkpoints",
                lm,
                optimizer,
                {
                    "step": step,
                    "identity": run_identity,
                    "train_config": cfg.train.model_dump(),
                    "real_checkpoint": True,
                    "experiment_kind": cfg.experiment_kind,
                },
                sampler,
            )
            if deadline.stop_requested:
                break
            dev = summarize(losses(lm, paths["dev"], cfg, deadline))
            if sum(dev.values()) < best_dev:
                best_dev, best = sum(dev.values()), saved
                atomic_json(
                    run / "best.json",
                    {
                        "checkpoint": saved.name,
                        "dev_loss": best_dev,
                        "selection": "predeclared development native loss; no test selection",
                    },
                )
            if full:
                commit_volumes()
    if full and last_step < cap:
        saved = checkpoint.save(
            run / "checkpoints",
            lm,
            optimizer,
            {
                "step": last_step,
                "identity": run_identity,
                "train_config": cfg.train.model_dump(),
                "real_checkpoint": True,
                "experiment_kind": cfg.experiment_kind,
            },
            sampler,
        )
        status = {
            "kind": "openyap_full_partial",
            "step": last_step,
            "target_steps": cap,
            "checkpoint": str(saved),
            "complete": False,
            "next": "Resume the same full-train command with --resume in a fresh budgeted session",
        }
        atomic_json(run / "run-status.json", status)
        commit_volumes()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        return status
    for sig, handler in previous_handlers.items():
        signal.signal(sig, handler)
    deadline.reserve_s = 90
    if full and history and best is None:
        candidates = sorted(
            p for p in (run / "checkpoints").glob("step-*") if (p / "complete.json").exists()
        )
        best = candidates[-1]
        load_adapter(lm, best / "adapter.safetensors")
        best_dev = sum(summarize(losses(lm, paths["dev"], cfg, deadline)).values())
        atomic_json(
            run / "best.json",
            {
                "checkpoint": best.name,
                "dev_loss": best_dev,
                "selection": "recovered last completed checkpoint; development loss measured, no test selection",
            },
        )
    if not history or best is None:
        raise ValueError("No complete bounded training checkpoint; no success claimed")
    load_adapter(lm, best / "adapter.safetensors")
    gradient_evidence = modality_gradients(lm, paths["train"][0], cfg)
    adapted_test = losses(lm, paths["test"], cfg, deadline)
    adapted_dev = losses(lm, paths["dev"], cfg, deadline)
    adapter_hash = file_hash(best / "adapter.safetensors")
    adapted_audio = replay("adapted-audio", adapter_hash)
    changed = {
        name: int(torch.count_nonzero(t - initial[name])) for name, t in adapter_state(lm).items()
    }
    before_test, after_test = summarize(base_test), summarize(adapted_test)
    frozen_unchanged = all(
        p._version == frozen_versions[name]
        for name, p in lm.named_parameters()
        if not p.requires_grad
    )
    if not frozen_unchanged:
        raise ValueError("A frozen base parameter was modified")
    result = {
        "kind": "openyap_full_training" if full else "openyap_conversation_feasibility",
        "frozen_parameter_versions_unchanged": frozen_unchanged,
        "dataset": prepared,
        "load_s": load_s,
        "optimizer_steps": len(history),
        "unique_train_examples": len(paths["train"]),
        "unique_train_audio_s": prepared["audio_s"]["train"],
        "audio_exposure_s": sum(r["audio_seconds"] for r in history),
        "effective_epochs": sum(r["audio_seconds"] for r in history) / prepared["audio_s"]["train"],
        "seconds_per_optimizer_step": sum(r["optimizer_step_s"] for r in history) / len(history),
        "selected_checkpoint": best.name,
        "changed_adapter_elements": changed,
        "separate_modality_gradients": gradient_evidence,
        "base_dev": summarize(base_dev),
        "adapted_dev": summarize(adapted_dev),
        "base_test": before_test,
        "adapted_test": after_test,
        "test_loss_delta": {key: after_test[key] - before_test[key] for key in before_test},
        "per_example_test": {"base": base_test, "adapted": adapted_test},
        "generated_audio_changed": [
            b["generated_audio_sha256"] != a["generated_audio_sha256"]
            for b, a in zip(base_audio, adapted_audio, strict=True)
        ],
        "wall_s": time.perf_counter() - started,
        "capability_improvement": "unverified_without_audible_review",
        "limitations": [
            "Human-human conversational targets, no correction/tool-grounding labels.",
            "Machine ASR timestamps and strict window selection bias.",
            "Tiny speaker-disjoint held-out sample; loss change or output change is not demonstrated conversational quality.",
            "No target transcript or assistant audio is passed to free generation; no voice prompt or cloning.",
        ],
    }
    atomic_json(run / "results.json", result)
    checkpoint.export(best, run / "export", ROOT / "THIRD_PARTY_NOTICES.md")
    if full:
        atomic_json(
            run / "run-status.json",
            {"kind": "openyap_full_training", "complete": True, "step": len(history)},
        )
        commit_volumes()
    ledger.stage(
        "openyap_conversation_pilot",
        {"wall_s": result["wall_s"], "steps": len(history), "result": str(run / "results.json")},
    )
    return result
