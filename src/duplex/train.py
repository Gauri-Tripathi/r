"""Single-GPU supervised training with native text and audio losses."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from . import checkpoint
from .budget import Deadline, Ledger, derive_plan, measured_evaluation_s
from .config import Config
from .experiment import exposure_plan, research_readiness
from .io import atomic_json, digest, file_hash, read_json
from .model import PIN, adapter_state, attach, load_adapter, load_real, native_loss
from .replay import mix_frames
from .tracking import Tracker


def dataset_paths(directory: Path, split: str, max_frames: int) -> list[Path]:
    paths = []
    for path in sorted(directory.glob("*.safetensors")):
        meta = read_json(path.with_suffix(".json"))
        if meta["split"] != split:
            continue
        if meta["frames"] > max_frames:
            # Reject long sessions to preserve the post-correction target.
            raise ValueError(
                f"{path.name} exceeds max_frames; profile/choose shorter complete sessions"
            )
        if file_hash(path) != meta["sha256"] or meta["codec_revision"] != PIN:
            raise ValueError("Encoded sample integrity/revision failure")
        paths.append(path)
    if not paths:
        raise ValueError(f"No complete {split} encoded examples")
    return paths


def batch(path: Path, device: str, post_weight: float) -> dict[str, torch.Tensor]:
    sample = load_file(str(path))
    if set(sample) != {"codes", "valid", "weights", "post", "prompt_frames"}:
        raise ValueError("Unexpected encoded sample fields")
    if sample["valid"][:, 9:].any():
        raise ValueError("User channel loss must be masked")
    sample["weights"][..., sample["post"]] *= post_weight
    return {k: v.to(device) for k, v in sample.items()}


def _step(
    lm: Any, optimizer: Any, paths: list[Path], sampler: int, config: Config
) -> tuple[dict[str, float], int]:
    cfg = config.train
    optimizer.zero_grad(set_to_none=True)
    metrics: dict[str, float] = {}
    for micro in range(cfg.accumulation):
        sample = batch(paths[(sampler + micro) % len(paths)], str(lm.device), cfg.post_event_weight)
        output = lm.forward_train(sample["codes"])
        loss, stats = native_loss(
            output, sample["codes"], sample["valid"], sample["weights"], cfg.text_pad_weight
        )
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native objective")
        (loss / cfg.accumulation).backward()
        stats["loss"] = loss.item()
        stats["audio_seconds"] = (sample["codes"].shape[-1] - int(sample["prompt_frames"])) / 12.5
        for key, value in stats.items():
            divisor = cfg.accumulation if key in ("loss", "text_loss", "audio_loss") else 1
            metrics[key] = metrics.get(key, 0) + value / divisor
    params = [p for p in lm.parameters() if p.requires_grad]
    missing = [
        name
        for name, p in lm.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    if missing:
        raise ValueError(f"Missing/nonfinite adapter gradients: {missing}")
    # At step 1, zero-initialized B means A has a zero gradient. Every B must be nonzero.
    if any(
        p.grad is None or not p.grad.abs().sum()
        for name, p in lm.named_parameters()
        if p.requires_grad and name.endswith(".b")
    ):
        raise ValueError("Audio/text loss does not reach every intended temporal adapter")
    norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
    optimizer.step()
    metrics.update(grad_norm=float(norm), lr=cfg.lr, examples=float(cfg.accumulation))
    return metrics, sampler + cfg.accumulation


def identity(paths: list[Path], config: Config, modules: list[str]) -> dict[str, Any]:
    return {
        "base_revision": PIN,
        "codec_revision": PIN,
        "tokenizer_revision": PIN,
        "official_commit": "3428dfd95309a7f3c84fd93259ded0f810d1ff91",
        "runtime_patch": "native-step-facts-v1",
        "runtime_source_sha256": {
            p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "targets": modules,
        "data": digest([file_hash(p) for p in paths]),
        "train": config.train.model_dump(),
    }


def modality_gradients(lm: Any, path: Path, config: Config) -> dict[str, Any]:
    """Evidence from separate backward passes, without an optimizer update."""
    result = {}
    for modality in ("audio", "text"):
        lm.zero_grad(set_to_none=True)
        sample = batch(path, str(lm.device), config.train.post_event_weight)
        output = lm.forward_train(sample["codes"])
        if modality == "audio":
            output.text_logits = output.text_logits.detach()
        else:
            output.logits = output.logits.detach()
        loss, _ = native_loss(
            output,
            sample["codes"],
            sample["valid"],
            sample["weights"],
            config.train.text_pad_weight,
        )
        loss.backward()
        norms = {
            name: float(p.grad.float().norm()) if p.grad is not None else 0.0
            for name, p in lm.named_parameters()
            if p.requires_grad
        }
        if not norms or any(not 0 < value < float("inf") for value in norms.values()):
            raise ValueError(f"{modality}-only loss failed to reach every adapter: {norms}")
        result[modality] = norms
    lm.zero_grad(set_to_none=True)
    return result


def profile(
    assets: Path,
    encoded: Path,
    run: Path,
    config: Config,
    deadline: Deadline,
    ledger: Ledger,
    visible: Any,
    audio_root: Path,
    voice: Path | None = None,
    eval_examples: int = 12,
) -> dict[str, Any]:
    """Actual load, audio+text backward, save/reload, optimizer, and audible generation."""
    from .inference import generate

    os.environ.update(NO_TORCH_COMPILE="1", NO_CUDA_GRAPH="1")
    if not deadline.can_step(120):
        raise ValueError("Insufficient smoke-test time")
    start = time.perf_counter()
    lm, mimi, tok = load_real(assets)
    model_load_s = time.perf_counter() - start
    modules = attach(lm, config.train)
    lm.train()
    optimizer = torch.optim.AdamW(
        [p for p in lm.parameters() if p.requires_grad],
        lr=config.train.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    paths, replay_stats = mix_frames(
        dataset_paths(encoded, "train", config.train.max_frames), config.train.replay_frame_fraction
    )
    atomic_json(run / "replay-selection.json", replay_stats)
    frozen = {name: file_hash_tensor(p) for name, p in lm.named_parameters() if not p.requires_grad}
    torch.cuda.reset_peak_memory_stats()
    step_start = time.perf_counter()
    stats, sampler = _step(lm, optimizer, paths, 0, config)
    torch.cuda.synchronize()
    step_s = time.perf_counter() - step_start
    if any(
        file_hash_tensor(p) != frozen[name]
        for name, p in lm.named_parameters()
        if not p.requires_grad
    ):
        raise ValueError("Frozen base parameter changed")
    save_start = time.perf_counter()
    meta = {
        "step": 1,
        "identity": identity(paths, config, modules),
        "train_config": config.train.model_dump(),
        "trainable": sum(p.numel() for p in lm.parameters() if p.requires_grad),
        "ledger": str(ledger.path),
        "real_checkpoint": True,
    }
    saved = checkpoint.save(run / "smoke-checkpoints", lm, optimizer, meta, sampler)
    before = adapter_state(lm)
    load_adapter(lm, saved / "adapter.safetensors")
    if any(not torch.equal(t, adapter_state(lm)[n]) for n, t in before.items()):
        raise ValueError("Adapter reload differs")
    checkpoint_s = time.perf_counter() - save_start
    gradient_evidence = modality_gradients(lm, paths[0], config)
    lm.eval()
    generated = generate(
        lm,
        mimi,
        tok,
        visible,
        audio_root,
        run / "smoke-generation",
        voice=voice,
        adapter_sha256=file_hash(saved / "adapter.safetensors"),
    )
    result = {
        "real_checkpoint": True,
        "config_fingerprint": digest(config.train.model_dump()),
        "data_fingerprint": meta["identity"]["data"],
        "optimizer_step_s": step_s,
        "checkpoint_s": checkpoint_s,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "audio_s_per_step": stats["audio_seconds"],
        "generation_rtf": generated["rtf"],
        "generation_calibration": "complete_session_v1",
        "evaluation_examples_per_model": eval_examples,
        "calibration_duration_s": generated["duration_s"],
        "calibration_generation_s": generated["generation_s"],
        "calibration_load_s": model_load_s,
        "matched_eval_s": measured_evaluation_s(generated, eval_examples, model_load_s),
        "stage_s": time.perf_counter() - start,
        "modules": modules,
        "trainable": meta["trainable"],
        "gradients_verified": True,
        "native_stream_delays": list(lm.delays),
        "separate_modality_adapter_gradient_norms": gradient_evidence,
        "frozen_verified": True,
        "save_reload_verified": True,
    }
    atomic_json(run / "profile.json", result)
    ledger.stage("real_checkpoint_smoke", result)
    return result


def calibrate_generation(
    assets: Path,
    encoded: Path,
    run: Path,
    config: Config,
    deadline: Deadline,
    ledger: Ledger,
    visible: Any,
    audio_root: Path,
    voice: Path | None = None,
    eval_examples: int = 12,
) -> dict[str, Any]:
    """Calibrate generation time on a training conversation."""
    from .inference import generate

    prior = read_json(run / "profile.json")
    if prior.get("real_checkpoint") is not True or prior["config_fingerprint"] != digest(
        config.train.model_dump()
    ):
        raise ValueError("A successful smoke profile with the same config is required first")
    if eval_examples < 4 or eval_examples > 48 or eval_examples % 4:
        raise ValueError("Evaluation count must be 4..48 in groups of four")
    paths, _ = mix_frames(
        dataset_paths(encoded, "train", config.train.max_frames), config.train.replay_frame_fraction
    )
    if prior["data_fingerprint"] != digest([file_hash(p) for p in paths]):
        raise ValueError("Calibration data differs from the successful smoke test")
    if not deadline.can_step(120 + visible.duration_s * float(prior["generation_rtf"])):
        raise ValueError("Insufficient session time for complete-conversation calibration")
    started = time.perf_counter()
    lm, mimi, tok = load_real(assets)
    modules = attach(lm, config.train)
    saved = run / "smoke-checkpoints" / "step-000001"
    meta = checkpoint.complete(saved)
    if meta["identity"] != identity(paths, config, modules):
        raise ValueError("Smoke checkpoint identity differs")
    load_adapter(lm, saved / "adapter.safetensors")
    load_s = time.perf_counter() - started
    generated = generate(
        lm,
        mimi,
        tok,
        visible,
        audio_root,
        run / "generation-calibration",
        voice=voice,
        adapter_sha256=file_hash(saved / "adapter.safetensors"),
    )
    result = {
        **prior,
        "generation_rtf": generated["rtf"],
        "generation_calibration": "complete_session_v1",
        "evaluation_examples_per_model": eval_examples,
        "calibration_duration_s": generated["duration_s"],
        "calibration_generation_s": generated["generation_s"],
        "calibration_load_s": load_s,
        "matched_eval_s": measured_evaluation_s(generated, eval_examples, load_s),
    }
    if not (run / "profile.before-generation-calibration.json").exists():
        atomic_json(run / "profile.before-generation-calibration.json", prior)
    atomic_json(run / "profile.json", result)
    ledger.stage(
        "complete_generation_calibration", {"wall_s": time.perf_counter() - started, **result}
    )
    return result


def file_hash_tensor(tensor: torch.Tensor) -> str:
    import hashlib

    return hashlib.sha256(tensor.detach().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def run_training(
    assets: Path,
    encoded: Path,
    run: Path,
    config: Config,
    deadline: Deadline,
    ledger: Ledger,
    profile_path: Path,
    resume: Path | None = None,
    baseline: Path | None = None,
    workload_plan: Path | None = None,
) -> Path:
    stage_started = time.perf_counter()
    if config.experiment_kind in ("conversation_pilot", "conversation_full"):
        raise ValueError("Use duplex openyap pilot for the separate real-conversation experiment")
    if resume is None and any((run / "checkpoints").glob("step-*")):
        raise ValueError(
            "Completed run exists; use an explicit resume checkpoint or a new run directory"
        )
    os.environ.update(NO_TORCH_COMPILE="1", NO_CUDA_GRAPH="1")
    measured = read_json(profile_path)
    if measured["config_fingerprint"] != digest(config.train.model_dump()):
        raise ValueError("Profile must use the exact training configuration")
    if config.experiment_kind == "research":
        research_readiness(encoded, config, baseline)
        if workload_plan is None:
            raise ValueError("Research training requires the complete measured --workload-plan")
        workload = read_json(workload_plan)
        if (
            workload.get("kind") != "conditional_research_plan"
            or not workload.get("fits")
            or workload.get("feasible_training_steps", 0) < 1
        ):
            raise ValueError("Complete research workload is not feasible")
        if workload.get("prepared_sha256") != file_hash(
            encoded.parent / "prepared.json"
        ) or workload.get("config_fingerprint") != digest(config.train.model_dump()):
            raise ValueError("Workload plan differs from prepared data/configuration")
    plan = derive_plan(measured, deadline.remaining(), config.budget)
    all_paths = dataset_paths(encoded, "train", config.train.max_frames)
    if config.experiment_kind == "research":
        plan.update(
            exposure_plan(
                [read_json(p.with_suffix(".json")) for p in all_paths],
                config.train.accumulation,
                config.max_effective_epochs,
                min(plan["steps"], workload["feasible_training_steps"]),
                config.train.max_steps,
            )
        )
    elif config.train.max_steps is None:
        raise ValueError(
            "Smoke tests require an explicit small step cap; research steps require prepared-data planning"
        )
    plan["train_s"] = plan["steps"] * measured["optimizer_step_s"]
    deadline.reserve_s = plan["evaluation_reserve_s"] + plan["checkpoint_reserve_s"]
    atomic_json(run / "plan.json", plan)
    atomic_json(
        run / "run-status.json",
        {"kind": config.experiment_kind, "capability_improvement": "unverified", "plan": plan},
    )
    lm, mimi, _ = load_real(assets)
    del mimi
    torch.manual_seed(config.train.seed)
    modules = attach(lm, config.train)
    paths, replay_stats = mix_frames(
        dataset_paths(encoded, "train", config.train.max_frames), config.train.replay_frame_fraction
    )
    atomic_json(run / "replay-selection.json", replay_stats)
    dev_paths = dataset_paths(encoded, "dev", config.train.max_frames)
    run_identity = identity(paths, config, modules)
    if measured["data_fingerprint"] != run_identity["data"]:
        raise ValueError("Training data differs from profiled data")
    optimizer = torch.optim.AdamW(
        [p for p in lm.parameters() if p.requires_grad],
        lr=config.train.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    step, sampler = checkpoint.restore(resume, lm, optimizer, run_identity) if resume else (0, 0)
    if resume is None:
        save_file(adapter_state(lm), str(run / "initial-adapter.safetensors"))
        atomic_json(
            run / "initial-adapter.json",
            {
                "sha256": file_hash(run / "initial-adapter.safetensors"),
                "identity": run_identity,
                "native_stream_delays": list(lm.delays),
                "step": 0,
            },
        )
    tracker = Tracker(
        run, config.train.wandb_mode, config.train.wandb_project, config.train.wandb_entity
    )
    if tracker.last_step > step:
        tracker.close()
        raise ValueError(
            "Logs are ahead of checkpoint; resume in a new run directory linked to the prior checkpoint"
        )
    previous_handlers = {}

    def interrupt(signum: int, frame: Any) -> None:
        deadline.stop_requested = True

    for sig in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[sig] = signal.signal(sig, interrupt)
    best_metric = float("inf")
    best: Path | None = None
    if (run / "best.json").exists():
        prior_best = read_json(run / "best.json")
        candidate = run / "checkpoints" / prior_best["checkpoint"]
        if candidate.resolve().parent != (run / "checkpoints").resolve():
            raise ValueError("Invalid best-checkpoint pointer")
        checkpoint.complete(candidate)
        best, best_metric = candidate, float(prior_best["dev_loss"])
    latest: Path | None = None
    lm.train()
    total_cap = config.train.max_steps or plan["steps"]
    if config.experiment_kind == "research":
        total_cap = min(
            total_cap,
            int(len(paths) * config.max_effective_epochs / config.train.accumulation),
            workload["feasible_training_steps"],
        )
    stop_step = min(total_cap, step + plan["steps"])
    plan.update(
        cumulative_step_cap=total_cap,
        resume_from_step=step,
        steps_this_session=max(0, stop_step - step),
    )
    atomic_json(run / "plan.json", plan)
    print(
        f"Starting {config.experiment_kind}: {len(paths)} unique examples, optimizer steps {step}..{stop_step}",
        flush=True,
    )
    try:
        while step < stop_step and deadline.can_step(measured["optimizer_step_s"] * 1.3):
            started = time.perf_counter()
            stats, sampler = _step(lm, optimizer, paths, sampler, config)
            step += 1
            if step == 1 or step % 5 == 0:
                print(
                    f"step {step}/{stop_step}: text_loss={stats['text_loss']:.4f} audio_loss={stats['audio_loss']:.4f}",
                    flush=True,
                )
            elapsed = time.perf_counter() - started
            tracker.log(
                {
                    "step": step,
                    **stats,
                    "steps_per_s": 1 / elapsed,
                    "trainable": sum(p.numel() for p in lm.parameters() if p.requires_grad),
                    "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                    "deadline_remaining_s": deadline.remaining(),
                    "stage_s": elapsed,
                    "data_fingerprint": run_identity["data"],
                }
            )
            if step % config.train.checkpoint_every == 0 or step == stop_step:
                latest = checkpoint.save(
                    run / "checkpoints",
                    lm,
                    optimizer,
                    {
                        "step": step,
                        "identity": run_identity,
                        "train_config": config.train.model_dump(),
                        "ledger": str(ledger.path),
                        "real_checkpoint": True,
                        "experiment_kind": config.experiment_kind,
                    },
                    sampler,
                )
                commit_volumes()
                if deadline.can_step(
                    measured["optimizer_step_s"] * len(dev_paths) / config.train.accumulation * 1.5
                ):
                    metric = dev_loss(lm, dev_paths, config)
                    tracker.log({"step": step, "dev_loss": metric})
                    if metric < best_metric:
                        best_metric, best = metric, latest
                        atomic_json(
                            run / "best.json",
                            {
                                "checkpoint": latest.name,
                                "dev_loss": metric,
                                "selection": "native dev loss; audible held-out evaluation separate",
                            },
                        )
                checkpoint.retain(run / "checkpoints", best)
        if latest is None or complete_step(latest) != step:
            latest = checkpoint.save(
                run / "checkpoints",
                lm,
                optimizer,
                {
                    "step": step,
                    "identity": run_identity,
                    "train_config": config.train.model_dump(),
                    "ledger": str(ledger.path),
                    "real_checkpoint": True,
                    "experiment_kind": config.experiment_kind,
                },
                sampler,
            )
            commit_volumes()
        ledger.stage(
            "training",
            {
                "step": step,
                "checkpoint": latest.name,
                "wall_s": time.perf_counter() - stage_started,
            },
        )
    finally:
        tracker.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return latest


def complete_step(path: Path) -> int:
    return int(checkpoint.complete(path)["step"])


def dev_loss(lm: Any, paths: list[Path], config: Config) -> float:
    lm.eval()
    losses = []
    with torch.no_grad():
        for path in paths:
            sample = batch(path, str(lm.device), 1)
            loss, _ = native_loss(
                lm.forward_train(sample["codes"]),
                sample["codes"],
                sample["valid"],
                sample["weights"],
                config.train.text_pad_weight,
            )
            losses.append(loss.item())
    lm.train()
    return sum(losses) / len(losses)


def commit_volumes() -> None:
    if os.environ.get("DUPLEX_MODAL") == "1":
        import subprocess
        import sys

        # SDK commit retries can exceed a save grace. Bound the entire flush worker.
        subprocess.run(
            [sys.executable, "-m", "duplex.volume_commit"],
            check=True,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
