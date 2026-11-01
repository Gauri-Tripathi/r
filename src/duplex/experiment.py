"""Dataset readiness checks and workload planning."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import Config
from .io import digest, file_hash, read_json


def research_readiness(encoded: Path, cfg: Config, baseline: Path | None) -> dict[str, Any]:
    report_path = encoded.parent / "prepared.json"
    if not report_path.exists():
        raise ValueError(
            "Research training needs prepared.json from checked teacher audio preparation"
        )
    report = read_json(report_path)
    if (
        report["kind"] != "checked_teacher_audio"
        or report["splits"].get("train", 0) < cfg.min_train_examples
        or report["splits"].get("dev", 0) < 4
    ):
        raise ValueError(
            "Prepared data is a fixture/calibration or too small for the declared research workload"
        )
    actual = {p.name: file_hash(p) for p in encoded.glob("*.safetensors")}
    if actual != report["encoded"]:
        raise ValueError("Prepared tensor manifest differs from the actual encoded dataset")
    groups: dict[str, str] = {}
    for path in encoded.glob("*.safetensors"):
        metadata = read_json(path.with_suffix(".json"))
        provenance = metadata.get("supervision_provenance", {})
        if (
            provenance.get("supervision_kind") != "checked_teacher"
            or provenance.get("teacher_contract") != "duplex.teacher-response.v1"
        ):
            raise ValueError(
                "Reference fixtures/unversioned targets cannot be the research training dataset"
            )
        group, split = metadata["group_id"], metadata["split"]
        if split == "test" or (group in groups and groups[group] != split):
            raise ValueError("Encoded group/test leakage")
        groups[group] = split
    if baseline is None:
        raise ValueError("Run and score the locked audible base baseline before research training")
    measured = read_json(baseline)
    from .inference import runtime_fingerprint
    from .model import PIN

    if (
        measured.get("kind") != "locked_base_baseline"
        or measured.get("evidence") != "audible_asr"
        or measured.get("count", 0) < 8
    ):
        raise ValueError(
            "Baseline must be hash-matched generated speech with cached ASR, not an internal-text proxy"
        )
    if measured.get("all_cases_solved"):
        raise ValueError(
            "Base already solved this locked baseline; no demonstrated capability gap. Review development cases before training, without inspecting final test results."
        )
    if (
        measured.get("base_revision") != PIN
        or measured.get("runtime_fingerprint") != runtime_fingerprint()
    ):
        raise ValueError(
            "Baseline must use the current pinned base and identical inference runtime"
        )
    return report


def exposure_plan(
    metadata: list[dict[str, Any]],
    accumulation: int,
    epochs: float,
    wall_step_cap: int,
    step_cap: int | None,
) -> dict[str, Any]:
    if not metadata or accumulation < 1 or not 0 < epochs <= 3:
        raise ValueError("Invalid bounded data-exposure plan")
    steps = min(
        math.floor(len(metadata) * epochs / accumulation), wall_step_cap, step_cap or wall_step_cap
    )
    if steps < 1:
        raise ValueError("Not enough examples/time for one bounded optimizer step")
    durations = [m["audio_frames"] / 12.5 for m in metadata]
    exposure = sum(durations[i % len(durations)] for i in range(steps * accumulation))
    return {
        "steps": steps,
        "unique_train_examples": len(metadata),
        "unique_train_audio_s": sum(durations),
        "planned_audio_exposure_s": exposure,
        "approx_effective_epochs": exposure / sum(durations),
        "maximum_example_passes": math.ceil(steps * accumulation / len(metadata)),
        "epoch_limit": epochs,
    }


def aggregate_plan(
    prepared: Path,
    profile: Path,
    ledger: Path,
    cfg: Config,
    remaining_credit_usd: float,
    previous_metered_usd: float,
    dev_evaluation_lock: Path,
    test_evaluation_lock: Path,
    extra_preparation_usd: float = 0,
) -> dict[str, Any]:
    if remaining_credit_usd <= 0 or extra_preparation_usd < 0 or previous_metered_usd < 0:
        raise ValueError("Enter current dashboard credits and a nonnegative preparation estimate")
    data, measured, book = read_json(prepared), read_json(profile), read_json(ledger)
    if data.get("kind") != "checked_teacher_audio":
        raise ValueError("A preparation measurement/fixture is not the research dataset")
    if any(s["status"] == "active" for s in book["sessions"]):
        raise ValueError("Reconcile active sessions before planning another allocation")
    used_s = sum(s["charged_s"] for s in book["sessions"])
    rate = cfg.budget.rate_usd_s * (1 + cfg.budget.safety_margin)
    accounting = book.get("accounting", [])
    recorded = (
        accounting[-1]["metered_usd"] + max(0, used_s - accounting[-1]["ledger_s"]) * rate
        if accounting
        else 0
    )
    used_estimate = max(used_s * rate, previous_metered_usd, recorded)
    encoded = prepared.parent / "encoded"
    metadata = [
        read_json(p.with_suffix(".json"))
        for p in sorted(encoded.glob("*.safetensors"))
        if read_json(p.with_suffix(".json"))["split"] == "train"
    ]
    exposure = exposure_plan(
        metadata, cfg.train.accumulation, cfg.max_effective_epochs, 500, cfg.train.max_steps
    )
    mean_audio = sum(m["audio_frames"] for m in metadata) / (12.5 * len(metadata))
    # Scale an old pilot measurement by mean session length; a new dataset profile must confirm it.
    step_s = measured["optimizer_step_s"] * max(
        1, mean_audio * cfg.train.accumulation / measured["audio_s_per_step"]
    )
    dev_lock, test_lock = read_json(dev_evaluation_lock), read_json(test_evaluation_lock)
    if any(lock.get("kind") != "locked_user_audio_evaluation" for lock in (dev_lock, test_lock)):
        raise ValueError("Plan needs measured locked evaluation waveform durations")
    load_s = measured.get("calibration_load_s", 60)
    baseline_s = (dev_lock["audio_s"] * measured["generation_rtf"] + load_s) * 1.25
    evaluation_s = max(
        cfg.budget.evaluation_reserve_s,
        2 * (test_lock["audio_s"] * measured["generation_rtf"] + load_s) * 1.25,
    )
    profile_s = max(300, measured.get("stage_s", 180) * 1.5)
    monitoring_s = (
        math.ceil(exposure["steps"] / cfg.train.checkpoint_every)
        * data["split_audio_s"]["dev"]
        / measured["audio_s_per_step"]
        * step_s
    )
    training_s = (
        exposure["steps"] * step_s + load_s + monitoring_s + cfg.budget.checkpoint_reserve_s + 120
    )
    total_s = baseline_s + profile_s + training_s + evaluation_s + 3 * 120
    proposed = total_s * rate + extra_preparation_usd
    available = min(
        remaining_credit_usd,
        cfg.budget.ceiling_usd - used_estimate - cfg.budget.non_gpu_reserve_usd,
    )
    # Reduce optimization work to fit the complete allocation, without changing
    # the locked evaluation or exceeding the data-exposure cap.
    original_steps = exposure["steps"]
    feasible = 0
    for steps in range(original_steps, 0, -1):
        monitor = (
            math.ceil(steps / cfg.train.checkpoint_every)
            * data["split_audio_s"]["dev"]
            / measured["audio_s_per_step"]
            * step_s
        )
        train_s = steps * step_s + load_s + monitor + cfg.budget.checkpoint_reserve_s + 120
        candidate_s = baseline_s + profile_s + train_s + evaluation_s + 360
        if (
            candidate_s * rate + extra_preparation_usd <= available
            and used_s + candidate_s <= cfg.budget.max_gpu_s
        ):
            feasible = steps
            monitoring_s, training_s, total_s = monitor, train_s, candidate_s
            proposed = total_s * rate + extra_preparation_usd
            exposure = exposure_plan(
                metadata,
                cfg.train.accumulation,
                cfg.max_effective_epochs,
                feasible,
                cfg.train.max_steps,
            )
            break
    return {
        "kind": "conditional_research_plan",
        "prepared_sha256": file_hash(prepared),
        "config_fingerprint": digest(cfg.train.model_dump()),
        **exposure,
        "seconds_per_optimizer_step_estimate": step_s,
        "baseline_s": baseline_s,
        "fresh_profile_reserve_s": profile_s,
        "development_monitor_s": monitoring_s,
        "baseline_examples": dev_lock["examples"],
        "held_out_examples_per_model": test_lock["examples"],
        "preparation_measured_wall_s": data["wall_s"] + data.get("codec_loading_s", 0),
        "training_load_save_monitor_s": training_s,
        "matched_evaluation_s": evaluation_s,
        "shutdown_and_idle_margin_s": 360,
        "previous_ledger_s": used_s,
        "previous_estimated_usd": used_estimate,
        "previous_metered_usd": previous_metered_usd,
        "future_non_gpu_reserve_usd": cfg.budget.non_gpu_reserve_usd,
        "additional_estimated_usd": proposed,
        "remaining_dashboard_credit_usd": remaining_credit_usd,
        "fits": feasible > 0,
        "feasible_training_steps": feasible,
        "exposure_limited_steps_before_budget": original_steps,
        "available_under_original_ceiling_usd": available,
        "exact_billing": False,
        "limitations": "Requires fresh dataset profile, actual held-out durations and cold preparation cost. Do not fill the time cap. Evaluation may need separate <=3300s shells.",
    }
