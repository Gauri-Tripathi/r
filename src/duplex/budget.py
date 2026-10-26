"""Persistent conservative reservations. Estimates are never exact Modal invoices."""

from __future__ import annotations

import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

from filelock import FileLock
from pydantic import Field, model_validator

from .io import atomic_json, read_json
from .schema import Strict


class BudgetConfig(Strict):
    balance_usd: float = Field(gt=0, le=24.51, default=24.51)
    ceiling_usd: float = Field(gt=0, le=18, default=18)
    max_gpu_s: int = Field(gt=0, le=18000, default=18000)
    target_gpu_s: int = Field(gt=0, le=14400, default=14400)
    rate_usd_s: float = Field(gt=0, default=0.00098572)
    safety_margin: float = Field(ge=0.1, le=1, default=0.15)
    non_gpu_reserve_usd: float = Field(ge=0.5, default=1.5)
    evaluation_reserve_s: int = Field(ge=300, default=1800)
    checkpoint_reserve_s: int = Field(ge=60, default=180)
    prices_checked: str = "2026-10-31"

    @model_validator(mode="after")
    def limits(self) -> BudgetConfig:
        if self.ceiling_usd > self.balance_usd or self.target_gpu_s > self.max_gpu_s:
            raise ValueError("Budget exceeds balance or aggregate time cap")
        return self


class Ledger:
    def __init__(self, path: Path, config: BudgetConfig):
        self.path = path
        self.config = config
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(path) + ".lock", timeout=5)
        with self.lock:
            if path.exists():
                data = read_json(path)
                if data["config"] != config.model_dump():
                    raise ValueError(
                        "Existing ledger configuration differs; ceilings cannot reset on resume"
                    )
            else:
                atomic_json(
                    path,
                    {
                        "config": config.model_dump(),
                        "sessions": [],
                        "stages": [],
                        "oom_recoveries": 0,
                    },
                )

    def reserve(
        self, seconds: int, now: float | None = None, session_id: str | None = None
    ) -> dict[str, Any]:
        if seconds <= self.config.evaluation_reserve_s + self.config.checkpoint_reserve_s:
            raise ValueError("Session must reserve evaluation and checkpoint time")
        now = time.time() if now is None else now
        with self.lock:
            data = read_json(self.path)
            if any(s["status"] == "active" for s in data["sessions"]):
                raise ValueError(
                    "One GPU session at a time; reconcile previous reservation after verifying termination"
                )
            allocated = sum(s["charged_s"] for s in data["sessions"])
            total = allocated + seconds
            rate = self.config.rate_usd_s * (1 + self.config.safety_margin)
            accounting = data.get("accounting", [])
            prior = accounting[-1] if accounting else None
            since = max(0, allocated - prior["ledger_s"]) * rate if prior else 0
            spent = (
                max(allocated * rate, prior["metered_usd"] + since) if prior else allocated * rate
            )
            estimate = spent + seconds * rate + self.config.non_gpu_reserve_usd
            if total > self.config.max_gpu_s or estimate > self.config.ceiling_usd:
                raise ValueError("Reservation exceeds aggregate hours or estimated spend ceiling")
            if (
                prior
                and seconds * rate + self.config.non_gpu_reserve_usd
                > prior["remaining_credit_usd"] - since
            ):
                raise ValueError("Reservation exceeds current recorded remaining credits")
            session = {
                "id": session_id or uuid.uuid4().hex,
                "started": now,
                "deadline": now + seconds,
                "reserved_s": seconds,
                "charged_s": seconds,
                "status": "active",
                "estimated_usd": seconds * self.config.rate_usd_s * (1 + self.config.safety_margin),
            }
            data["sessions"].append(session)
            atomic_json(self.path, data)
            return session

    def account(self, remaining_credit_usd: float, metered_usd: float) -> None:
        """Append billing evidence without resetting historical sessions/ceilings."""
        if (
            not math.isfinite(remaining_credit_usd)
            or not math.isfinite(metered_usd)
            or remaining_credit_usd < 0
            or metered_usd < 0
        ):
            raise ValueError("Invalid billing amounts")
        with self.lock:
            data = read_json(self.path)
            if any(s["status"] == "active" for s in data["sessions"]):
                raise ValueError("Reconcile active sessions before recording billing")
            allocated = sum(s["charged_s"] for s in data["sessions"])
            rate = self.config.rate_usd_s * (1 + self.config.safety_margin)
            prior = data.get("accounting", [])
            floor = (
                prior[-1]["metered_usd"] + max(0, allocated - prior[-1]["ledger_s"]) * rate
                if prior
                else 0
            )
            data.setdefault("accounting", []).append(
                {
                    "at": time.time(),
                    "ledger_s": allocated,
                    "remaining_credit_usd": remaining_credit_usd,
                    "metered_usd": max(metered_usd, floor),
                    "basis": "operator dashboard and conservative cumulative metered usage",
                }
            )
            atomic_json(self.path, data)

    def close(self, session_id: str, confirmed_stopped: bool, now: float | None = None) -> None:
        if not confirmed_stopped:
            raise ValueError("Verify GPU termination before releasing reservation")
        now = time.time() if now is None else now
        with self.lock:
            data = read_json(self.path)
            session = next(s for s in data["sessions"] if s["id"] == session_id)
            if session["status"] != "active":
                return
            # Overruns remain charged and block subsequent reservations.
            session.update(status="closed", stopped=now, charged_s=max(0, now - session["started"]))
            atomic_json(self.path, data)

    def stage(self, name: str, metadata: dict[str, Any]) -> None:
        with self.lock:
            data = read_json(self.path)
            data["stages"].append({"name": name, "at": time.time(), **metadata})
            atomic_json(self.path, data)

    def oom_recovery(self) -> None:
        with self.lock:
            data = read_json(self.path)
            if data["oom_recoveries"] >= 1:
                raise ValueError("The single project OOM recovery attempt is exhausted")
            data["oom_recoveries"] += 1
            atomic_json(self.path, data)


class Deadline:
    def __init__(self, epoch: float, reserve_s: float = 180):
        self.epoch = epoch
        self.reserve_s = reserve_s
        self.stop_requested = False

    @classmethod
    def from_env(cls, reserve_s: float = 180) -> Deadline:
        value = os.environ.get("DUPLEX_DEADLINE")
        session_id = os.environ.get("DUPLEX_SESSION")
        ledger_path = Path(os.environ.get("DUPLEX_LEDGER", "/runs/budget.json"))
        if not value or not session_id or not ledger_path.exists():
            raise ValueError(
                "Use the budgeted shell launcher; active session/deadline/ledger is missing"
            )
        sessions = read_json(ledger_path)["sessions"]
        active = next(
            (s for s in sessions if s["id"] == session_id and s["status"] == "active"), None
        )
        if active is None or abs(active["deadline"] - float(value)) > 0.01:
            raise ValueError("Deadline does not match the persistent active reservation")
        if float(value) <= time.time():
            raise ValueError("Reservation deadline expired")
        return cls(float(value), reserve_s)

    def remaining(self) -> float:
        return self.epoch - time.time()

    def can_step(self, measured_step_s: float) -> bool:
        return not self.stop_requested and self.remaining() > self.reserve_s + measured_step_s


def derive_plan(profile: dict[str, Any], remaining_s: float, cfg: BudgetConfig) -> dict[str, Any]:
    if profile.get("real_checkpoint") is not True:
        raise ValueError("A real-checkpoint GPU smoke profile is required")
    if profile.get("generation_calibration") != "complete_session_v1":
        raise ValueError(
            "Run duplex profile --generation-only to measure a complete conversation; a short smoke rollout cannot budget matched evaluation"
        )
    step = float(profile["optimizer_step_s"])
    save = max(cfg.checkpoint_reserve_s, float(profile["checkpoint_s"]) * 2)
    evaluation = max(cfg.evaluation_reserve_s, float(profile["matched_eval_s"]) * 1.25)
    available = remaining_s - evaluation - save
    if step <= 0 or available < step:
        raise ValueError(
            f"Cannot fit training and matched evaluation in remaining reservation: {remaining_s:.0f}s left, {evaluation:.0f}s evaluation + {save:.0f}s checkpoint reserved"
        )
    steps = min(500, int(available / step))
    return {
        "steps": steps,
        "train_s": steps * step,
        "evaluation_reserve_s": evaluation,
        "checkpoint_reserve_s": save,
        "selected_audio_s": min(3600, steps * float(profile["audio_s_per_step"])),
        "estimated_usd": remaining_s * cfg.rate_usd_s * (1 + cfg.safety_margin),
    }


def measured_evaluation_s(generation: dict[str, Any], examples: int, load_s: float) -> float:
    """Complete-conversation elapsed time includes prompting once per example/model."""
    if examples < 4 or examples > 48 or examples % 4:
        raise ValueError("Choose 4..48 evaluation examples in complete groups of four")
    duration, elapsed = float(generation["duration_s"]), float(generation["generation_s"])
    if generation.get("truncated") is not False or not all(
        math.isfinite(x) and x > 0 for x in (duration, elapsed)
    ):
        raise ValueError("Evaluation budgeting needs a finite, complete conversation measurement")
    if not math.isfinite(load_s) or load_s < 0:
        raise ValueError("Invalid model-loading time")
    return 2 * examples * elapsed + 2 * load_s
