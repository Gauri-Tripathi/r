"""Supported modal shell wrapper. No manually managed Sandbox or public service."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .budget import Ledger
from .config import load_config
from .io import atomic_json, read_json


def launch(
    config: Path, ledger_path: Path, seconds: int = 3300, stage: str = "interactive"
) -> None:
    if sys.platform == "win32":
        raise ValueError(
            "Modal 1.6.3 shell does not support Windows; run this command from WSL/Linux"
        )
    if stage not in ("interactive", "doctor", "openyap") or not 300 < seconds <= 3300:
        raise ValueError(
            "Supported reservation is <=3300s, leaving room before Modal's 3600s shell limit"
        )
    cfg = load_config(config)
    # Values are validated/generated internally; Modal's --cmd string is not built from dataset/tool content.
    if cfg.budget != load_config(Path("configs/train.yaml")).budget:
        raise ValueError(
            "GPU launcher uses the audited configs/train.yaml budget; custom ceilings require updating both persistent ledgers"
        )
    ledger = Ledger(ledger_path, cfg.budget)
    accounting = read_json(ledger_path).get("accounting", [])
    if (
        cfg.experiment_kind in ("research", "conversation_pilot", "conversation_full")
        and not accounting
    ):
        raise ValueError(
            "Record current credits and cumulative usage with duplex budget-account before research allocation"
        )
    session = ledger.reserve(seconds)
    import modal

    lease = modal.Dict.from_name("duplex-gpu-reservation", create_if_missing=True)
    if not lease.put(
        "active",
        {
            "session": session["id"],
            "deadline": session["deadline"],
            "accounting": accounting[-1] if accounting else None,
        },
        skip_if_exists=True,
    ):
        raise ValueError(
            "Another project GPU reservation exists in Modal. Do not launch a second shell; reconcile the existing reservation first."
        )
    command = f"cd /opt/duplex && /opt/duplex/.venv/bin/python -m duplex.launcher remote {session['deadline']:.3f} {session['id']} {seconds} {stage}"
    # Reserve BEFORE allocation, including CLI startup/image wait and interactive idle time.
    env = os.environ.copy()
    env["DUPLEX_LAUNCH_ALLOWED"] = "1"
    args = [sys.executable, "-m", "modal", "shell", "modal_app.py::gpu_shell", "--cmd", command]
    process = subprocess.Popen(args, env=env, start_new_session=True)
    try:
        while process.poll() is None and time.time() < session["deadline"] + 15:
            time.sleep(1)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    except KeyboardInterrupt:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
    finally:
        # A disconnected CLI is not proof of remote termination. Reservation stays charged.
        ledger.stage(
            "shell_client_exit",
            {"session": session["id"], "returncode": process.poll(), "cleanup_verified": False},
        )
        print(
            f"Reservation {session['id']} remains charged. Verify termination in Modal, then duplex budget-reconcile --ledger {ledger_path} --session {session['id']} --confirmed-stopped."
        )


def remote(deadline_epoch: float, session_id: str, seconds: int, stage: str) -> None:
    """Best effort remote process-group watchdog plus volume lease/checkpoint flushing.

    Modal's independent one-hour shell timeout is the fallback if this worker is killed.
    Client disconnect and server cleanup require paid-session verification, not assumptions.
    """
    if sys.platform == "win32":
        raise ValueError("Remote watchdog requires Linux")
    import re

    from filelock import FileLock

    if (
        stage not in ("interactive", "doctor", "openyap")
        or deadline_epoch - time.time() > 3300
        or deadline_epoch <= time.time()
        or not 1980 < seconds <= 3300
        or re.fullmatch(r"[0-9a-f]{32}", session_id) is None
    ):
        raise ValueError("Invalid remote reservation")
    lease = Path("/runs/gpu-lease.json")
    cfg = load_config(Path("/opt/duplex/configs/train.yaml"))
    remote_ledger = Ledger(Path("/runs/budget.json"), cfg.budget)
    import modal

    cloud_lease = modal.Dict.from_name("duplex-gpu-reservation")
    active = cloud_lease.get("active")
    if not active or active["session"] != session_id:
        raise ValueError("No matching project cloud lease; use the recommended launcher")
    if active.get("accounting"):
        evidence = active["accounting"]
        remote_ledger.account(evidence["remaining_credit_usd"], evidence["metered_usd"])
    remote_ledger.reserve(seconds, now=deadline_epoch - seconds, session_id=session_id)
    with FileLock(str(lease) + ".lock", timeout=2):
        if lease.exists():
            # Modal Volumes are not a distributed lock. Refuse a stale lease until manual reconciliation.
            raise ValueError(
                "GPU volume lease exists; verify previous shell stopped before explicit reconciliation"
            )
        atomic_json(
            lease, {"session": session_id, "deadline": deadline_epoch, "started": time.time()}
        )
    from .train import commit_volumes

    commit_volumes()
    env = os.environ.copy()
    env.update(DUPLEX_DEADLINE=str(deadline_epoch), DUPLEX_SESSION=session_id, DUPLEX_MODAL="1")
    env["DUPLEX_LEDGER"] = "/runs/budget.json"
    if stage == "openyap":
        args = [
            sys.executable,
            "-m",
            "duplex.cli",
            "openyap",
            "pilot",
            "--root",
            "/data/openyap/pilot-v1",
            "--assets",
            "/cache/assets",
            "--run",
            "/runs/openyap/pilot-v1",
            "--ledger",
            "/runs/budget.json",
        ]
    elif stage == "interactive":
        args = ["/bin/bash", "--noprofile", "--norc"]
    else:
        args = [sys.executable, "-m", "duplex.cli", "doctor"]
    child = subprocess.Popen(args, env=env, start_new_session=True)
    try:
        # Deadline sends TERM with 180s save/flush grace counted INSIDE reservation.
        while child.poll() is None and time.time() < deadline_epoch - 180:
            time.sleep(1)
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            while child.poll() is None and time.time() < deadline_epoch - 45:
                time.sleep(1)
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
    finally:
        # Exiting bash may leave background jobs in its group. Bound their cleanup too.
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            time.sleep(2)
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        atomic_json(
            Path(f"/runs/session-{session_id}.json"),
            {
                "session": session_id,
                "ended": time.time(),
                "child_returncode": child.poll(),
                "remote_worker_exit": True,
            },
        )
        lease.unlink(missing_ok=True)
        commit_volumes()


def main() -> None:
    if len(sys.argv) != 6 or sys.argv[1] != "remote":
        raise ValueError("Use duplex shell; remote is an internal watchdog entry point")
    remote(float(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])


if __name__ == "__main__":
    main()
