"""Run Bandit, preserving findings; allow only reviewed subprocess/PRNG diagnostics."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REVIEWED = {
    "src/duplex/audio.py": {"B404", "B603"},
    "src/duplex/launcher.py": {"B404", "B603"},
    "src/duplex/data.py": {"B311"},
    # Seeded dataset ordering only; never used for credentials or security tokens.
    "src/duplex/openyap_train.py": {"B311"},
    "src/duplex/train.py": {"B404", "B603"},
    "src/duplex/volume_commit.py": {"B404", "B603", "B607"},
    "scripts/secret_scan.py": {"B404", "B603", "B607"},
    "scripts/security_check.py": {"B404", "B603"},
    "scripts/check_modal_lock.py": {"B404", "B603", "B607"},
}


def main() -> None:
    dest = Path("runs/security/bandit.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "bandit",
            "-r",
            "src",
            "scripts",
            "modal_app.py",
            "-f",
            "json",
            "-o",
            str(dest),
            "-q",
        ],
        check=False,
        timeout=60,
    )
    if result.returncode not in (0, 1):
        raise SystemExit("Bandit failed to run")
    report = json.loads(dest.read_text(encoding="utf-8"))
    if report["errors"]:
        raise SystemExit("Bandit had scan errors")
    unresolved = []
    for finding in report["results"]:
        path = finding["filename"].replace("\\", "/").removeprefix("./")
        if finding["issue_severity"] != "LOW" or finding["test_id"] not in REVIEWED.get(
            path, set()
        ):
            unresolved.append(f"{path}:{finding['line_number']} {finding['test_id']}")
    if unresolved:
        raise SystemExit("Unreviewed security findings: " + ", ".join(unresolved))
    print(f"Bandit reviewed: {len(report['results'])} low-severity findings; full report in {dest}")


if __name__ == "__main__":
    main()
