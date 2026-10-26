"""Validate a minimal Modal image context without allocating cloud resources."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="duplex-image-lock-") as temporary:
        context = Path(temporary)
        for name in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(root / name, context / name)
        shutil.copytree(
            root / "src",
            context / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        command = ["uvx", "--from", "uv==0.8.10", "uv"]
        help_text = subprocess.run(
            command + ["sync", "--help"], check=True, capture_output=True, text=True, timeout=30
        ).stdout
        args = ["sync", "--locked", "--extra", "gpu", "--no-dev", "--python", "3.11.12"]
        if "--dry-run" in help_text:
            args.append("--dry-run")
        else:
            args = ["lock", "--check", "--python", "3.11.12", "-v"]
        result = subprocess.run(
            command + args, cwd=context, capture_output=True, text=True, timeout=60
        )
        print(result.stdout)
        print(result.stderr)
        if result.returncode:
            raise SystemExit(result.returncode)
        exported = context / "requirements-modal.txt"
        subprocess.run(
            command
            + [
                "export",
                "--locked",
                "--extra",
                "gpu",
                "--no-dev",
                "--no-emit-project",
                "-o",
                str(exported),
            ],
            cwd=context,
            check=True,
            capture_output=True,
            timeout=30,
        )

        def body(path: Path) -> str:
            return "\n".join(
                line for line in path.read_text().splitlines() if not line.startswith("#")
            )

        if body(exported) != body(root / "requirements-modal.txt"):
            raise ValueError(
                "requirements-modal.txt differs from uv.lock; regenerate it with uv export"
            )
        subprocess.run(
            command + ["venv", "--python", "3.11.12", str(context / ".venv")],
            check=True,
            capture_output=True,
            timeout=30,
        )
        result = subprocess.run(
            command
            + [
                "pip",
                "install",
                "--python",
                str(context / ".venv" / "bin" / "python"),
                "--require-hashes",
                "--dry-run",
                "-r",
                str(exported),
            ],
            cwd=context,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode:
            print(result.stderr)
            raise SystemExit(result.returncode)
        print("Hashed Modal export matches uv.lock; Python 3.11.12 install dry run passed.")


if __name__ == "__main__":
    main()
