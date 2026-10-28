"""Small fail-closed secret scanner for tracked/reviewable text, with no payload logging."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\b(?:hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9_-]{32,}|AKIA[A-Z0-9]{16})\b"),
    re.compile(
        r"(?i)(?:api_key|access_token|token_secret|password)\s*[:=]\s*['\"][A-Za-z0-9/+_-]{24,}['\"]"
    ),
]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # Git inventory avoids ignored raw data, caches, virtualenvs, weights and .git internals.
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=root,
        capture_output=True,
        check=True,
        timeout=30,
    )
    failed = []
    for name in result.stdout.decode().split("\0"):
        if not name:
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path.is_symlink():
            raise ValueError("Unsafe scanner path")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"Oversized review file; inspect explicitly: {name}")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"Unexpected binary in release sources: {name}") from None
        if any(pattern.search(text) for pattern in PATTERNS):
            failed.append(name)
    if failed:
        raise SystemExit("Potential secret in: " + ", ".join(failed))
    print(
        "Secret scan passed for reviewable repository text; absence of findings is not proof of absence."
    )


if __name__ == "__main__":
    main()
