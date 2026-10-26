"""Flush shell mounts without the Function-only Volume.commit RPC.

Modal shell uses a managed Sandbox. Volume.commit() is not available in that
process. `sync MOUNT` explicitly commits v2 Volumes; v1 additionally relies on
Modal's documented background/final-shutdown commits. Verify persisted files
from a fresh reader before treating a checkpoint as recovered.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

MOUNTS = ("/data", "/runs", "/cache")


def flush_mounts() -> None:
    if os.environ.get("DUPLEX_MODAL") != "1":
        raise ValueError("Run volume flushing inside the project Modal shell")
    if any(not Path(path).is_dir() for path in MOUNTS):
        raise ValueError("Expected /data, /runs and /cache mounts are missing")
    subprocess.run(["sync", *MOUNTS], check=True, timeout=20)


def main() -> None:
    flush_mounts()
    print(
        "Mount sync completed. v1 Volumes use Modal background/shutdown commits; verify persisted files from outside this shell."
    )


if __name__ == "__main__":
    main()
