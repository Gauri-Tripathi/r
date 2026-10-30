"""Fetch the pinned public sample explicitly; never downloads the 1,000-hour corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "TheAgenticDataCompany/open-yap-1k"
REVISION = "a08b1c80e50159d462ddcbbabfdea5bf7ccfc626"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="store_true")
    args = parser.parse_args()
    root = Path("data/openyap/source")
    files = ["README.md", "LICENSE.txt", "metadata.jsonl", "verification.json"]
    if args.audio:
        files += ["shard-00000.tar", "shard-00001.tar"]
    for name in files:
        print(f"Fetching pinned public sample: {name}", flush=True)
        try:
            path = hf_hub_download(
                REPO, name, repo_type="dataset", revision=REVISION, local_dir=root, token=False
            )
        except Exception as exc:
            raise SystemExit(
                f"Public sample download failed ({type(exc).__name__}); no fallback or retry was launched"
            ) from None
        print(f"Available: {Path(path).name}, {Path(path).stat().st_size} bytes", flush=True)


if __name__ == "__main__":
    main()
