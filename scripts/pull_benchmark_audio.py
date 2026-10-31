"""Read-only bounded fetch of the existing full-run speech benchmark artifacts."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal

from duplex.io import atomic_json, confined, file_hash

volume = modal.Volume.from_name("duplex-runs")
prefix = "openyap/full-v1/"
root = Path("runs/openyap/full-v1")
entries = [e for e in volume.listdir(prefix, recursive=True) if e.type.name == "FILE"]
entries = [
    e
    for e in entries
    if any(marker in e.path for marker in ("/base-audio/", "/adapted-audio/"))
    or e.path.endswith("/locked-generation-inputs.jsonl")
]
if sum(e.size for e in entries) > 128 * 1024**2:
    raise ValueError("Speech artifact transfer exceeds bound")


def fetch(entry):
    relative = entry.path.lstrip("/").removeprefix(prefix)
    path = confined(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size != entry.size:
        size = 0
        with path.open("wb") as stream:
            for block in volume.read_file(entry.path):
                size += len(block)
                if size > entry.size:
                    raise ValueError("File size changed")
                stream.write(block)
        if size != entry.size:
            raise ValueError("Incomplete speech artifact")
    return {"file": relative, "size": entry.size, "sha256": file_hash(path)}


with ThreadPoolExecutor(max_workers=4) as pool:
    inventory = list(pool.map(fetch, entries))
atomic_json(root / "speech-download-inventory.json", inventory)
print({"files": len(inventory), "bytes": sum(r["size"] for r in inventory)})
