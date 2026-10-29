"""Read-only, bounded download of executed smoke artifacts; never creates compute."""

from __future__ import annotations

from pathlib import Path

import modal

from duplex.io import atomic_json, confined, file_hash


def main() -> None:
    root = Path("runs/audit-2026-11-01/remote")
    inventory = []
    total = 0
    for volume_name, prefixes in {
        "duplex-runs": ["pilot"],
        "duplex-data": ["train", "dev", "encoded"],
        "duplex-cache": ["assets/assets.json"],
    }.items():
        volume = modal.Volume.from_name(volume_name)
        for prefix in prefixes:
            for entry in volume.listdir(prefix, recursive=True):
                if entry.type.name != "FILE":
                    continue
                if entry.size > 64 * 1024 * 1024:
                    raise ValueError("Audit refuses oversized file")
                total += entry.size
                if total > 256 * 1024 * 1024:
                    raise ValueError("Audit exceeds bounded download allowance")
                relative = f"{volume_name}/{entry.path.lstrip('/')}"
                dest = confined(root, relative)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.stat().st_size != entry.size:
                    temporary = dest.with_suffix(dest.suffix + ".partial")
                    actual = 0
                    with temporary.open("wb") as stream:
                        for chunk in volume.read_file(entry.path):
                            actual += len(chunk)
                            if actual > entry.size or actual > 64 * 1024 * 1024:
                                raise ValueError("Artifact exceeded declared size")
                            stream.write(chunk)
                    if actual != entry.size:
                        raise ValueError("Artifact size changed during read")
                    temporary.replace(dest)
                inventory.append(
                    {
                        "volume": volume_name,
                        "path": entry.path,
                        "size": entry.size,
                        "mtime": entry.mtime,
                        "sha256": file_hash(dest),
                    }
                )
    atomic_json(root / "inventory.json", {"read_only": True, "files": inventory, "bytes": total})
    print(f"Read {len(inventory)} files ({total / 1024**2:.1f} MiB); no compute allocated.")


if __name__ == "__main__":
    main()
