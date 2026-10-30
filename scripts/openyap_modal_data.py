"""Transfer only selected pilot artifacts to/from the user's existing Modal volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from duplex.io import atomic_json, confined, file_hash, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["upload", "pull", "status"])
    parser.add_argument("--root", type=Path, default=Path("data/openyap/pilot-v1"))
    parser.add_argument("--remote", default="openyap/pilot-v1")
    parser.add_argument("--download-dir", type=Path, default=Path("runs/openyap/pilot-v1"))
    args = parser.parse_args()
    from pathlib import PurePosixPath

    remote = str(PurePosixPath(args.remote))
    if not remote.startswith("openyap/") or ".." in PurePosixPath(remote).parts or "\\" in remote:
        raise ValueError("Use an isolated relative prefix under openyap/")
    if args.action == "status":
        print({"project_gpu_lease": modal.Dict.from_name("duplex-gpu-reservation").get("active")})
        volume = modal.Volume.from_name("duplex-runs")
        files = [
            e.path
            for e in volume.listdir("openyap/pilot-v1", recursive=True)
            if e.type.name == "FILE"
        ]
        print(
            {
                "checkpoints": sum(p.endswith("complete.json") for p in files),
                "adapted_clips": sum(
                    "adapted-audio/" in p and p.endswith("generation.json") for p in files
                ),
                "results_ready": "openyap/pilot-v1/results.json" in files,
                "export_ready": "openyap/pilot-v1/export/release.json" in files,
            }
        )
        if "openyap/pilot-v1/results.json" in files:
            import json

            result = json.loads(b"".join(volume.read_file("openyap/pilot-v1/results.json")))
            atomic_json(Path("runs/openyap/pilot-v1/results.json"), result)
            print(
                {
                    key: result[key]
                    for key in (
                        "optimizer_steps",
                        "wall_s",
                        "base_test",
                        "adapted_test",
                        "test_loss_delta",
                        "generated_audio_changed",
                    )
                }
            )
        return
    if args.action == "upload":
        root = args.root
        prepared = read_json(root / "prepared.json")
        if file_hash(root / "replay.jsonl") != prepared["manifest_sha256"]:
            raise ValueError("Selected manifest changed")
        files = ["prepared.json", "prepared-input.json", "replay.jsonl"]
        for row in read_jsonl(root / "replay.jsonl"):
            files += [row["user"], row["assistant"], f"audio/{row['id']}/manifest.json"]
        volume = modal.Volume.from_name("duplex-data")
        with volume.batch_upload(force=False) as upload:
            for name in files:
                upload.put_file(confined(root, name), f"{remote}/{name}")
        print(
            {
                "uploaded_files": len(files),
                "bytes": sum(confined(root, n).stat().st_size for n in files),
                "source_shards_uploaded": False,
            }
        )
        return
    volume = modal.Volume.from_name("duplex-runs")
    destination = args.download_dir
    inventory = []
    total = 0
    for entry in volume.listdir(remote, recursive=True):
        if entry.type.name != "FILE":
            continue
        total += entry.size
        if entry.size > 64 * 1024**2 or total > 256 * 1024**2:
            raise ValueError("Result transfer exceeded bounded allowance")
        relative = entry.path.lstrip("/").removeprefix(remote + "/")
        path = confined(destination, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            size = 0
            for block in volume.read_file(entry.path):
                size += len(block)
                if size > entry.size:
                    raise ValueError("Remote file changed while reading")
                stream.write(block)
        if size != entry.size:
            raise ValueError("Incomplete result transfer")
        inventory.append({"file": relative, "bytes": size, "sha256": file_hash(path)})
    atomic_json(destination / "download-inventory.json", inventory)
    print({"downloaded_files": len(inventory), "bytes": total})


if __name__ == "__main__":
    main()
