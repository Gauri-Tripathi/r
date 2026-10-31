"""Read-only check of saved full-run training evidence; never allocates compute."""

import json
from pathlib import Path

import modal

from duplex.io import atomic_json

root = Path("runs/openyap/full-v1")
volume = modal.Volume.from_name("duplex-runs")
entries = {e.path.lstrip("/") for e in volume.listdir("openyap/full-v1", recursive=False)}
output = {}
for name in ("run-status.json", "results.json", "training.jsonl", "best.json", "run-identity.json"):
    remote = "openyap/full-v1/" + name
    if remote not in entries:
        continue
    payload = b"".join(volume.read_file(remote))
    if len(payload) > 8 * 1024**2:
        raise ValueError("Oversized audit file")
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(payload)
    if name == "training.jsonl":
        rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
        output["training_logs"] = {
            "steps": len(rows),
            "first_step": rows[0]["step"] if rows else None,
            "last_step": rows[-1]["step"] if rows else None,
            "example_exposures": sum(r.get("examples", 0) for r in rows),
            "audio_exposure_s": sum(r.get("audio_seconds", 0) for r in rows),
        }
    elif name == "results.json":
        result = json.loads(payload)
        output["results"] = {
            k: result.get(k)
            for k in (
                "kind",
                "optimizer_steps",
                "unique_train_examples",
                "unique_train_audio_s",
                "audio_exposure_s",
                "effective_epochs",
                "selected_checkpoint",
                "base_test",
                "adapted_test",
                "test_loss_delta",
            )
        }
        output["dataset"] = {
            k: result.get("dataset", {}).get(k)
            for k in ("source_hours", "selected", "source_audio_s", "audio_s", "split_mode")
        }
    elif name != "run-identity.json":
        output[name] = json.loads(payload)
output["remote_files"] = sorted(entries)
atomic_json(root / "training-check.json", output)
print(json.dumps(output, indent=2))
