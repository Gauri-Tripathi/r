"""Inspect executed artifacts offline. No weights/model download or generation is run."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from safetensors.torch import load_file

from duplex import checkpoint
from duplex.data import load_examples, reference
from duplex.io import atomic_json, digest, file_hash, read_json, read_jsonl


def main() -> None:
    torch.set_num_threads(1)
    root = Path("runs/audit-2026-11-01/remote")
    pilot = root / "duplex-runs/pilot"
    data = root / "duplex-data"
    completed = sorted((pilot / "checkpoints").glob("step-*"))
    if len(completed) < 2:
        raise ValueError("Need two actual completed training checkpoints to audit change")
    first, last = completed[-2:]
    metadata = checkpoint.complete(last)
    if checkpoint.complete(first)["identity"] != metadata["identity"]:
        raise ValueError("Checkpoint identities differ; cannot claim one continuous run")
    before, after = (
        load_file(str(first / "adapter.safetensors")),
        load_file(str(last / "adapter.safetensors")),
    )
    changes = []
    for name, tensor in after.items():
        delta = tensor.float() - before[name].float()
        changes.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "parameters": tensor.numel(),
                "changed_elements": int(torch.count_nonzero(delta)),
                "max_abs_change": float(delta.abs().max()),
                "l2_change": float(delta.norm()),
            }
        )
    effective = {}
    projection_change = {}
    for name in after:
        if name.endswith(".a"):
            a, b = after[name].float(), after[name[:-1] + "b"].float()
            squared = torch.trace((a @ a.T) @ (b.T @ b)).clamp_min(0)
            effective[name[:-2]] = (
                math.sqrt(float(squared))
                * metadata["train_config"]["alpha"]
                / metadata["train_config"]["rank"]
            )
            a1, b1 = a.double(), b.double()
            a0, b0 = before[name].double(), before[name[:-1] + "b"].double()
            norm2 = (
                torch.trace((a1 @ a1.T) @ (b1.T @ b1))
                + torch.trace((a0 @ a0.T) @ (b0.T @ b0))
                - 2 * torch.trace((a1 @ a0.T) @ (b0.T @ b1))
            )
            projection_change[name[:-2]] = (
                math.sqrt(max(0, float(norm2)))
                * metadata["train_config"]["alpha"]
                / metadata["train_config"]["rank"]
            )
    state = torch.load(last / "state.pt", map_location="cpu", weights_only=True)
    history = read_jsonl(pilot / "metrics.jsonl")
    training = [r for r in history if "audio_loss" in r]
    modality_logs = {
        key: {
            "first": training[0].get(key),
            "last": training[-1].get(key),
            "min": min(float(r[key]) for r in training),
            "max": max(float(r[key]) for r in training),
        }
        for key in (
            "loss",
            "audio_loss",
            "text_loss",
            "grad_norm",
            "audio_seconds",
            "valid_audio_tokens",
            "valid_text_tokens",
        )
    }
    summaries = {}
    for split in ("train", "dev"):
        rows = load_examples(data / split / "rendered.jsonl")
        audits = []
        physical = set()
        for row in rows:
            audio_dir = data / split / "audio" / row.id
            timeline = read_json(audio_dir / "manifest.json")
            sample_path = data / "encoded" / f"{row.id}.safetensors"
            sample_meta = read_json(sample_path.with_suffix(".json"))
            tensors = load_file(str(sample_path))
            assistant, sr_a = sf.read(audio_dir / "assistant.wav", dtype="float32")
            user, sr_u = sf.read(audio_dir / "user.wav", dtype="float32")
            hashes = [file_hash(audio_dir / name) for name in ("user.wav", "assistant.wav")]
            physical.add(tuple(hashes))
            p = int(tensors["prompt_frames"])
            masks = tensors["valid"]
            actual_overlap = False
            correction = next((u for u in row.script if u.kind == "correction"), None)
            if correction:
                start = round(timeline["correction_start_s"] * sr_a)
                end = round(timeline["correction_end_s"] * sr_a)
                actual_overlap = bool(
                    np.any(
                        (np.abs(user[start:end]) > 0.003) & (np.abs(assistant[start:end]) > 0.003)
                    )
                )
            accepted_context_frames = sum(
                e.get("frames", 0) for e in timeline.get("context", []) if e.get("accepted")
            )
            assistant_valid = int(masks[0, 1].sum())
            audits.append(
                {
                    "id": row.id,
                    "group": row.group_id,
                    "family": row.family,
                    "timing": row.timing,
                    "duration_s": len(user) / sr_u,
                    "sample_rate": sr_u,
                    "same_timeline": len(user) == len(assistant) and sr_a == sr_u == 24000,
                    "finite": bool(np.isfinite(user).all() and np.isfinite(assistant).all()),
                    "clipped": bool(max(np.max(np.abs(user)), np.max(np.abs(assistant))) >= 0.999),
                    "audio_hashes_match_manifest": hashes
                    == [timeline["user_sha256"], timeline["assistant_sha256"]],
                    "encoded_checksum_valid": file_hash(sample_path) == sample_meta["sha256"],
                    "codes_shape": list(tensors["codes"].shape),
                    "prompt_frames": p,
                    "user_loss_targets": int(masks[:, 9:].sum()),
                    "prompt_loss_targets": int(masks[..., :p].sum()),
                    "context_frames": accepted_context_frames,
                    "masked_nonprompt_frames": int((~masks[0, 0, p:]).sum()),
                    "assistant_valid_frames": assistant_valid,
                    "expected_native_audio_tokens": 8 * assistant_valid - 7,
                    "correction_present": correction is not None,
                    "waveform_overlap_during_correction": actual_overlap,
                    "correction_end_s": timeline["correction_end_s"],
                    "response_start_s": timeline["response_start_s"],
                    "response_after_correction": timeline["response_start_s"]
                    >= timeline["correction_end_s"],
                    "events": timeline["events"],
                    "context_delivered_before_response": all(
                        e["at_s"] + e.get("frames", 0) / 12.5 <= timeline["response_start_s"]
                        for e in timeline.get("context", [])
                        if e.get("accepted")
                    ),
                    "tool_arrives_during_prefix": any(
                        timeline["prefix_start_s"] <= e["arrival_s"] < timeline["prefix_end_s"]
                        for e in timeline["events"]
                    ),
                    "alignment": timeline["alignment"],
                    "tts_version": timeline["renderer_version"],
                    "supervision_sha256": digest(row.supervision),
                    "matches_deterministic_reference": row.supervision == reference(row.truth),
                    "provenance": row.provenance,
                }
            )
        summaries[split] = {
            "examples": len(rows),
            "groups": len({r.group_id for r in rows}),
            "joint_audio_pairs": len(physical),
            "audio_s": sum(x["duration_s"] for x in audits),
            "unique_supervision_texts": len({r.supervision for r in rows}),
            "rows": audits,
        }
    generation = {}
    for p in pilot.glob("*/generation.json"):
        raw = read_json(p)
        generation[p.parent.name] = {
            k: v for k, v in raw.items() if k not in ("internal_text_proxy", "text_tokens")
        }
        generation[p.parent.name]["internal_text_sha256"] = digest(raw["internal_text_proxy"])
        generation[p.parent.name]["text_token_count"] = len(raw["text_tokens"])
    result = {
        "classification": "smoke_test",
        "source_inventory_sha256": file_hash(root / "inventory.json"),
        "base_revision": metadata["identity"]["base_revision"],
        "identity": metadata["identity"],
        "checkpoint_comparison": {"from": first.name, "to": last.name},
        "adapter_tensors": changes,
        "trainable_count_from_weights": sum(t.numel() for t in after.values()),
        "effective_projection_delta_frobenius": effective,
        "effective_projection_change_75_to_100_frobenius": projection_change,
        "optimizer": {
            "steps_in_state": sorted(
                {int(s["step"]) for s in state["optimizer"]["state"].values()}
            ),
            "param_groups": state["optimizer"]["param_groups"],
        },
        "sampler_position": state["sampler"],
        "training_log_steps": [r["step"] for r in training],
        "losses": modality_logs,
        "data": summaries,
        "logged_audio_exposure_s": sum(r["audio_seconds"] for r in training),
        "effective_epochs_by_audio": sum(r["audio_seconds"] for r in training)
        / summaries["train"]["audio_s"],
        "generation_artifacts": generation,
        "missing_evidence": [
            "No initial step-zero adapter checkpoint; step-75 to step-100 change is verified, not exact initialization-to-final change.",
            "No per-modality per-adapter GPU gradients retained. Combined gradient norm is logged; separate audio-gradient reach is only a tiny upstream CPU test.",
            "No matched base/adapter full generation pair or ASR transcription. Existing smoke generations cannot establish audible improvement or adapter output sensitivity.",
            "Native delay array was not serialized at runtime. Referenced pinned native code uses [0,0,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1].",
            "Word timings are construction boundaries, not natural-speech forced alignment; no independent listening/ASR alignment audit.",
        ],
    }
    atomic_json(Path("docs/smoke-audit.json"), result)
    print(
        {
            "trainable": result["trainable_count_from_weights"],
            "changed_tensors": sum(r["changed_elements"] > 0 for r in changes),
            "train": summaries["train"]["examples"],
            "epochs": result["effective_epochs_by_audio"],
            "losses": modality_logs,
        }
    )


if __name__ == "__main__":
    main()
