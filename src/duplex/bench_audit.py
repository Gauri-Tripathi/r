"""Paired loss and speech diagnostics from saved run artifacts."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .evaluate import import_asr
from .io import atomic_json, digest, file_hash, read_json, read_jsonl


def paired_loss(
    base: list[dict[str, Any]], adapted: list[dict[str, Any]], key: str, pad_weight: float = 0.1
) -> dict[str, Any]:
    before = {r["id"]: r for r in base}
    after = {r["id"]: r for r in adapted}
    if (
        len(before) != len(base)
        or len(after) != len(adapted)
        or before.keys() != after.keys()
        or not before
    ):
        raise ValueError("Loss cases are missing, duplicated, or unmatched")
    groups: dict[str, list[float]] = defaultdict(list)
    deltas = []
    weights = []
    for identifier, b in before.items():
        a = after[identifier]
        if a["conversation"] != b["conversation"] or any(
            a[k] != b[k] for k in ("valid_audio_tokens", "valid_text_tokens", "real_text_tokens")
        ):
            raise ValueError("Evaluation groups/masks differ")
        delta = float(a[key] - b[key])
        deltas.append(delta)
        groups[b["conversation"]].append(delta)
        weights.append(
            b["valid_audio_tokens"]
            if key == "audio_loss"
            else b["real_text_tokens"]
            + pad_weight * (b["valid_text_tokens"] - b["real_text_tokens"])
        )
    b_values = [before[k][key] for k in before]
    a_values = [after[k][key] for k in before]
    b_mean, a_mean = float(np.mean(b_values)), float(np.mean(a_values))
    ci = None
    if len(groups) >= 5:
        rng = np.random.default_rng(101)
        clustered = list(groups.values())
        estimates = [
            np.mean(
                [v for i in rng.integers(0, len(clustered), len(clustered)) for v in clustered[i]]
            )
            for _ in range(2000)
        ]
        ci = np.quantile(estimates, [0.025, 0.975]).tolist()
    return {
        "cases": len(before),
        "independent_conversations": len(groups),
        "base_clip_mean": b_mean,
        "adapted_clip_mean": a_mean,
        "relative_change_percent": 100 * (a_mean - b_mean) / b_mean,
        "base_token_weighted": float(np.average(b_values, weights=weights)),
        "adapted_token_weighted": float(np.average(a_values, weights=weights)),
        "improved_cases": sum(x < -1e-8 for x in deltas),
        "worsened_cases": sum(x > 1e-8 for x in deltas),
        "tied_cases": sum(abs(x) <= 1e-8 for x in deltas),
        "median_paired_delta": float(np.median(deltas)),
        "paired_cluster_ci95": ci,
        "ci_reason": None
        if ci is not None
        else "Fewer than five independent conversations; no significance/generalization claim",
    }


def word_error(reference: str, hypothesis: str) -> float | None:
    def words(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    a, b = words(reference), words(hypothesis)
    if not a:
        return None
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1] / len(a)


def audio_metrics(wav: Path, metadata: dict[str, Any], transcript: Any = None) -> dict[str, Any]:
    pcm, sr = sf.read(wav, dtype="float32")
    if (
        pcm.ndim != 1
        or sr != 24000
        or not np.isfinite(pcm).all()
        or file_hash(wav) != metadata["generated_audio_sha256"]
    ):
        raise ValueError("Invalid generated waveform/checksum")
    framed = np.pad(pcm, (0, (-len(pcm)) % 480)).reshape(-1, 480)
    energy = np.sqrt(np.mean(framed**2, axis=1))
    active = np.flatnonzero(energy >= 0.005)
    result = {
        "duration_s": len(pcm) / sr,
        "rtf": metadata["rtf"],
        "energy_silence_fraction": float(np.mean(energy < 0.005)),
        "first_energy_s": float(active[0] * 0.02) if len(active) else None,
        "rms": float(np.sqrt(np.mean(pcm**2))),
        "clipped_fraction": float(np.mean(np.abs(pcm) >= 0.999)),
        "truncated": metadata["truncated"],
        "audio_sha256": metadata["generated_audio_sha256"],
    }
    if transcript is not None:
        if transcript.audio_sha256 != metadata["generated_audio_sha256"]:
            raise ValueError("ASR belongs to different generated audio")
        result.update(
            asr_words=len(transcript.words),
            asr_uncertainty=transcript.uncertainty,
            internal_text_asr_wer=word_error(metadata["internal_text_proxy"], transcript.text),
            asr_text=transcript.text,
        )
    return result


def audit(
    run: Path, prior: Path, data_root: Path, out: Path, asr: Path | None = None
) -> dict[str, Any]:
    result = read_json(run / "results.json")
    identity = read_json(run / "run-identity.json")
    old = read_json(prior / "results.json")
    raw = result["per_example_test"]
    losses = {
        key: paired_loss(raw["base"], raw["adapted"], key, identity["train"]["text_pad_weight"])
        for key in ("audio_loss", "text_loss")
    }
    full_splits = read_json(data_root / "prepared-input.json")["conversation_splits"]
    old_splits = {r["conversation"]: r["split"] for r in old["dataset"]["conversations"]}
    clean = [
        key
        for key, split in full_splits.items()
        if split == "test" and old_splits.get(key) == "test"
    ]
    prior_comparison = {
        "direct_pilot_vs_full_ranking_valid": False,
        "common_untouched_conversations": clean,
        "full_test_prior_roles": {
            key: old_splits.get(key) for key, split in full_splits.items() if split == "test"
        },
        "reason": "Pilot/full used different windows and splits; full test conversation previously participated in pilot development selection. A fresh common suite is required.",
    }
    transcripts = {
        label: import_asr(asr / f"{label}.jsonl")
        if asr is not None and (asr / f"{label}.jsonl").exists()
        else {}
        for label in ("base", "full")
    }
    sources = {r["id"]: r for r in read_jsonl(run / "locked-generation-inputs.jsonl")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    speech = []
    blind = []
    cards = []
    for index, identifier in enumerate(sources):
        metadata = [
            read_json(run / folder / identifier / "generation.json")
            for folder in ("base-audio", "adapted-audio")
        ]
        for key in (
            "input_fingerprint",
            "user_audio_sha256",
            "seed",
            "decoding",
            "voice_sha256",
            "runtime_fingerprint",
            "system_prompt",
            "base_revision",
            "duration_s",
        ):
            if metadata[0][key] != metadata[1][key]:
                raise ValueError(f"Unmatched speech settings: {key}")
        if metadata[0]["adapter_sha256"] is not None or metadata[1]["adapter_sha256"] is None:
            raise ValueError("Require adapter-disabled base and adapted speech")
        paths = [
            run / folder / identifier / "generated.wav"
            for folder in ("base-audio", "adapted-audio")
        ]
        signals = [sf.read(p, dtype="float32")[0] for p in paths]
        metrics = {
            label: audio_metrics(path, meta, transcripts[label].get(identifier))
            for label, path, meta in zip(("base", "full"), paths, metadata, strict=True)
        }
        speech.append(
            {"id": identifier, "pcm_changed": not np.array_equal(*signals), "metrics": metrics}
        )
        source = data_root / sources[identifier]["user"]
        if file_hash(source) != metadata[0]["user_audio_sha256"]:
            raise ValueError("Original user waveform changed")
        user, sr = sf.read(source, dtype="float32")
        order = sorted(("base", "full"), key=lambda label: digest([101, identifier, label]))
        mapping = {}
        for letter, label in zip(("A", "B"), order, strict=True):
            assistant = signals[0 if label == "base" else 1]
            if len(user) != len(assistant) or sr != 24000:
                raise ValueError("Listening pair timelines differ")
            sf.write(
                out / "audio" / f"case-{index}-{letter}.wav",
                np.column_stack((user, assistant)),
                24000,
                subtype="FLOAT",
            )
            mapping[letter] = label
        blind.append({"case": f"case-{index}", "source_id": identifier, "mapping": mapping})
        cards.append(
            f'<article data-case="case-{index}"><h2>Case {index + 1}</h2><p>Left channel: user. Right channel: assistant. Same timeline.</p>'
            + "".join(
                f'<label>{letter}</label><audio controls preload="none" src="audio/case-{index}-{letter}.wav"></audio><label>Intelligibility {letter} (1–5)</label><select name="intelligibility{letter}"><option value="">Choose</option>'
                + "".join(f"<option>{n}</option>" for n in range(1, 6))
                + "</select><label>Relevance "
                + letter
                + ' (1–5)</label><select name="relevance'
                + letter
                + '"><option value="">Choose</option>'
                + "".join(f"<option>{n}</option>" for n in range(1, 6))
                + "</select>"
                for letter in ("A", "B")
            )
            + '<label>Overall preference</label><select name="preference"><option value="">Choose</option><option>A</option><option>B</option><option>Tie</option></select></article>'
        )
    protocol = digest(
        {
            "source_result_sha256": file_hash(run / "results.json"),
            "speech": speech,
            "blind_order": blind,
        }
    )
    report = {
        "kind": "saved_artifact_benchmark",
        "losses": losses,
        "prior_comparison": prior_comparison,
        "speech": speech,
        "protocol_sha256": protocol,
        "human_ratings_available": False,
        "limitations": [
            "Only one independent full-test conversation; speakers overlap training.",
            "ASR/internal-text agreement is not answer correctness or a calibrated intelligibility score.",
            "Energy activity is not a speech detector; overlap can be appropriate in duplex dialogue.",
            "Generation order/warmup were not controlled for the old saved speech; RTF differences are descriptive.",
            "No response-latency or official Full-Duplex-Bench score is inferred from clipped conversations.",
        ],
    }
    atomic_json(out / "benchmark.json", report)
    atomic_json(out / "blind-key.json", {"protocol_sha256": protocol, "cases": blind})
    script = """<script>function saveRatings(){const ratings=[...document.querySelectorAll('article')].map(a=>({case:a.dataset.case,...Object.fromEntries([...a.querySelectorAll('select')].map(s=>[s.name,s.value]))}));if(ratings.some(r=>Object.values(r).some(v=>!v))){alert('Rate every case first');return;}const data={protocol_sha256:PROTOCOL,ratings};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.download='ratings.json';a.click();}</script>""".replace(
        "PROTOCOL", json.dumps(protocol)
    )
    (out / "blind-listening.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Blind speech comparison</title><style>body{font:16px system-ui;max-width:850px;margin:40px auto;padding:20px;background:#f4f6f9}article{background:white;padding:24px;margin:24px 0;border-radius:12px}audio{display:block;width:100%}label{display:block;margin-top:12px}select{padding:6px}button{padding:12px}</style><h1>Blind A/B speech comparison</h1><p>Model labels are hidden. Listen to both versions with headphones, then rate intelligibility, relevance and preference. These three clips are descriptive, not a population-level quality estimate.</p>'
        + "".join(cards)
        + '<button onclick="saveRatings()">Download ratings</button>'
        + script,
        encoding="utf-8",
    )
    lines = [
        "# Saved-artifact benchmark",
        "",
        "| Held-out native loss | Base | Full adapter | Change | Improved cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, metric in losses.items():
        lines.append(
            f"| {key} | {metric['base_clip_mean']:.8f} | {metric['adapted_clip_mean']:.8f} | {metric['relative_change_percent']:+.3f}% | {metric['improved_cases']}/{metric['cases']} |"
        )
    lines += [
        "",
        "Predictive loss improves on this held-out recording. There is only one independent test conversation: no confidence interval, significance or broad generalization claim.",
        "",
        "Pilot versus full cannot be ranked from old scores: the test windows differ and the full test recording was used for pilot development selection. No untouched common OpenYap conversation remains.",
        "",
        "Three speech pairs were checksum/settings/PCM checked. Use blind-listening.html and export ratings; no human score is invented. See benchmark.json for audio/ASR diagnostics and limitations.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def score_ratings(directory: Path, ratings_file: Path) -> dict[str, Any]:
    key = read_json(directory / "blind-key.json")
    submitted = read_json(ratings_file)
    if submitted["protocol_sha256"] != key["protocol_sha256"]:
        raise ValueError("Ratings belong to another benchmark")
    mapping = {r["case"]: r["mapping"] for r in key["cases"]}
    if len(submitted["ratings"]) != len(mapping) or {
        r["case"] for r in submitted["ratings"]
    } != set(mapping):
        raise ValueError("Rate every case exactly once")
    wins = {"base": 0, "full": 0, "Tie": 0}
    scores = {
        label: {metric: [] for metric in ("intelligibility", "relevance")}
        for label in ("base", "full")
    }
    for row in submitted["ratings"]:
        labels = mapping[row["case"]]
        choice = row["preference"]
        if choice not in ("A", "B", "Tie"):
            raise ValueError("Invalid preference")
        wins["Tie" if choice == "Tie" else labels[choice]] += 1
        for letter, model in labels.items():
            for metric in scores[model]:
                value = int(row[metric + letter])
                if not 1 <= value <= 5:
                    raise ValueError("Ratings must be 1–5")
                scores[model][metric].append(value)
    result = {
        "cases": len(mapping),
        "preferences": wins,
        "mean_scores": {
            model: {k: float(np.mean(v)) for k, v in metrics.items()}
            for model, metrics in scores.items()
        },
        "evidence": "one submitted blind listening assessment; descriptive only",
    }
    atomic_json(directory / "human-ratings-result.json", result)
    return result
