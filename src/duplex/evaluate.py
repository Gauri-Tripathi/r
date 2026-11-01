"""Speech scoring with timestamped ASR and grouped uncertainty estimates."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field

from .data import check_target, normalize
from .io import atomic_json, file_hash, read_jsonl
from .schema import Example, Strict, Word


class Transcript(Strict):
    id: str
    audio_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str = Field(max_length=4000)
    words: list[Word] = Field(max_length=800)
    uncertainty: float = Field(ge=0, le=1)
    engine: str = Field(min_length=1, max_length=128)
    revision: str = Field(min_length=1, max_length=128)


def import_asr(path: Path) -> dict[str, Transcript]:
    result = {}
    for raw in read_jsonl(path):
        row = Transcript.model_validate(raw)
        if row.id in result:
            raise ValueError("Duplicate ASR ID")
        result[row.id] = row
    return result


def score(row: Example, text: str, words: list[Word], available_s: float) -> dict[str, Any]:
    if words:
        text = " ".join(w.text for w in words if w.start_s >= available_s)
    n = normalize(text)
    r = row.truth.final
    errors = check_target(text, row.truth)
    present = {
        k: bool(re.search(rf"\b{getattr(r, k).lower()}\b", n)) for k in ("person", "day", "hour")
    }
    stale_values = [
        getattr(row.truth.initial, k)
        for k in ("person", "day", "hour")
        if getattr(row.truth.initial, k) != getattr(r, k)
    ]
    superseded = any(re.search(rf"\b{v.lower()}\b", n) for v in stale_values)
    correction = all(present.values()) and not superseded
    preserve = all(present[k] for k in row.truth.unchanged)
    commitment = bool(re.search(r"\b(booked|reserved|scheduled|cancelled|canceled|done)\b", n))
    hallucination = bool(re.search(r"\bis available\b", n)) and row.truth.action != "offer"
    incorporation = None
    # First word-ending time at which a suffix is a complete valid new response.
    for index, word in enumerate(words):
        prefix = " ".join(w.text for w in words[: index + 1] if w.start_s >= available_s)
        if not check_target(prefix, row.truth):
            incorporation = max(0, word.end_s - available_s)
            break
    return {
        "id": row.id,
        "group_id": row.group_id,
        "timing": row.timing,
        "family": row.family,
        "axes": row.axes,
        "correction_success": correction,
        "joint_success": correction and not errors,
        "unchanged_preserved": preserve,
        "latest_fact_success": not errors,
        "stale_fact_usage": any(e.version < row.truth.latest_version for e in row.visible.events)
        and hallucination,
        "unsupported_commitment": commitment,
        "hallucinated_availability": hallucination,
        "appropriate_clarification": row.truth.action == "clarify" and not errors,
        "clarification_required": row.truth.action == "clarify",
        "incorporation_latency_s": incorporation,
        "latency_definition": "first complete valid response suffix after max(correction end, latest relevant event arrival), measured at word end",
        "repetition": len(n.split()) > 6 and len(set(n.split())) / len(n.split()) < 0.35,
        "errors": errors,
    }


def bootstrap_delta(
    base: list[dict[str, Any]],
    adapted: list[dict[str, Any]],
    key: str,
    seed: int = 17,
    n: int = 2000,
) -> dict[str, Any]:
    if {r["id"] for r in base} != {r["id"] for r in adapted}:
        raise ValueError("Paired evaluation IDs differ")
    lookup = {r["id"]: r for r in adapted}
    # Cluster resample by counterfactual group; variants are not independent samples.
    groups: dict[str, list[float]] = {}
    for row in base:
        groups.setdefault(row["group_id"], []).append(
            float(lookup[row["id"]][key]) - float(row[key])
        )
    values = list(groups.values())
    delta = float(np.mean([x for group in values for x in group]))
    if len(values) < 5:
        return {
            "delta": delta,
            "ci95": None,
            "groups": len(values),
            "reason": "fewer than five independent groups",
        }
    rng = np.random.default_rng(seed)
    means = [
        np.mean([x for i in rng.integers(0, len(values), len(values)) for x in values[i]])
        for _ in range(n)
    ]
    return {
        "delta": delta,
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "groups": len(values),
        "bootstrap": "paired group-cluster percentile, 2000 draws",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "correction_success",
        "joint_success",
        "unchanged_preserved",
        "latest_fact_success",
        "stale_fact_usage",
        "unsupported_commitment",
        "hallucinated_availability",
        "repetition",
    )
    out: dict[str, Any] = {"count": len(rows)}
    for key in keys:
        out[key] = {
            "successes": sum(bool(r[key]) for r in rows),
            "count": len(rows),
            "rate": sum(bool(r[key]) for r in rows) / len(rows) if rows else None,
        }
    latencies = [
        r["incorporation_latency_s"] for r in rows if r["incorporation_latency_s"] is not None
    ]
    out["latency"] = {
        "count": len(latencies),
        "p50_s": float(np.median(latencies)) if latencies else None,
        "p95_s": float(np.quantile(latencies, 0.95)) if len(latencies) >= 20 else None,
    }
    eligible = [r for r in rows if r.get("clarification_required")]
    out["clarification"] = {
        "count": len(eligible),
        "appropriate": sum(r["appropriate_clarification"] for r in eligible),
    }
    for metric in ("rtf", "silence_fraction", "truncation", "asr_uncertainty"):
        values = [float(r[metric]) for r in rows if r.get(metric) is not None]
        out[metric] = {
            "count": len(values),
            "mean": float(np.mean(values)) if values else None,
            "p50": float(np.median(values)) if values else None,
            "p95": float(np.quantile(values, 0.95)) if len(values) >= 20 else None,
        }
    return out


def report(
    base: list[dict[str, Any]], adapted: list[dict[str, Any]], dest: Path, metadata: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metadata": metadata,
        "base": summarize(base),
        "adapted": summarize(adapted),
        "paired": {
            k: bootstrap_delta(base, adapted, k)
            for k in ("correction_success", "joint_success", "unchanged_preserved")
        },
        "timing": {
            timing: {
                "base": summarize([r for r in base if r["timing"] == timing]),
                "adapted": summarize([r for r in adapted if r["timing"] == timing]),
            }
            for timing in ("clean", "overlap")
        },
        "families": {
            family: {
                "base": summarize([r for r in base if r["family"] == family]),
                "adapted": summarize([r for r in adapted if r["family"] == family]),
            }
            for family in sorted({r["family"] for r in base})
        },
        "rows": {"base": base, "adapted": adapted},
        "limitations": [
            "Conservative lexical scoring is not a semantic or audio-quality guarantee.",
            "Latency requires timestamped ASR, and missing latency is not success.",
            "Review generated WAVs for ambiguous cases; no public benchmark claim.",
        ],
    }
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest / "results.json", result)
    lines = [
        "# Matched appointment replay",
        "",
        f"Evidence: {metadata.get('evidence', 'unverified')}",
        f"Examples per model: {len(base)}",
        "",
        "|Metric|Base|Adapted|",
        "|---|---:|---:|",
    ]
    for key in (
        "correction_success",
        "joint_success",
        "unchanged_preserved",
        "unsupported_commitment",
    ):
        lines.append(f"|{key}|{result['base'][key]['rate']}|{result['adapted'][key]['rate']}|")
    lines += [
        "",
        "No guaranteed improvement. See results.json for paired intervals, clean/overlap strata, latency, provenance and limitations.",
    ]
    (dest / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def audible_text(
    generation: dict[str, Any], wav: Path, transcript: Transcript | None
) -> tuple[str, list[Word], str]:
    if transcript is None:
        return generation["internal_text_proxy"], [], "internal_text_proxy"
    if transcript.audio_sha256 != file_hash(wav):
        raise ValueError("ASR is not paired to the generated audible WAV")
    if transcript.text.strip() and not transcript.words:
        raise ValueError(
            "Audible scoring needs timestamped ASR words to exclude pre-correction speech"
        )
    from .audio import validate_alignment

    validate_alignment(transcript.words, generation["duration_s"])
    return transcript.text, transcript.words, "audible_asr"
