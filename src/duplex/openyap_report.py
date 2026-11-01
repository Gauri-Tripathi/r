"""Saved-run reports and speech comparisons."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any

from .checkpoint import adapter_metadata
from .io import file_hash, read_json


def report(run: Path) -> dict[str, Any]:
    result = read_json(run / "results.json")
    verdict = (
        "Both held-out native losses decreased; conversational quality is still unverified."
        if all(v < 0 for v in result["test_loss_delta"].values())
        else "Held-out native losses are mixed or unchanged; no consistent improvement demonstrated."
    )
    adapter_metadata(run / "export")
    pairs = []
    for directory in sorted((run / "base-audio").iterdir()):
        other = run / "adapted-audio" / directory.name
        base, adapted = (
            read_json(directory / "generation.json"),
            read_json(other / "generation.json"),
        )
        for key in (
            "input_fingerprint",
            "user_audio_sha256",
            "seed",
            "decoding",
            "voice_sha256",
            "runtime_fingerprint",
            "system_prompt",
            "duration_s",
        ):
            if base[key] != adapted[key]:
                raise ValueError(f"Unmatched speech pair: {key}")
        for folder, metadata in ((directory, base), (other, adapted)):
            if file_hash(folder / "generated.wav") != metadata["generated_audio_sha256"]:
                raise ValueError("Generated waveform checksum mismatch")
        pairs.append(
            f'<article><h2>{escape(directory.name)}</h2><label>Base model</label><audio controls preload="none" src="base-audio/{escape(directory.name)}/generated.wav"></audio><label>Adapter</label><audio controls preload="none" src="adapted-audio/{escape(directory.name)}/generated.wav"></audio></article>'
        )
    lines = [
        "# OpenYap: measured result",
        "",
        verdict + " Changed waveforms alone do not establish quality.",
        "",
        "| Held-out metric | Base | Adapted | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for key in ("audio_loss", "text_loss"):
        before, after = result["base_test"][key], result["adapted_test"][key]
        lines.append(
            f"| {key} | {before:.8f} | {after:.8f} | {100 * (after - before) / before:+.5f}% |"
        )
    lines += [
        "",
        f"{result['unique_train_examples']} unique train windows, {result['unique_train_audio_s']} seconds, {result['optimizer_steps']} optimizer steps, {result['effective_epochs']} passes. Selected checkpoint: {result['selected_checkpoint']}.",
        f"Mean optimizer step: {result['seconds_per_optimizer_step']:.3f}s. Total measured pilot: {result['wall_s']:.1f}s.",
        "",
        f"Changed adapter tensors: {sum(v > 0 for v in result['changed_adapter_elements'].values())}. Frozen parameter versions unchanged: {result['frozen_parameter_versions_unchanged']}. No ASR or listening quality score has been claimed.",
        "",
        "Open comparison.html to listen. This human-conversation pilot does not establish correction or tool-grounding capability.",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (run / "comparison.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>OpenYap base versus adapter</title><style>body{font:16px system-ui;max-width:800px;margin:40px auto;padding:20px;background:#f7f8fa;color:#172033}article{background:white;padding:20px;margin:20px 0;border:1px solid #ddd;border-radius:12px}h2{font-size:16px}label{display:block;margin-top:14px}audio{width:100%}</style><h1>OpenYap: base versus adapter</h1><p>The pilot completed. Held-out losses are essentially flat. Listen for coherence, intelligibility and speech regressions. No improvement is established by a changed waveform.</p><p>Same user input, conversation prompt, greedy decoding and seed. No assistant target was supplied during generation.</p>'
        + f"<p>{escape(verdict)}</p>"
        + "".join(pairs)
        + "</html>",
        encoding="utf-8",
    )
    return {
        "report": str(run / "report.md"),
        "comparison": str(run / "comparison.html"),
        "adapter": str(run / "export"),
        "pairs_verified": len(pairs),
        "held_out_loss_delta": result["test_loss_delta"],
    }
