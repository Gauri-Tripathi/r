"""Licensed WAV import and real CPU eSpeak NG rendering with measured segment alignment."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .io import atomic_json, confined, digest, file_hash, read_json
from .runtime import schedule
from .schema import Example, Word

SAMPLE_RATE = 24000
PREPROCESS_VERSION = "mono-resample-poly-v1"


def load_wav(path: Path, max_s: float = 300) -> np.ndarray:
    info = sf.info(path)
    if info.channels != 1 or info.duration <= 0 or info.duration > max_s:
        raise ValueError("Expected bounded, nonempty, isolated mono WAV")
    if not 8000 <= info.samplerate <= 96000:
        raise ValueError("Unsupported sample rate")
    samples, sr = sf.read(path, dtype="float32")
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) >= 0.999:
        raise ValueError("Nonfinite or clipped audio")
    if sr != SAMPLE_RATE:
        divisor = math.gcd(sr, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // divisor, sr // divisor).astype(np.float32)
    if np.max(np.abs(samples)) >= 0.999:
        raise ValueError("Resampling caused clipping; lower gain in source")
    return samples


class SpeechRenderer:
    """Word-concatenation deliberately provides exact construction boundaries.

    Robotic prosody is a limitation, not a forced-alignment claim. Import natural
    speech plus externally checked word timings for a higher quality experiment.
    """

    def __init__(self, backend: str, voice: str = "en-us"):
        if backend not in ("espeak", "tone") or voice not in ("en-us", "en-gb"):
            raise ValueError("Unsupported backend or allowlisted voice")
        self.backend = backend
        self.voice = voice
        self.exe = shutil.which("espeak-ng") if backend == "espeak" else None
        if backend == "espeak" and not self.exe:
            raise ValueError(
                "Install espeak-ng (Debian: apt-get install espeak-ng) or import licensed WAVs; tone is plumbing only"
            )
        self.version = (
            subprocess.run(
                [self.exe, "--version"], capture_output=True, text=True, check=True, timeout=10
            ).stdout.strip()
            if self.exe
            else "tone-fixture-v1"
        )

    def render(self, text: str) -> tuple[np.ndarray, list[Word]]:
        pieces = []
        words = []
        offset = 0
        with tempfile.TemporaryDirectory() as temp:
            for index, word in enumerate(text.split()):
                if self.exe:
                    path = Path(temp) / f"{index}.wav"
                    subprocess.run(
                        [self.exe, "-v", self.voice, "-s", "175", "-w", str(path), "--stdin"],
                        input=word,
                        text=True,
                        check=True,
                        capture_output=True,
                        timeout=15,
                    )
                    audio = load_wav(path, max_s=10)
                else:
                    t = np.arange(6000, dtype=np.float32) / SAMPLE_RATE
                    audio = (0.08 * np.sin(2 * np.pi * (300 + index % 5 * 40) * t)).astype(
                        np.float32
                    )
                words.append(
                    Word(
                        text=word,
                        start_s=offset / SAMPLE_RATE,
                        end_s=(offset + len(audio)) / SAMPLE_RATE,
                    )
                )
                pieces.append(audio)
                gap = np.zeros(480, dtype=np.float32)
                pieces.append(gap)
                offset += len(audio) + len(gap)
        return np.concatenate(pieces), words


def validate_alignment(words: list[Word], duration_s: float, text: str | None = None) -> None:
    previous = 0.0
    for word in words:
        if word.start_s < previous - 1e-5 or word.end_s > duration_s + 1 / SAMPLE_RATE:
            raise ValueError("Nonmonotonic or out-of-bounds word alignment")
        previous = word.end_s
    if text is not None and " ".join(w.text for w in words) != text:
        raise ValueError("Alignment text differs from supervision")


def render_example(row: Example, root: Path, renderer: SpeechRenderer, tokenizer: Any) -> Example:
    if not row.supervision:
        raise ValueError("Build checked targets before rendering")
    key = digest(
        {
            "example": row.model_dump(),
            "renderer": renderer.version,
            "voice": renderer.voice,
            "preprocess": PREPROCESS_VERSION,
            "timeline": "context-aware-prefix-v2",
        }
    )
    directory = root / "audio" / row.id
    manifest = directory / "manifest.json"
    if manifest.exists():
        prior = read_json(manifest)
        if prior["key"] == key and all(
            file_hash(confined(root, prior[k])) == prior[k + "_sha256"]
            for k in ("user", "assistant")
        ):
            return _rendered_row(row, prior)
        raise ValueError("Audio cache inputs changed; use a fresh output directory")
    # Render preceding conversation identically across a counterfactual group.
    request_audio, _ = renderer.render(row.script[0].text)
    prefix_audio, prefix_words = renderer.render(row.script[1].text)
    assistant_start = len(request_audio) / SAMPLE_RATE + 0.32
    correction = next((u for u in row.script if u.kind == "correction"), None)
    correction_audio, _ = (
        renderer.render(correction.text) if correction else (np.zeros(0, dtype=np.float32), [])
    )
    correction_start = assistant_start + (
        0.48 if row.timing == "overlap" else len(prefix_audio) / SAMPLE_RATE + 0.32
    )
    correction_end = correction_start + len(correction_audio) / SAMPLE_RATE
    arrival = max(correction_end, assistant_start + len(prefix_audio) / SAMPLE_RATE) + 0.24
    events = [
        e.model_copy(
            update={
                "arrival_s": assistant_start + 0.48
                if e.arrival_s < 10
                else arrival + (1 if e.version == 1 else 0)
            }
        )
        for e in row.visible.events
    ]
    # Use the actual tokenizer to measure injection frames; this never changes user constraints.
    context, audit = schedule(
        events, math.ceil((arrival + 40) * 12.5), 12.5, row.visible.request_id, tokenizer
    )
    injection_end = (
        max((i + 1 for i, token in enumerate(context) if token is not None), default=0) / 12.5
    )
    response_start = max(correction_end + 0.32, arrival, injection_end + 0.32)
    response_audio, response_words = renderer.render(row.supervision)
    # Prefix is valid only until the correction has become fully audible.
    prefix_end = (
        min(assistant_start + len(prefix_audio) / SAMPLE_RATE, correction_end)
        if correction
        else assistant_start + len(prefix_audio) / SAMPLE_RATE
    )
    # The facts interface forces assistant silence. End the reference prefix
    # before the first early injection, rather than encoding contradictory speech.
    if events:
        prefix_end = min(prefix_end, min(e.arrival_s for e in events))
    duration = max(response_start + len(response_audio) / SAMPLE_RATE + 1, arrival + 1)
    if duration > 120:
        raise ValueError("Rendered clip exceeds pilot limit")
    n = math.ceil(duration * SAMPLE_RATE / 1920) * 1920
    user = np.zeros(n, dtype=np.float32)
    assistant = np.zeros(n, dtype=np.float32)

    def put(dest: np.ndarray, audio: np.ndarray, start: float, end: float | None = None) -> None:
        offset = round(start * SAMPLE_RATE)
        count = len(audio) if end is None else min(len(audio), round((end - start) * SAMPLE_RATE))
        dest[offset : offset + count] = audio[:count]

    put(user, request_audio, 0)
    put(user, correction_audio, correction_start)
    put(assistant, prefix_audio, assistant_start, prefix_end)
    put(assistant, response_audio, response_start)
    aligned = [
        Word(text=w.text, start_s=w.start_s + assistant_start, end_s=w.end_s + assistant_start)
        for w in prefix_words
        if w.end_s + assistant_start <= prefix_end
    ]
    aligned += [
        Word(text=w.text, start_s=w.start_s + response_start, end_s=w.end_s + response_start)
        for w in response_words
    ]
    validate_alignment(aligned, n / SAMPLE_RATE)
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(directory / "user.wav", user, SAMPLE_RATE, subtype="FLOAT")
    sf.write(directory / "assistant.wav", assistant, SAMPLE_RATE, subtype="FLOAT")
    payload = {
        "key": key,
        "sample_rate": SAMPLE_RATE,
        "duration_s": n / SAMPLE_RATE,
        "user": (directory / "user.wav").relative_to(root).as_posix(),
        "assistant": (directory / "assistant.wav").relative_to(root).as_posix(),
        "user_sha256": file_hash(directory / "user.wav"),
        "assistant_sha256": file_hash(directory / "assistant.wav"),
        "words": [w.model_dump() for w in aligned],
        "events": [e.model_dump() for e in events],
        "context": audit,
        "response_start_s": response_start,
        "correction_end_s": correction_end,
        "prefix_start_s": assistant_start,
        "correction_start_s": correction_start,
        "prefix_end_s": prefix_end,
        "alignment": "measured word-concatenation boundaries; within-word onset uncertainty up to segment duration; 80ms codec quantization",
        "backend": renderer.backend,
        "renderer_version": renderer.version,
        "voice": renderer.voice,
        "license": "fixture-only"
        if renderer.backend == "tone"
        else "eSpeak NG GPL-3.0-or-later program; see third-party notices for output terms",
    }
    atomic_json(manifest, payload)
    _timeline_report(directory, row, payload)
    return _rendered_row(row, payload)


def _timeline_report(directory: Path, row: Example, payload: dict[str, Any]) -> None:
    lines = [
        f"# Timeline: {row.id}",
        "",
        f"Backend: {payload['backend']}; 24kHz isolated user/assistant WAVs.",
        "",
        "|Event|Start (s)|End (s)|",
        "|---|---:|---:|",
    ]
    for kind, start, end in (
        ("Assistant prefix", "prefix_start_s", "prefix_end_s"),
        ("Spoken correction", "correction_start_s", "correction_end_s"),
    ):
        if start in payload:
            lines.append(f"|{kind}|{payload[start]:.3f}|{payload[end]:.3f}|")
    for event in row.visible.events:
        actual = next((e for e in payload["events"] if e["event_id"] == event.event_id), None)
        if actual:
            lines.append(f"|Calendar revision {event.version}|{actual['arrival_s']:.3f}|arrives|")
    lines += [
        f"|Checked target speech|{payload['response_start_s']:.3f}|{payload['duration_s']:.3f}|",
        "",
        "Listen to both isolated WAVs together to audit overlap. Future facts/target answers must not enter generation. Context forces silence; see manifest.json for exact accepted injection frame counts and measured word boundaries.",
    ]
    (directory / "timeline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rendered_row(row: Example, payload: dict[str, Any]) -> Example:
    from .schema import ContextEvent

    visible = row.visible.model_copy(
        update={
            "user_audio": payload["user"],
            "events": [ContextEvent.model_validate(e) for e in payload["events"]],
            "duration_s": payload["duration_s"],
        }
    )
    from .schema import Utterance

    script = []
    for utterance in row.script:
        if utterance.kind == "response":
            continue
        key = (
            "prefix_start_s"
            if utterance.kind == "prefix"
            else "correction_start_s"
            if utterance.kind == "correction"
            else ""
        )
        script.append(
            utterance.model_copy(update={"start_s": payload[key]})
            if key and key in payload
            else utterance
        )
    if row.supervision:
        script.append(
            Utterance(
                role="assistant",
                kind="response",
                text=row.supervision,
                start_s=payload["response_start_s"],
            )
        )
    return row.model_copy(update={"visible": visible, "script": script})


def import_audio(row: Example, source_root: Path, spec: dict[str, Any], root: Path) -> Example:
    required = {
        "user",
        "assistant",
        "words",
        "license",
        "provenance",
        "response_start_s",
        "correction_end_s",
    }
    if set(spec) != required or not spec["license"] or not spec["provenance"]:
        raise ValueError("Audio import requires exact fields, license and provenance")
    user = load_wav(confined(source_root, spec["user"]))
    assistant = load_wav(confined(source_root, spec["assistant"]))
    if len(user) != len(assistant):
        raise ValueError("Isolated channels must share a common timeline and length")
    words = [Word.model_validate(w) for w in spec["words"]]
    validate_alignment(words, len(user) / SAMPLE_RATE)
    if spec["response_start_s"] < spec["correction_end_s"]:
        raise ValueError("Target response cannot precede audible correction")
    # Dataset-specific natural alignments are accepted only with exact supervision suffix.
    response = " ".join(w.text for w in words if w.start_s >= spec["response_start_s"])
    if response != row.supervision:
        raise ValueError("Imported response words must exactly match checked supervision")
    directory = root / "audio" / row.id
    directory.mkdir(parents=True, exist_ok=True)
    sf.write(directory / "user.wav", user, SAMPLE_RATE, subtype="FLOAT")
    sf.write(directory / "assistant.wav", assistant, SAMPLE_RATE, subtype="FLOAT")
    payload = {
        **spec,
        "backend": "import",
        "sample_rate": SAMPLE_RATE,
        "duration_s": len(user) / SAMPLE_RATE,
        "events": [e.model_dump() for e in row.visible.events],
        "user": f"audio/{row.id}/user.wav",
        "assistant": f"audio/{row.id}/assistant.wav",
        "user_sha256": file_hash(directory / "user.wav"),
        "assistant_sha256": file_hash(directory / "assistant.wav"),
    }
    atomic_json(directory / "manifest.json", payload)
    return _rendered_row(row, payload)
