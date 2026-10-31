"""Optional local CPU ASR of generated speech. No teacher or paid transcription API."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=Path("runs/openyap/full-v1"))
    parser.add_argument("--out", type=Path, default=Path("runs/benchmarks/openyap-full/asr"))
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    from faster_whisper import WhisperModel
    from huggingface_hub import HfApi, snapshot_download

    directory = Path("cache/benchmark-whisper-tiny-en")
    stamp = directory / "benchmark-model.json"
    if not stamp.exists():
        repo = "Systran/faster-whisper-tiny.en"
        revision = HfApi().model_info(repo, token=False).sha
        snapshot_download(repo, revision=revision, local_dir=str(directory), token=False)
        stamp.write_text(
            json.dumps(
                {"repo": repo, "revision": revision, "model_sha256": sha(directory / "model.bin")}
            ),
            encoding="utf-8",
        )
    provenance = json.loads(stamp.read_text(encoding="utf-8"))
    if sha(directory / "model.bin") != provenance["model_sha256"]:
        raise ValueError("ASR model checksum changed")
    if args.download_only:
        print({"local_cpu_asr_ready": True, "revision": provenance["revision"]})
        return
    model = WhisperModel(
        str(directory), device="cpu", compute_type="int8", cpu_threads=2, local_files_only=True
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "engine.json").write_text(
        json.dumps(
            {
                **provenance,
                "package_version": importlib.metadata.version("faster-whisper"),
                "device": "cpu",
                "compute_type": "int8",
                "beam_size": 1,
                "vad_filter": True,
                "word_timestamps": True,
                "limitations": "Small automatic recognizer; agreement is a diagnostic, not speech quality or answer correctness.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for folder, label in (("base-audio", "base"), ("adapted-audio", "full")):
        destination = args.out / f"{label}.jsonl"
        cached = (
            {
                r["id"]: r
                for r in [
                    json.loads(line)
                    for line in destination.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            }
            if destination.exists()
            else {}
        )
        for wav in sorted((args.run / folder).glob("*/generated.wav")):
            checksum = sha(wav)
            if wav.parent.name in cached and cached[wav.parent.name]["audio_sha256"] == checksum:
                continue
            metadata = json.loads((wav.parent / "generation.json").read_text(encoding="utf-8"))
            segments, _ = model.transcribe(
                str(wav),
                language="en",
                beam_size=1,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=True,
            )
            words = []
            probabilities = []
            for segment in segments:
                for w in segment.words or []:
                    start = max(0, float(w.start))
                    end = min(metadata["duration_s"], float(w.end))
                    if end > start:
                        words.append({"text": w.word.strip(), "start_s": start, "end_s": end})
                        probabilities.append(w.probability)
            cached[wav.parent.name] = {
                "id": wav.parent.name,
                "audio_sha256": checksum,
                "text": " ".join(w["text"] for w in words),
                "words": words,
                "uncertainty": 1 - sum(probabilities) / len(probabilities)
                if probabilities
                else 1.0,
                "engine": "faster-whisper/tiny.en/int8",
                "revision": provenance["revision"],
            }
            temporary = destination.with_suffix(".tmp")
            temporary.write_text(
                "".join(json.dumps(r) + "\n" for r in cached.values()), encoding="utf-8"
            )
            temporary.replace(destination)
            print(f"ASR {label}/{wav.parent.name}: {len(words)} words", flush=True)


if __name__ == "__main__":
    main()
