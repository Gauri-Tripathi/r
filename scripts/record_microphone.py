"""Run locally on your computer, never inside Modal. Optional sounddevice dependency."""

import argparse
from pathlib import Path

import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("input.wav"))
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--device")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(
            "Install the optional local recorder: uv pip install --python .venv/Scripts/python.exe sounddevice"
        ) from None
    if args.list_devices:
        print(sd.query_devices())
        return
    if not 0 < args.seconds <= 60:
        raise SystemExit("Choose 0–60 recording seconds")
    if args.out.exists():
        raise SystemExit("Output already exists; choose another --out filename")
    device = int(args.device) if args.device and args.device.isdecimal() else args.device
    print(f"Recording {args.seconds:g} seconds. Speak now.", flush=True)
    pcm = sd.rec(
        round(args.seconds * 24000), samplerate=24000, channels=1, dtype="float32", device=device
    )
    sd.wait()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, pcm, 24000, subtype="PCM_16")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
