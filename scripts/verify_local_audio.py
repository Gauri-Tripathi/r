"""Real eSpeak NG CPU rendering check, without model assets or paid TTS."""

import json
from pathlib import Path

from duplex.audio import SpeechRenderer, load_wav, render_example
from duplex.codec import align_text
from duplex.data import generate, reference
from duplex.fixtures import FixtureTokenizer
from duplex.io import atomic_json, read_json
from duplex.schema import Word
from duplex.teacher import checked

root = Path("runs/espeak-check")
renderer = SpeechRenderer("espeak")
rows = [next(r for r in generate(6) if r.timing == timing) for timing in ("clean", "overlap")]
results = []
for row in rows:
    row = checked(row, reference(row.truth))
    rendered = render_example(row, root, renderer, FixtureTokenizer())
    meta = read_json(root / "audio" / row.id / "manifest.json")
    audio = load_wav(root / meta["assistant"])
    words = [Word.model_validate(w) for w in meta["words"]]
    align_text(words, len(audio) // 1920, FixtureTokenizer())
    results.append(
        {
            "id": row.id,
            "duration_s": rendered.visible.duration_s,
            "timing": row.timing,
            "renderer_version": renderer.version,
        }
    )
atomic_json(
    root / "verification.json",
    {"real_speech_renderer": True, "native_tokenizer_verified": False, "results": results},
)
print(json.dumps(results, indent=2))
