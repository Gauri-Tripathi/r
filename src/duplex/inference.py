"""Audible replay; generation sees only past/current user audio and arrived facts."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from .audio import SAMPLE_RATE, load_wav
from .io import atomic_json, confined, digest, file_hash
from .model import PIN, LoRALinear, native
from .runtime import SYSTEM, ContextQueue
from .schema import ModelInput


def runtime_fingerprint(system_prompt: str = SYSTEM) -> str:
    return digest(
        {
            "protocol": "native-step-facts-v1",
            "system": system_prompt,
            "source": file_hash(Path(__file__)),
            "queue": file_hash(Path(__file__).with_name("runtime.py")),
        }
    )


def generate(
    lm: Any,
    mimi: Any,
    tokenizer: Any,
    visible: ModelInput,
    root: Path,
    output: Path,
    voice: Path | None = None,
    seed: int = 17,
    max_frames: int | None = None,
    adapter_sha256: str | None = None,
    system_prompt: str = SYSTEM,
) -> dict[str, Any]:
    if visible.user_audio is None:
        raise ValueError("Runtime requires user WAV, never a correction transcript")
    if (
        any(isinstance(m, LoRALinear) and m.enabled for m in lm.modules())
        and adapter_sha256 is None
    ):
        raise ValueError("Enabled adapter generation must record the actual adapter checksum")
    os.environ["NO_TORCH_COMPILE"] = "1"
    os.environ["NO_CUDA_GRAPH"] = "1"
    _, _, LMGen = native()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    lm.eval()
    audio = load_wav(confined(root, visible.user_audio))
    audio = np.pad(audio, (0, (-len(audio)) % 1920))
    device = str(lm.device)
    gen = LMGen(
        lm,
        device=device,
        sample_rate=SAMPLE_RATE,
        frame_rate=12.5,
        use_sampling=False,
        audio_silence_frame_cnt=6,
        text_prompt_tokens=tokenizer.encode(f"<system> {system_prompt} <system>"),
    )
    queue = ContextQueue(visible.request_id, tokenizer)
    pending = sorted(visible.events, key=lambda e: (e.arrival_s, e.event_id))
    index = 0
    frames = []
    tokens = []
    timestamps = []
    forced_frames: set[int] = set()
    started = time.perf_counter()
    frame_count = len(audio) // 1920
    if max_frames is not None:
        frame_count = min(frame_count, max_frames)
    with torch.inference_mode(), mimi.streaming(1), gen.streaming(1):
        if voice:
            if voice.suffix.lower() != ".wav":
                raise ValueError(
                    "Only licensed WAV voice prompts, never upstream pickle embeddings"
                )
            gen.voice_prompt_audio = torch.from_numpy(load_wav(voice, max_s=15)).view(1, -1)
        # Native role prompt and licensed voice behavior, identical for baseline/adapted.
        gen.step_system_prompts(mimi)
        mimi.reset_streaming()
        silence = mimi.encode(torch.zeros(1, 1, 1920, device=device))
        mimi.reset_streaming()
        for frame_index in range(frame_count + gen.max_delay):
            now = frame_index / 12.5
            while (
                index < len(pending)
                and pending[index].arrival_s <= now
                and frame_index < frame_count
            ):
                queue.accept(pending[index], now)
                index += 1
            if frame_index < frame_count:
                chunk = (
                    torch.from_numpy(audio[frame_index * 1920 : (frame_index + 1) * 1920].copy())
                    .view(1, 1, -1)
                    .to(device)
                )
                user = mimi.encode(chunk)
            else:
                user = silence
            forced = queue.step()
            if forced is not None:
                forced_frames.add(frame_index)
            result = gen.step(
                input_tokens=user,
                moshi_tokens=silence if forced is not None else None,
                text_token=forced,
            )
            if result is None:
                continue
            # Prompts have already advanced LMGen.offset. Output at this call is
            # current_offset - max_delay, so conversation-relative index is f-delay.
            output_index = frame_index - gen.max_delay
            if not 0 <= output_index < frame_count:
                continue
            pcm = mimi.decode(result[:, 1:9]).cpu().numpy()[0, 0]
            is_context = output_index in forced_frames
            frames.append(np.zeros_like(pcm) if is_context else pcm)
            token = int(result[0, 0, 0])
            if not is_context and token not in (0, 3) and token < lm.text_card:
                tokens.append(token)
                timestamps.append(
                    {
                        "at_s": output_index / 12.5,
                        "token": token,
                        "piece": tokenizer.decode([token]),
                    }
                )
    elapsed = time.perf_counter() - started
    generated = np.concatenate(frames) if frames else np.zeros(1920, dtype=np.float32)
    if not np.isfinite(generated).all():
        raise ValueError("Nonfinite generated audio")
    output.mkdir(parents=True, exist_ok=True)
    sf.write(output / "generated.wav", generated, SAMPLE_RATE, subtype="FLOAT")
    duration = len(generated) / SAMPLE_RATE
    text = tokenizer.decode(tokens)
    result_data = {
        "internal_text_proxy": text,
        "text_tokens": timestamps,
        "events": queue.audit,
        "input_fingerprint": digest(visible.model_dump()),
        "user_audio_sha256": file_hash(confined(root, visible.user_audio)),
        "seed": seed,
        "adapter_sha256": adapter_sha256,
        "base_revision": PIN,
        "runtime_fingerprint": runtime_fingerprint(system_prompt),
        "system_prompt": system_prompt,
        "decoding": "greedy-v1",
        "voice_sha256": file_hash(voice) if voice else None,
        "generated_audio_sha256": file_hash(output / "generated.wav"),
        "duration_s": duration,
        "generation_s": elapsed,
        "rtf": elapsed / max(duration, 0.08),
        "truncated": frame_count < len(audio) // 1920,
        "context_forced_silence_s": len(forced_frames) / 12.5,
        "silence_fraction": float(np.mean(np.abs(generated) < 0.001)),
        "sample_rate": SAMPLE_RATE,
    }
    atomic_json(output / "generation.json", result_data)
    return result_data
