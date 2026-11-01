"""Synthetic CPU fixtures for pipeline tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from . import checkpoint
from .audio import SpeechRenderer, render_example
from .config import TrainConfig
from .data import generate, reference, save_examples
from .io import atomic_json, digest
from .model import adapter_state, attach, native_loss
from .supervision import checked


class FixtureTokenizer:
    """Stable whitespace token IDs; NOT the PersonaPlex tokenizer."""

    def encode(self, text: str) -> list[int]:
        return [4 + int(digest(word)[:6], 16) % 60 for word in text.split()]


class TinyDuplex(nn.Module):
    """Tiny causal tensor fixture with frozen downstream operations and 17 channels."""

    def __init__(self) -> None:
        super().__init__()
        self.emb = nn.Embedding(64, 16)
        self.transformer = nn.Module()
        self.transformer.layers = nn.ModuleList()
        for _ in range(2):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.out_proj = nn.Linear(16, 16, bias=False)
            self.transformer.layers.append(layer)
        self.depformer = nn.Linear(16, 16)
        self.audio_head = nn.Linear(16, 64 * 16)
        self.text_head = nn.Linear(16, 64)

    def forward_train(self, codes: torch.Tensor) -> Any:
        x = self.emb(codes).sum(1)
        # Causal prior-frame input, with session state kept entirely within this call.
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1).cumsum(1) / 17
        layers: Any = self.transformer.layers
        for layer in layers:
            x = x + layer.self_attn.out_proj(x).tanh()
        x = self.depformer(x).tanh()  # frozen parameters, autograd still propagates
        b, t, _ = x.shape
        logits = self.audio_head(x).view(b, t, 16, 64).permute(0, 2, 1, 3)
        return SimpleNamespace(
            logits=logits,
            mask=torch.ones(b, 16, t, dtype=torch.bool),
            text_logits=self.text_head(x)[:, None],
            text_mask=torch.ones(b, 1, t, dtype=torch.bool),
        )


def fixture(directory: Path) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rows = [checked(r, reference(r.truth)) for r in generate(groups=6)]
    save_examples(directory / "scenarios.jsonl", rows)
    rendered = render_example(rows[0], directory, SpeechRenderer("tone"), FixtureTokenizer())
    save_examples(directory / "rendered.jsonl", [rendered])
    lm = TinyDuplex()
    cfg = TrainConfig(layers=[0, 1], accumulation=1, max_steps=2, wandb_mode="disabled")
    modules = attach(lm, cfg)
    frozen = {n: p.detach().clone() for n, p in lm.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-3)
    codes = torch.randint(4, 64, (1, 17, 32))
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[:, 9:] = False
    valid[..., :3] = False
    weights = torch.ones_like(codes, dtype=torch.float32)
    history = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = native_loss(lm.forward_train(codes), codes, valid, weights)
        loss.backward()
        optimizer.step()
        history.append(metrics)
    if any(p.grad is None or not p.grad.abs().sum() for p in lm.parameters() if p.requires_grad):
        raise ValueError("Fixture adapter gradient failure")
    if any(not torch.equal(frozen[n], p) for n, p in lm.named_parameters() if not p.requires_grad):
        raise ValueError("Fixture frozen parameter changed")
    identity = {"fixture": True, "targets": modules, "data": digest(codes.tolist())}
    metadata = {
        "step": 2,
        "identity": identity,
        "train_config": cfg.model_dump(),
        "real_checkpoint": False,
    }
    saved = checkpoint.save(directory / "checkpoints", lm, optimizer, metadata, sampler=2)
    clone = TinyDuplex()
    attach(clone, cfg)
    restored_optimizer = torch.optim.AdamW(
        [p for p in clone.parameters() if p.requires_grad], lr=1e-3
    )
    step, sampler = checkpoint.restore(saved, clone, restored_optimizer, identity)
    if any(not torch.equal(t, adapter_state(clone)[n]) for n, t in adapter_state(lm).items()):
        raise ValueError("Fixture adapter round trip failed")
    result = {
        "fixture_only": True,
        "speech_model_training": False,
        "real_checkpoint": False,
        "examples": len(rows),
        "tone_audio": rendered.visible.user_audio,
        "optimizer_steps": step,
        "sampler": sampler,
        "adapter_modules": modules,
        "losses": history,
        "checkpoint": str(saved),
        "status": "CPU plumbing passed; no PersonaPlex quality evidence",
    }
    atomic_json(directory / "fixture-result.json", result)
    return result
