"""Pinned native Moshi APIs and selective temporal output-projection LoRA.

Fused QKV is a bare parameter used by functional.linear, not an nn.Linear.
We deliberately adapt only audited temporal self_attn.out_proj modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from .config import TrainConfig
from .io import atomic_json, file_hash, read_json

ROOT = Path(__file__).resolve().parents[2]
PIN = "fdaf4090a61cb315c138a1faee287ffd6c716309"
FILES = {
    "base": "model.safetensors",
    "codec": "tokenizer-e351c8d8-checkpoint125.safetensors",
    "tokenizer": "tokenizer_spm_32k_3.model",
}


def native() -> tuple[Any, Any, Any]:
    vendor = ROOT / "vendor" / "personaplex"
    if "moshi" in sys.modules and not Path(
        sys.modules["moshi"].__file__ or ""
    ).resolve().is_relative_to(vendor):
        raise ValueError("Competing Moshi package detected; use only the pinned vendored runtime")
    sys.path.insert(0, str(vendor)) if str(vendor) not in sys.path else None
    try:
        from moshi.models import LMGen, LMModel, loaders
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"Missing runtime dependency {exc.name}. On Linux use uv sync --locked --extra gpu. CPU fixture tests do not require the codec."
        ) from None
    return loaders, LMModel, LMGen


def prepare_assets(directory: Path) -> dict[str, Any]:
    """CPU only; explicit caller must have accepted model terms and configured HF_TOKEN."""
    from huggingface_hub import hf_hub_download

    directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for kind, filename in FILES.items():
        path = Path(
            hf_hub_download("nvidia/personaplex-7b-v1", filename, revision=PIN, local_dir=directory)
        )
        files[kind] = {"filename": filename, "sha256": file_hash(path)}
    manifest = {"revision": PIN, "files": files}
    atomic_json(directory / "assets.json", manifest)
    return manifest


def verify_assets(
    directory: Path, include_base: bool = True, include_codec: bool = True
) -> dict[str, Any]:
    manifest = read_json(directory / "assets.json")
    if manifest["revision"] != PIN:
        raise ValueError("Wrong checkpoint/tokenizer/codec revision")
    for kind, item in manifest["files"].items():
        if kind not in FILES or item["filename"] != FILES[kind]:
            raise ValueError("Unexpected asset filename")
        if (kind != "base" or include_base) and (kind != "codec" or include_codec):
            if file_hash(directory / item["filename"]) != item["sha256"]:
                raise ValueError(f"Asset checksum mismatch: {kind}")
    if set(manifest["files"]) != set(FILES):
        raise ValueError("Incomplete asset manifest")
    return manifest


def tokenizer(directory: Path) -> Any:
    import sentencepiece

    return sentencepiece.SentencePieceProcessor(model_file=str(directory / FILES["tokenizer"]))


def load_real(directory: Path, device: str = "cuda") -> tuple[Any, Any, Any]:
    verify_assets(directory)
    if device != "cuda" or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError(
            "Real training requires one BF16-capable GPU; no automatic GPU substitution"
        )
    if (
        torch.cuda.device_count() != 1
        or "A100" not in torch.cuda.get_device_name(0)
        or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3
    ):
        raise ValueError("Expected exactly one A100 80GB")
    loaders, _, _ = native()
    lm = loaders.get_moshi_lm(directory / FILES["base"], device=device, dtype=torch.bfloat16)
    if lm.dep_q != 16 or lm.n_q != 16 or lm.audio_offset != 1:
        raise ValueError("Unsupported PersonaPlex channel configuration")
    if any(p.is_meta for p in lm.parameters()):
        raise ValueError("Upstream loader left missing parameters on meta device")
    mimi = loaders.get_mimi(directory / FILES["codec"], device=device)
    if mimi.sample_rate != 24000 or mimi.frame_rate != 12.5 or mimi.num_codebooks != 8:
        raise ValueError("Unexpected Mimi settings; refuse implicit channel/delay assumptions")
    mimi.requires_grad_(False)
    return lm, mimi, tokenizer(directory)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.a = nn.Parameter(
            torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32)
        )
        self.b = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.a, a=5**0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return self.base(x)
        delta = torch.nn.functional.linear(
            torch.nn.functional.linear(self.dropout(x.float()), self.a), self.b
        )
        return self.base(x) + (delta * (self.alpha / self.rank)).to(x.dtype)


def enable_adapters(lm: nn.Module, enabled: bool) -> None:
    for module in lm.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def attach(lm: nn.Module, config: TrainConfig) -> list[str]:
    lm.requires_grad_(False)
    names = []
    for layer in config.layers:
        if layer < 0:
            raise ValueError("Layer indices must be nonnegative")
        name = f"transformer.layers.{layer}.self_attn.out_proj"
        try:
            original = lm.get_submodule(name)
            attention = lm.get_submodule(name.rsplit(".", 1)[0])
        except AttributeError:
            raise ValueError(f"Missing exact target module {name}") from None
        if not isinstance(original, nn.Linear) or bool(
            getattr(attention, "weights_per_step", False)
        ):
            raise ValueError("Only ordinary temporal output projections are supported")
        lm.set_submodule(name, LoRALinear(original, config.rank, config.alpha, config.dropout))
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("Duplicate target layers")
    return names


def adapter_state(lm: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().contiguous() for name, p in lm.named_parameters() if p.requires_grad
    }


def save_adapter(lm: nn.Module, path: Path) -> None:
    save_file(adapter_state(lm), str(path))


def load_adapter(lm: nn.Module, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    expected = {name: p for name, p in lm.named_parameters() if p.requires_grad}
    if set(state) != set(expected):
        raise ValueError("Adapter exact target names differ")
    with torch.no_grad():
        for name, p in expected.items():
            if p.shape != state[name].shape or not torch.isfinite(state[name]).all():
                raise ValueError("Adapter shape or finite-value validation failed")
            p.copy_(state[name].to(p.device))


def native_loss(
    output: Any,
    codes: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    text_pad_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Native forward_train already undelays logits. Never shift them twice.

    Architecture predicts 16 audio codebooks; 1..8 are assistant, 9..16 user.
    Only assistant audio and text targets have nonzero mask here.
    """
    audio_mask = valid[:, 1:9] & output.mask[:, :8]
    text_mask = valid[:, :1] & output.text_mask
    audio_logits = output.logits[:, :8]
    text_logits = output.text_logits

    def ce(
        logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        # Select valid entries before CE: delayed NaNs must never enter backward.
        if not mask.any():
            raise ValueError("No valid targets for modality")
        loss = torch.nn.functional.cross_entropy(
            logits[mask].float(), targets[mask], reduction="none"
        )
        selected = w[mask]
        return (loss * selected).sum() / selected.sum().clamp_min(1)

    audio_loss = ce(audio_logits, codes[:, 1:9], audio_mask, weights[:, 1:9])
    text_weights = weights[:, :1] * torch.where(codes[:, :1] == 3, text_pad_weight, 1.0)
    text_loss = ce(text_logits, codes[:, :1], text_mask, text_weights)
    return audio_loss + text_loss, {
        "text_loss": text_loss.item(),
        "audio_loss": audio_loss.item(),
        "valid_audio_tokens": audio_mask.sum().item(),
        "valid_text_tokens": text_mask.sum().item(),
        "real_text_tokens": (text_mask & (codes[:, :1] != 3)).sum().item(),
    }
