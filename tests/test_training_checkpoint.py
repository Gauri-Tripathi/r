from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from duplex import checkpoint
from duplex.config import TrainConfig
from duplex.fixtures import TinyDuplex, fixture
from duplex.model import adapter_state, attach, native_loss


def test_end_to_end_fixture(tmp_path) -> None:
    result = fixture(tmp_path)
    assert result["fixture_only"] and not result["real_checkpoint"]
    assert result["optimizer_steps"] == 2
    assert (tmp_path / result["tone_audio"]).is_file()


def test_mask_user_and_nan_delays() -> None:
    torch.set_num_threads(1)
    lm = TinyDuplex()
    cfg = TrainConfig(layers=[0, 1])
    attach(lm, cfg)
    codes = torch.randint(4, 64, (1, 17, 20))
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[:, 9:] = False
    valid[..., :3] = False
    output = lm.forward_train(codes)
    output.mask[..., -1] = False
    output.logits[..., -1, :] = float("nan")
    output.logits[:, 8:] = float("nan")  # user-codebook logits must never reach CE
    loss, stats = native_loss(output, codes, valid, torch.ones_like(codes, dtype=torch.float32))
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["valid_audio_tokens"] == 8 * 16
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in lm.parameters()
        if p.requires_grad
    )
    assert all(p.grad is None for p in lm.parameters() if not p.requires_grad)


def test_checkpoint_resume_incomplete_and_integrity(tmp_path) -> None:
    torch.set_num_threads(1)
    lm = TinyDuplex()
    attach(lm, TrainConfig(layers=[0, 1]))
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad])
    meta = {"step": 1, "identity": {"fixture": True}}
    path = checkpoint.save(tmp_path, lm, optimizer, meta, sampler=7)
    state = adapter_state(lm)
    step, sampler = checkpoint.restore(path, lm, optimizer, meta["identity"])
    assert (step, sampler) == (1, 7)
    assert all(torch.equal(t, adapter_state(lm)[n]) for n, t in state.items())
    with pytest.raises(ValueError, match="Incompatible"):
        checkpoint.restore(path, lm, optimizer, {"other": True})
    incomplete = tmp_path / ".incomplete"
    incomplete.mkdir()
    with pytest.raises(FileNotFoundError):
        checkpoint.complete(incomplete)
    (path / "metadata.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        checkpoint.complete(path)


def test_loss_valid_normalization() -> None:
    codes = torch.full((1, 17, 4), 1, dtype=torch.long)
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[:, 9:] = False
    valid[..., -2:] = False
    output = SimpleNamespace(
        logits=torch.zeros(1, 16, 4, 2),
        text_logits=torch.zeros(1, 1, 4, 2),
        mask=torch.ones(1, 16, 4, dtype=torch.bool),
        text_mask=torch.ones(1, 1, 4, dtype=torch.bool),
    )
    loss, stats = native_loss(output, codes, valid, torch.ones_like(codes, dtype=torch.float32))
    assert loss.item() == pytest.approx(2 * 0.69314718)
    assert stats["valid_audio_tokens"] == 16
