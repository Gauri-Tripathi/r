"""Real upstream CODE on tiny tensors; no real PersonaPlex weights or codec quality claim."""

from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from duplex.config import TrainConfig
from duplex.model import attach, native, native_loss

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sphn") is None, reason="Linux sphn runtime extra needed"
)


def tiny_native():
    os.environ["NO_TORCH_COMPILE"] = "1"
    os.environ["NO_CUDA_GRAPH"] = "1"
    _, LMModel, _ = native()
    return LMModel(
        delays=[0, 0] + [1] * 7 + [0] + [1] * 7,
        n_q=16,
        dep_q=16,
        card=64,
        text_card=64,
        dim=32,
        num_heads=4,
        num_layers=2,
        causal=True,
        context=64,
        positional_embedding="rope",
        existing_text_padding_id=3,
        depformer_dim=16,
        depformer_num_heads=4,
        depformer_num_layers=1,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_pos_emb="none",
        gating="silu",
        depformer_gating="silu",
        device="cpu",
        dtype=torch.float32,
    )


@pytest.mark.parametrize("modality", ["audio", "text"])
def test_actual_native_delays_frozen_depth_gradients_export(tmp_path, modality):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    lm = tiny_native()
    attach(lm, TrainConfig(layers=[0, 1]))
    before = {n: p.detach().clone() for n, p in lm.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-3)
    codes = torch.randint(4, 64, (1, 17, 12))
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[:, 9:] = False
    valid[..., :2] = False
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = lm.forward_train(codes)
        # Detach text head evidence: adapters must receive AUDIO-loss gradients
        # through the frozen depth transformer, independently of text CE.
        if modality == "audio":
            output.text_logits = output.text_logits.detach()
        else:
            output.logits = output.logits.detach()
        assert output.mask.shape == (1, 16, 12)
        assert output.mask[0, 0, -1] and not output.mask[0, 1, -1]
        loss, _ = native_loss(output, codes, valid, torch.ones_like(codes, dtype=torch.float32))
        loss.backward()
        optimizer.step()
    assert all(
        p.grad is not None and p.grad.abs().sum() > 0 for p in lm.parameters() if p.requires_grad
    )
    assert all(torch.equal(before[n], p) for n, p in lm.named_parameters() if not p.requires_grad)
    from duplex.model import adapter_state, load_adapter, save_adapter

    save_adapter(lm, tmp_path / "adapter.safetensors")
    clone = tiny_native()
    attach(clone, TrainConfig(layers=[0, 1]))
    load_adapter(clone, tmp_path / "adapter.safetensors")
    assert all(torch.equal(t, adapter_state(clone)[n]) for n, t in adapter_state(lm).items())


def test_actual_native_context_step_and_reset():
    torch.set_num_threads(1)
    lm = tiny_native().eval()
    _, _, LMGen = native()
    gen = LMGen(lm, device="cpu", sample_rate=24000, frame_rate=12.5, use_sampling=False)
    user = torch.randint(0, 64, (1, 8, 1))
    silence = torch.zeros(1, 8, 1, dtype=torch.long)
    outputs = []
    for _ in range(2):
        session = []
        with torch.no_grad(), gen.streaming(1):
            for frame in range(6):
                out = gen.step(
                    input_tokens=user,
                    moshi_tokens=silence if frame == 2 else None,
                    text_token=5 if frame == 2 else None,
                )
                if out is not None:
                    session.append(out.clone())
        outputs.append(torch.cat(session, dim=-1))
    assert torch.equal(outputs[0], outputs[1])
    assert (outputs[0][:, 0] == 5).any()


def test_native_training_after_base_generation():
    """The real pilot uses the same model for base speech then adapter training."""
    torch.set_num_threads(1)
    lm = tiny_native()
    attach(lm, TrainConfig(layers=[0, 1]))
    lm.eval()
    _, _, LMGen = native()
    gen = LMGen(lm, device="cpu", sample_rate=24000, frame_rate=12.5, use_sampling=False)
    with torch.inference_mode(), gen.streaming(1):
        for _ in range(6):
            gen.step(input_tokens=torch.randint(0, 64, (1, 8, 1)))
    codes = torch.randint(4, 64, (1, 17, 12))
    valid = torch.ones_like(codes, dtype=torch.bool)
    valid[:, 9:] = False
    lm.train()
    output = lm.forward_train(codes)
    loss, _ = native_loss(output, codes, valid, torch.ones_like(codes, dtype=torch.float32))
    loss.backward()
    assert all(p.grad is not None for p in lm.parameters() if p.requires_grad)
