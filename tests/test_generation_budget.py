from __future__ import annotations

import pytest

from duplex.budget import BudgetConfig, derive_plan, measured_evaluation_s


def test_complete_conversation_cost_counts_prompt_once_per_example() -> None:
    # 30 seconds of audio took 45 seconds INCLUDING its 10-second prompt.
    result = measured_evaluation_s(
        {"duration_s": 30, "generation_s": 45, "truncated": False}, 12, 20
    )
    assert result == 2 * 12 * 45 + 2 * 20
    profile = {
        "real_checkpoint": True,
        "generation_calibration": "complete_session_v1",
        "optimizer_step_s": 2.56,
        "checkpoint_s": 0.04,
        "matched_eval_s": result,
        "audio_s_per_step": 126.32,
    }
    plan = derive_plan(profile, 3000, BudgetConfig())
    assert plan["evaluation_reserve_s"] >= 1800
    assert plan["train_s"] + plan["evaluation_reserve_s"] + plan["checkpoint_reserve_s"] <= 3000


def test_short_rollout_cannot_estimate_full_evaluation() -> None:
    with pytest.raises(ValueError, match="complete conversation"):
        measured_evaluation_s({"duration_s": 1.6, "generation_s": 11.67, "truncated": True}, 12, 20)
    with pytest.raises(ValueError, match="generation-only"):
        derive_plan({"real_checkpoint": True, "optimizer_step_s": 2.56}, 3300, BudgetConfig())


def test_slow_full_generation_still_refuses_budget_overrun() -> None:
    cost = measured_evaluation_s(
        {"duration_s": 30, "generation_s": 240, "truncated": False}, 12, 20
    )
    profile = {
        "real_checkpoint": True,
        "generation_calibration": "complete_session_v1",
        "optimizer_step_s": 2.56,
        "checkpoint_s": 0.04,
        "matched_eval_s": cost,
        "audio_s_per_step": 126.32,
    }
    with pytest.raises(ValueError, match="Cannot fit"):
        derive_plan(profile, 3300, BudgetConfig())


@pytest.mark.parametrize("count", [0, 3, 5, 49])
def test_evaluation_groups_must_be_predeclared_and_complete(count: int) -> None:
    with pytest.raises(ValueError, match="complete groups"):
        measured_evaluation_s({"duration_s": 30, "generation_s": 45, "truncated": False}, count, 20)


def test_generation_calibration_reuses_checkpoint_without_training(tmp_path, monkeypatch) -> None:
    import time
    from types import SimpleNamespace

    import torch

    from duplex import checkpoint, inference, train
    from duplex.budget import Deadline, Ledger
    from duplex.config import Config, TrainConfig
    from duplex.fixtures import TinyDuplex
    from duplex.io import atomic_json, digest, file_hash, read_json
    from duplex.model import PIN, attach

    torch.set_num_threads(1)
    cfg = Config(train=TrainConfig(layers=[0, 1], wandb_mode="disabled"))
    encoded = tmp_path / "encoded"
    encoded.mkdir()
    sample = encoded / "example.safetensors"
    sample.write_bytes(b"identity-only fixture")
    atomic_json(
        sample.with_suffix(".json"),
        {"split": "train", "frames": 32, "sha256": file_hash(sample), "codec_revision": PIN},
    )
    run = tmp_path / "run"
    run.mkdir()
    prior = {
        "real_checkpoint": True,
        "config_fingerprint": digest(cfg.train.model_dump()),
        "data_fingerprint": digest([file_hash(sample)]),
        "generation_rtf": 7.294,
        "optimizer_step_s": 2.56,
        "checkpoint_s": 0.04,
        "audio_s_per_step": 126.32,
    }
    atomic_json(run / "profile.json", prior)
    lm = TinyDuplex()
    modules = attach(lm, cfg.train)
    optimizer = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad])
    checkpoint.save(
        run / "smoke-checkpoints",
        lm,
        optimizer,
        {"step": 1, "identity": train.identity([sample], cfg, modules)},
        1,
    )
    monkeypatch.setattr(train, "load_real", lambda _: (TinyDuplex(), object(), object()))

    def prohibit_training(*args, **kwargs):
        raise AssertionError("calibration must not run backward/training")

    monkeypatch.setattr(train, "_step", prohibit_training)

    def complete_generation(*args, **kwargs):
        assert "max_frames" not in kwargs
        return {"duration_s": 30, "generation_s": 45, "truncated": False, "rtf": 1.5}

    monkeypatch.setattr(inference, "generate", complete_generation)
    book = Ledger(tmp_path / "budget.json", cfg.budget)
    result = train.calibrate_generation(
        tmp_path,
        encoded,
        run,
        cfg,
        Deadline(time.time() + 3300),
        book,
        SimpleNamespace(duration_s=30),
        tmp_path,
    )
    assert result["generation_calibration"] == "complete_session_v1"
    assert result["matched_eval_s"] >= 1080
    assert read_json(run / "profile.before-generation-calibration.json") == prior
    assert read_json(run / "profile.json") == result
