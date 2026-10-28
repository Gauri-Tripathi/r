from __future__ import annotations

import time

import pytest
import yaml

from duplex.audio import SpeechRenderer, load_wav, render_example, validate_alignment
from duplex.budget import BudgetConfig, Deadline, Ledger, derive_plan
from duplex.data import generate, reference
from duplex.fixtures import FixtureTokenizer
from duplex.io import confined, read_json, read_yaml, redact
from duplex.schema import Word
from duplex.teacher import TeacherConfig, api_targets, checked
from duplex.tracking import Tracker


def test_budget_persists_and_counts_idle(tmp_path) -> None:
    cfg = BudgetConfig()
    book = Ledger(tmp_path / "ledger.json", cfg)
    first = book.reserve(3300, now=100)
    with pytest.raises(ValueError, match="One GPU"):
        Ledger(tmp_path / "ledger.json", cfg).reserve(3300, now=200)
    with pytest.raises(ValueError, match="Verify"):
        book.close(first["id"], False, now=300)
    book.close(first["id"], True, now=3400)
    assert read_json(book.path)["sessions"][0]["charged_s"] == 3300
    with pytest.raises(ValueError, match="cannot reset"):
        Ledger(book.path, cfg.model_copy(update={"ceiling_usd": 17}))
    with pytest.raises(ValueError, match="ceiling"):
        book.reserve(18000, now=3500)
    book.oom_recovery()
    with pytest.raises(ValueError, match="exhausted"):
        book.oom_recovery()


def test_deadline_and_measured_plan() -> None:
    assert not Deadline(time.time() + 5, reserve_s=10).can_step(1)
    d = Deadline(time.time() + 100, reserve_s=10)
    assert d.can_step(1)
    d.stop_requested = True
    assert not d.can_step(1)
    with pytest.raises(ValueError, match="real-checkpoint"):
        derive_plan({"real_checkpoint": False}, 1000, BudgetConfig())
    profile = {
        "real_checkpoint": True,
        "generation_calibration": "complete_session_v1",
        "optimizer_step_s": 5,
        "checkpoint_s": 20,
        "matched_eval_s": 200,
        "audio_s_per_step": 20,
    }
    p = derive_plan(profile, 3000, BudgetConfig())
    assert p["train_s"] + p["evaluation_reserve_s"] + p["checkpoint_reserve_s"] <= 3000


def test_secret_redaction_and_tracking_allowlist(tmp_path, monkeypatch) -> None:
    sentinel = "test-secret-do-not-log-123456"
    monkeypatch.setenv("TEACHER_API_KEY", sentinel)
    safe = redact(
        {"api_key": sentinel, "message": sentinel + " https://example.com/a?sig=secret", "loss": 1}
    )
    assert sentinel not in str(safe) and "sig=" not in str(safe)
    tracker = Tracker(tmp_path, mode="disabled")
    tracker.log({"step": 1, "loss": 0.5})
    with pytest.raises(ValueError, match="allowlisted"):
        tracker.log({"step": 2, "transcript": sentinel})
    with pytest.raises(ValueError, match="Nonmonotonic"):
        tracker.log({"step": 0, "loss": 1})
    tracker.close()
    assert sentinel not in (tmp_path / "metrics.jsonl").read_text()


@pytest.mark.parametrize("relative", ["../x", "/etc/passwd", "C:/secret", "..\\x", "a/../../b"])
def test_path_traversal_rejected(tmp_path, relative) -> None:
    with pytest.raises(ValueError):
        confined(tmp_path, relative)


def test_safe_yaml(tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("!!python/object/apply:os.system ['whoami']")
    with pytest.raises(yaml.YAMLError):
        read_yaml(path)


def test_teacher_does_not_fallback_without_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TEACHER_API_KEY", raising=False)
    cfg = TeacherConfig(
        endpoint="https://example.com/chat/completions",
        model="configured-model",
        budget_usd=0.1,
        input_per_million=1,
        output_per_million=1,
    )
    with pytest.raises(ValueError, match="no automatic"):
        api_targets(generate(3), cfg, tmp_path)


def test_real_timeline_overlap_and_idempotence(tmp_path) -> None:
    row = next(r for r in generate(6) if r.timing == "overlap")
    row = checked(row, reference(row.truth))
    renderer = SpeechRenderer("tone")
    rendered = render_example(row, tmp_path, renderer, FixtureTokenizer())
    again = render_example(row, tmp_path, renderer, FixtureTokenizer())
    assert rendered == again
    meta = read_json(tmp_path / "audio" / row.id / "manifest.json")
    assert meta["prefix_start_s"] < meta["correction_start_s"] < meta["prefix_end_s"]
    assert meta["response_start_s"] > meta["correction_end_s"]
    assert len(load_wav(tmp_path / meta["user"])) == len(load_wav(tmp_path / meta["assistant"]))
    with pytest.raises(ValueError, match="Nonmonotonic"):
        validate_alignment(
            [Word(text="a", start_s=0, end_s=1), Word(text="b", start_s=0.5, end_s=2)], 3
        )
