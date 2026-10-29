"""Research gates: external supervision, isolation, real inputs and bounded reuse."""

import subprocess
import sys

import numpy as np
import pytest
import torch

from duplex.config import Config
from duplex.data import reference, validate_splits
from duplex.experiment import exposure_plan, research_readiness
from duplex.io import atomic_json, read_json, write_jsonl
from duplex.model import LoRALinear, enable_adapters
from duplex.preparation import prepare_evaluation
from duplex.research import candidate_pool, select_groups, semantic_key
from duplex.supervision import (
    CONTRACT,
    TeacherIdentity,
    checked,
    imported,
    require_teacher,
    scenario_sha256,
    teacher_request,
)


def test_distinct_candidates_split_controls_and_context_timing():
    rows = candidate_pool()
    assert 300 <= len(rows) <= 800
    assert len({semantic_key(r) for r in rows}) == len(rows)
    assert rows == candidate_pool()
    validate_splits(rows)
    assert any(not r.visible.events for r in rows)
    assert any(not any(u.kind == "correction" for u in r.script) for r in rows)
    assert {r.timing for r in rows} == {"clean", "overlap"}
    assert any(e.arrival_s < 10 for r in rows for e in r.visible.events)
    assert any(e.version == 1 for r in rows for e in r.visible.events)
    corrections = {
        split: {
            u.text for r in rows if r.split == split for u in r.script if u.kind == "correction"
        }
        for split in ("train", "dev", "test")
    }
    assert not corrections["train"] & corrections["dev"]
    assert not corrections["train"] & corrections["test"]
    assert not corrections["dev"] & corrections["test"]
    for split in ("dev", "test"):
        assert len(select_groups(rows, split, 3)) >= 8


def test_versioned_external_teacher_contract(tmp_path):
    row = candidate_pool()[0]
    request = teacher_request(row)
    assert request["scenario"]["conversation"]
    assert request["scenario"]["simulator_expected"] == row.truth.model_dump()
    raw = {
        "schema_version": CONTRACT,
        "id": row.id,
        "scenario_sha256": scenario_sha256(row),
        "prompt_version": request["prompt_version"],
        "teacher": {"model": "external-model", "revision": "test-revision", "access": "external"},
        "text": reference(row.truth),
    }
    path = tmp_path / "teacher.jsonl"
    write_jsonl(path, [raw])
    result = imported([row], path)
    require_teacher(result)
    assert result[0].provenance["teacher_model"] == "external-model"
    raw["scenario_sha256"] = "0" * 64
    write_jsonl(path, [raw])
    with pytest.raises(ValueError, match="fingerprint"):
        imported([row], path)
    raw["scenario_sha256"] = scenario_sha256(row)
    raw["api_key"] = "disallowed-extra-field"
    write_jsonl(path, [raw])
    with pytest.raises(ValueError):
        imported([row], path)


def test_fixtures_and_unversioned_imports_cannot_be_research(tmp_path):
    row = candidate_pool()[0]
    with pytest.raises(ValueError, match="Reference fixtures"):
        require_teacher([checked(row, reference(row.truth))])
    path = tmp_path / "legacy.jsonl"
    write_jsonl(path, [{"id": row.id, "text": reference(row.truth)}])
    with pytest.raises(ValueError, match="unversioned"):
        require_teacher(imported([row], path))
    with pytest.raises(ValueError):
        TeacherIdentity(model="https://endpoint.example", revision="v1", access="api")


def test_training_cli_import_does_not_load_teacher_provider():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import duplex.cli; assert 'duplex.teacher' not in sys.modules; assert not any(x in sys.modules for x in ('openai', 'anthropic'))",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0


def test_exposure_is_bounded_not_a_default_100_steps():
    plan = exposure_plan([{"audio_frames": 375}] * 100, 4, 2, 500, 500)
    assert plan["steps"] == 50
    assert plan["approx_effective_epochs"] == 2
    assert plan["maximum_example_passes"] == 2
    assert exposure_plan([{"audio_frames": 375}] * 100, 4, 2, 7, 500)["steps"] == 7
    assert Config().train.max_steps is None


def test_complete_workload_reduces_steps_and_retains_evaluation(tmp_path):
    from duplex.experiment import aggregate_plan

    encoded = tmp_path / "encoded"
    encoded.mkdir()
    for index in range(100):
        path = encoded / f"plan-fixture-{index}.safetensors"
        path.write_bytes(b"planning-metadata-only")
        atomic_json(path.with_suffix(".json"), {"split": "train", "audio_frames": 375})
    prepared = tmp_path / "prepared.json"
    atomic_json(
        prepared,
        {
            "kind": "checked_teacher_audio",
            "split_audio_s": {"train": 3000, "dev": 120},
            "wall_s": 30,
        },
    )
    profile = tmp_path / "profile.json"
    atomic_json(
        profile,
        {
            "optimizer_step_s": 3,
            "audio_s_per_step": 120,
            "generation_rtf": 2,
            "calibration_load_s": 30,
        },
    )
    ledger = tmp_path / "ledger.json"
    atomic_json(ledger, {"sessions": [], "accounting": [{"metered_usd": 6, "ledger_s": 0}]})
    lock = tmp_path / "lock.json"
    atomic_json(lock, {"kind": "locked_user_audio_evaluation", "audio_s": 360, "examples": 12})
    plan = aggregate_plan(prepared, profile, ledger, Config(), 4.45, 0, lock, lock)
    assert plan["previous_estimated_usd"] == 6
    assert plan["fits"]
    assert 0 < plan["feasible_training_steps"] < 50
    assert plan["additional_estimated_usd"] <= 4.45
    assert plan["matched_evaluation_s"] >= 1800
    blocked = aggregate_plan(prepared, profile, ledger, Config(), 0.1, 0, lock, lock)
    assert not blocked["fits"] and blocked["feasible_training_steps"] == 0


def test_billing_accounting_preserves_history_and_caps_spend(tmp_path):
    from duplex.budget import BudgetConfig, Ledger

    book = Ledger(tmp_path / "budget.json", BudgetConfig())
    first = book.reserve(3300, now=100)
    book.close(first["id"], True, now=200)
    original = read_json(book.path)["sessions"]
    book.account(23.87, 15)
    assert read_json(book.path)["sessions"] == original
    with pytest.raises(ValueError, match="spend ceiling"):
        book.reserve(3300, now=300)
    # Later billing cannot silently lower the cumulative project cost floor.
    book.account(23.87, 0)
    assert read_json(book.path)["accounting"][-1]["metered_usd"] >= 15


def test_research_readiness_rejects_measurements(tmp_path):
    atomic_json(
        tmp_path / "prepared.json",
        {"kind": "preparation_measurement_only", "splits": {"train": 100, "dev": 12}},
    )
    with pytest.raises(ValueError, match="fixture/calibration"):
        research_readiness(tmp_path / "encoded", Config(experiment_kind="research"), None)


def test_perfect_base_cases_refuse_training(tmp_path):
    from duplex.io import file_hash

    encoded = tmp_path / "encoded"
    encoded.mkdir()
    hashes = {}
    for index in range(8):
        path = encoded / f"example-{index}.safetensors"
        path.write_bytes(b"metadata-gate-fixture-not-a-training-sample")
        hashes[path.name] = file_hash(path)
        atomic_json(
            path.with_suffix(".json"),
            {
                "group_id": str(index),
                "split": "train" if index < 4 else "dev",
                "supervision_provenance": {
                    "supervision_kind": "checked_teacher",
                    "teacher_contract": CONTRACT,
                },
            },
        )
    atomic_json(
        tmp_path / "prepared.json",
        {"kind": "checked_teacher_audio", "splits": {"train": 4, "dev": 4}, "encoded": hashes},
    )
    baseline = tmp_path / "baseline.json"
    atomic_json(
        baseline,
        {
            "kind": "locked_base_baseline",
            "evidence": "audible_asr",
            "count": 8,
            "all_cases_solved": True,
        },
    )
    with pytest.raises(ValueError, match="Base already solved"):
        research_readiness(
            encoded, Config(experiment_kind="research", min_train_examples=4), baseline
        )


def test_adapter_disable_is_exact_base_forward():
    torch.manual_seed(3)
    base = torch.nn.Linear(16, 16)
    layer = LoRALinear(base, 8, 8, 0)
    with torch.no_grad():
        layer.b.fill_(0.2)
    x = torch.randn(2, 16)
    enabled = layer(x)
    enable_adapters(layer, False)
    assert torch.equal(layer(x), base(x))
    assert not torch.equal(layer(x), enabled)


def test_evaluation_user_audio_has_no_target_continuation(tmp_path):
    class Renderer:
        backend, version, voice = "test-only", "fixture", "fixture"

        def render(self, text):
            return np.ones(24000, dtype=np.float32) * 0.01, []

    class Tokenizer:
        def encode(self, text):
            return [4, 5]

    rows = select_groups(candidate_pool(), "dev", 1)
    result = prepare_evaluation(rows, tmp_path, Renderer(), Tokenizer())
    assert result["teacher_required"] is False
    assert not list(tmp_path.rglob("assistant.wav"))
    for row in rows:
        manifest = read_json(tmp_path / "audio" / row.id / "manifest.json")
        assert not manifest["assistant_continuation_supplied"]
    from duplex.data import load_examples

    for row in load_examples(tmp_path / "rendered.jsonl"):
        assert row.supervision is None
        assert not any(u.kind == "response" for u in row.script)
        assert set(row.visible.model_dump()) == {"request_id", "user_audio", "events", "duration_s"}
