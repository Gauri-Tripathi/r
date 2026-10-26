from __future__ import annotations

import time

import pytest
from click.testing import CliRunner

from duplex.budget import BudgetConfig, Deadline, Ledger
from duplex.cli import cli
from duplex.fixtures import fixture


def test_deadline_requires_persistent_session(tmp_path, monkeypatch) -> None:
    path = tmp_path / "budget.json"
    book = Ledger(path, BudgetConfig())
    session = book.reserve(3300, now=time.time())
    monkeypatch.setenv("DUPLEX_LEDGER", str(path))
    monkeypatch.setenv("DUPLEX_SESSION", session["id"])
    monkeypatch.setenv("DUPLEX_DEADLINE", str(session["deadline"]))
    assert Deadline.from_env().remaining() > 3000
    monkeypatch.setenv("DUPLEX_DEADLINE", str(session["deadline"] + 100))
    with pytest.raises(ValueError, match="persistent"):
        Deadline.from_env()
    monkeypatch.delenv("DUPLEX_SESSION")
    with pytest.raises(ValueError, match="missing"):
        Deadline.from_env()


def test_fixture_adapter_cannot_be_released_as_speech(tmp_path) -> None:
    result = fixture(tmp_path / "fixture")
    response = CliRunner().invoke(
        cli, ["export", "--checkpoint", result["checkpoint"], "--out", str(tmp_path / "release")]
    )
    assert response.exit_code != 0
    assert not (tmp_path / "release").exists()
