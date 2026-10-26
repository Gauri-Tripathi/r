from __future__ import annotations

from unittest.mock import Mock

import pytest

from duplex import volume_commit


def test_shell_flush_uses_bounded_sync_without_function_rpc(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_MODAL", "1")
    monkeypatch.setattr(volume_commit.Path, "is_dir", lambda self: True)
    runner = Mock()
    monkeypatch.setattr(volume_commit.subprocess, "run", runner)
    volume_commit.flush_mounts()
    runner.assert_called_once_with(["sync", "/data", "/runs", "/cache"], check=True, timeout=20)


def test_flush_rejects_missing_mounts(monkeypatch) -> None:
    monkeypatch.setenv("DUPLEX_MODAL", "1")
    monkeypatch.setattr(volume_commit.Path, "is_dir", lambda self: False)
    with pytest.raises(ValueError, match="mounts are missing"):
        volume_commit.flush_mounts()


def test_flush_rejects_local_execution(monkeypatch) -> None:
    monkeypatch.delenv("DUPLEX_MODAL", raising=False)
    with pytest.raises(ValueError, match="inside the project Modal shell"):
        volume_commit.flush_mounts()
