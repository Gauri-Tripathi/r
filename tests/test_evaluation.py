from __future__ import annotations

import pytest

from duplex.data import generate, reference
from duplex.evaluate import bootstrap_delta, score, summarize


def test_paired_group_uncertainty_and_no_fabricated_latency() -> None:
    rows = generate(20)
    scored = [score(r, reference(r.truth), [], 10) for r in rows]
    assert all(r["joint_success"] for r in scored)
    assert summarize(scored)["latency"]["p50_s"] is None
    paired = bootstrap_delta(scored, scored, "joint_success")
    assert paired["delta"] == 0 and paired["ci95"] == [0, 0]
    assert bootstrap_delta(scored[:4], scored[:4], "joint_success")["ci95"] is None
    with pytest.raises(ValueError, match="IDs differ"):
        bootstrap_delta(scored, scored[:-1], "joint_success")
