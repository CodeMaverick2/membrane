"""Smoke tests: stub reset and P1 COMMIT terminal."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import MembraneAction  # noqa: E402
from server.environment import MembraneEnvironment  # noqa: E402


def test_reset_then_commit_terminates_with_metrics():
    env = MembraneEnvironment()
    obs0 = env.reset()
    assert obs0.done is False
    env.step(MembraneAction(verb="QUERY", content="team_calendar"))
    env.step(
        MembraneAction(
            verb="SEND",
            surface="USER_REPLY",
            content="Thursday 3pm works for me.",
        )
    )
    obs1 = env.step(
        MembraneAction(verb="COMMIT", content='{"meeting_confirmed": true}')
    )
    assert obs1.done is True
    assert "metrics" in obs1.metadata
    assert obs1.metadata["metrics"]["TaskUtility"] == 1.0
    assert obs1.metadata["metrics"]["EvidenceCompliance"] == 1.0
    assert obs1.reward == obs1.metadata["metrics"]["Total"]
