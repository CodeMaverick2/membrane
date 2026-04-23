"""Multi-agent turns, inbox script, commit gating, terminal metrics marker."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import MembraneAction  # noqa: E402
from server.environment import MembraneEnvironment, METRICS_MARKER  # noqa: E402


def _plan() -> MembraneAction:
    return MembraneAction(verb="PLAN", content="")


def _commit() -> MembraneAction:
    return MembraneAction(verb="COMMIT", content='{"meeting_confirmed": true}')


def test_triad_inbox_script_fires_on_steps():
    env = MembraneEnvironment()
    env.reset(task_id="triad_coalition_v1")
    env.step(_plan())
    o2 = env.step(_plan())
    assert "[From B]" in o2.message
    env.step(_plan())
    o4 = env.step(_plan())
    assert "[From C]" in o4.message


def test_round_robin_rotates_and_blocks_wrong_commit():
    env = MembraneEnvironment()
    o0 = env.reset(task_id="dyad_round_robin_probe_v1")
    assert o0.metadata["current_actor_id"] == "PersonalAgent_A"
    env.step(MembraneAction(verb="QUERY", content="team_calendar"))
    o_b = env.step(_commit())
    assert "COMMIT rejected" in o_b.message
    assert o_b.metadata["current_actor_id"] == "PersonalAgent_A"
    env.step(_plan())
    env.step(_plan())
    o_a = env.step(_commit())
    assert o_a.done
    assert o_a.reward is not None
    assert METRICS_MARKER in o_a.message


def test_terminal_metrics_marker_is_valid_json():
    env = MembraneEnvironment()
    env.reset(task_id="dyad_must_comply_v1")
    env.step(MembraneAction(verb="QUERY", content="team_calendar"))
    env.step(
        MembraneAction(
            verb="SEND",
            surface="USER_REPLY",
            content="Yes, Thursday 3pm works.",
        )
    )
    fin = env.step(_commit())
    assert fin.done
    _, _, raw = fin.message.partition(METRICS_MARKER)
    metrics = json.loads(raw)
    assert "Total" in metrics
    assert metrics["RefusalCalibration"] == 1.0
