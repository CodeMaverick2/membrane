"""Triad + conflicting pressure + must-refuse gold."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import MembraneAction  # noqa: E402
from server.environment import MembraneEnvironment  # noqa: E402
from train.policies import policy_for_task  # noqa: E402
from train.rollout import run_episode  # noqa: E402


def test_triad_must_refuse_heuristic_beats_baseline():
    task = "triad_must_refuse_v1"
    n = 8
    b = sum(run_episode(task, policy_for_task(task, "baseline")) for _ in range(n)) / n
    h = sum(run_episode(task, policy_for_task(task, "heuristic")) for _ in range(n)) / n
    assert h > b + 0.02


def test_triad_inbox_pressure_messages_present():
    env = MembraneEnvironment()
    env.reset(task_id="triad_must_refuse_v1")
    env.step(MembraneAction(verb="PLAN", content=""))
    o2 = env.step(MembraneAction(verb="PLAN", content=""))
    assert "URGENT" in o2.message or "PROJECT-CODE-ALPHA" in o2.message
