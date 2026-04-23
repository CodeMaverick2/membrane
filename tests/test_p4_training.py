"""P4: scripted policies — heuristic should beat baseline on refusal task."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from train.policies import policy_for_task  # noqa: E402
from train.rollout import run_episode  # noqa: E402


def test_heuristic_beats_baseline_on_must_refuse():
    task = "dyad_must_refuse_v1"
    n = 16
    b = sum(run_episode(task, policy_for_task(task, "baseline")) for _ in range(n)) / n
    h = sum(run_episode(task, policy_for_task(task, "heuristic")) for _ in range(n)) / n
    assert h > b + 0.05


def test_heuristic_beats_baseline_on_must_comply():
    task = "dyad_must_comply_v1"
    n = 16
    b = sum(run_episode(task, policy_for_task(task, "baseline")) for _ in range(n)) / n
    h = sum(run_episode(task, policy_for_task(task, "heuristic")) for _ in range(n)) / n
    assert h > b + 0.05


def test_heuristic_beats_baseline_on_must_refuse_long():
    task = "dyad_must_refuse_long_v1"
    n = 6
    b = sum(run_episode(task, policy_for_task(task, "baseline")) for _ in range(n)) / n
    h = sum(run_episode(task, policy_for_task(task, "heuristic")) for _ in range(n)) / n
    assert h > b + 0.05
