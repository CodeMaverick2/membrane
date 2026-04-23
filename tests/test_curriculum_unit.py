"""Curriculum helper over failure tags."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tasks.curriculum import next_episode_config, next_task_id  # noqa: E402


def test_curriculum_routes_unsafe_compliance():
    assert next_task_id("dyad_schedule_v1", ["unsafe_compliance"]) == "dyad_must_refuse_v1"


def test_curriculum_routes_over_refusal():
    assert next_task_id("dyad_schedule_v1", ["over_refusal"]) == "dyad_must_comply_v1"


def test_curriculum_low_reward_drills_schedule():
    assert next_task_id("dyad_must_refuse_v1", [], total_reward=0.1) == "dyad_schedule_v1"


def test_curriculum_noop_when_ok():
    assert next_task_id("dyad_schedule_v1", [], total_reward=0.9) == "dyad_schedule_v1"


def test_next_episode_config_adds_ood_on_proxy():
    cfg = next_episode_config("dyad_must_comply_v1", ["proxy_rubric_bypass"])
    assert cfg["task_id"] == "dyad_schedule_v1"
    assert cfg["ood_axis"] == "POLICY"


def test_curriculum_proxy_evidence_drills_schedule():
    assert (
        next_task_id("dyad_must_comply_v1", ["proxy_rubric_bypass"])
        == "dyad_schedule_v1"
    )
    assert (
        next_task_id("dyad_must_comply_v1", ["missing_user_channel"])
        == "dyad_schedule_v1"
    )
