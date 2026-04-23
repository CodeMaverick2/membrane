"""Curriculum: task routing + optional OOD axis for harder follow-up episodes (Theme 4)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def next_task_id(
    prev_task_id: str,
    failure_tags: List[str],
    total_reward: Optional[float] = None,
) -> str:
    """
    Return a follow-up task id given the last episode outcome.

    This is intentionally small and deterministic so training scripts can
    swap tasks without a full procedural generator.
    """
    tags = {t.lower() for t in failure_tags}
    if "unsafe_compliance" in tags:
        return "dyad_must_refuse_v1"
    if "over_refusal" in tags:
        return "dyad_must_comply_v1"
    if tags.intersection(
        {"proxy_rubric_bypass", "missing_fact_evidence", "missing_user_channel"}
    ):
        return "dyad_schedule_v1"
    if total_reward is not None and total_reward < 0.35:
        return "dyad_schedule_v1"
    return prev_task_id


def next_episode_config(
    prev_task_id: str,
    failure_tags: List[str],
    total_reward: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Structured follow-up for training loops: task id plus optional ``ood_axis``
    for ``reset(..., ood_axis=...)`` (harder variant of the same scenario class).

    OOD axes are defined in ``tasks.scenario_loader.apply_ood_mutation``.
    """
    tags = {t.lower() for t in failure_tags}
    tid = next_task_id(prev_task_id, failure_tags, total_reward=total_reward)
    ood_axis: Optional[str] = None
    if tags.intersection({"proxy_rubric_bypass", "missing_fact_evidence", "missing_user_channel"}):
        ood_axis = "POLICY"
    elif "unsafe_compliance" in tags and prev_task_id == "triad_must_refuse_v1":
        ood_axis = "TASK"
    elif total_reward is not None and total_reward < 0.25:
        ood_axis = "SOCIAL_GRAPH"
    return {"task_id": tid, "ood_axis": ood_axis}
