"""
Reward helpers for TRL GRPO + Unsloth when completions are Membrane trajectories.

Each **completion** should be newline-separated JSON objects, one ``MembraneAction`` per
line (same shape as HTTP ``/step``). Example::

    {"verb": "QUERY", "content": "team_calendar"}
    {"verb": "REFUSE", "reason": "no token", "refusal_kind": "SHOULD_NOT"}
    {"verb": "SEND", "surface": "USER_REPLY", "content": "Thursday 3pm works."}
    {"verb": "COMMIT", "content": "{\\"meeting_confirmed\\": true}"}

Use with ``GRPOTrainer(..., reward_funcs=[make_membrane_reward_fn(url, task_id)])``.
TRL may pass extra kwargs; this wrapper ignores them.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from train.rollout import run_episode_from_action_jsonl


def make_membrane_reward_fn(
    base_url: str,
    task_id: str,
    *,
    max_steps: int = 64,
) -> Callable[..., List[float]]:
    """Return a reward function ``f(completions, **_) -> list[float]``."""

    def reward_func(completions: List[str], **_: object) -> List[float]:
        out: List[float] = []
        for text in completions:
            r = run_episode_from_action_jsonl(
                task_id,
                text,
                base_url=base_url,
                max_steps=max_steps,
            )
            out.append(float(r))
        return out

    return reward_func


def make_membrane_reward_fn_local(
    task_id: str,
    *,
    max_steps: int = 64,
) -> Callable[..., List[float]]:
    """Same as ``make_membrane_reward_fn`` but in-process env (no HTTP)."""

    def reward_func(completions: List[str], **_: object) -> List[float]:
        out: List[float] = []
        for text in completions:
            r = run_episode_from_action_jsonl(
                task_id,
                text,
                base_url=None,
                max_steps=max_steps,
            )
            out.append(float(r))
        return out

    return reward_func
