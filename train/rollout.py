"""Run one episode until terminal; return final scalar reward (Total)."""
from __future__ import annotations

import json
from typing import Callable, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from models import MembraneAction, MembraneObservation
from server.environment import MembraneEnvironment

# Keep in sync with ``server.environment.METRICS_MARKER`` (avoid import cycles).
_METRICS_MARKER = "\n__MEMBRANE_METRICS__:"
_ALLOWED_SURFACES = frozenset(
    {"USER_REPLY", "AGENT_DM", "TEAM_MEMORY", "TOOL_PAYLOAD", "RUN_LOG"}
)


def _parse_observation_http(data: dict) -> MembraneObservation:
    """Support OpenEnv HTTP shape ``{observation, reward, done}`` and flat stub JSON."""
    inner = data.get("observation")
    if isinstance(inner, dict):
        merged = dict(inner)
        merged["reward"] = data.get("reward")
        merged["done"] = bool(data.get("done", False))
        merged.setdefault("metadata", {})
        msg = str(merged.get("message", ""))
        if _METRICS_MARKER in msg and merged.get("done"):
            base, _, rest = msg.partition(_METRICS_MARKER)
            merged["message"] = base.rstrip()
            try:
                merged["metadata"]["terminal_metrics"] = json.loads(rest)
            except json.JSONDecodeError:
                merged["metadata"]["terminal_metrics_parse_error"] = rest[:500]
        return MembraneObservation(**merged)
    obs = MembraneObservation(**data)
    if _METRICS_MARKER in str(obs.message) and obs.done:
        base, _, rest = str(obs.message).partition(_METRICS_MARKER)
        md = dict(obs.metadata)
        try:
            md["terminal_metrics"] = json.loads(rest)
        except json.JSONDecodeError:
            md["terminal_metrics_parse_error"] = rest[:500]
        return MembraneObservation(
            done=obs.done,
            reward=obs.reward,
            episode_goal=obs.episode_goal,
            message=base.rstrip(),
            visible_fact_ids=obs.visible_fact_ids,
            metadata=md,
        )
    return obs


def run_episode(
    task_id: str,
    policy: Callable[[MembraneObservation, int], MembraneAction],
    max_steps: int = 64,
    base_url: Optional[str] = None,
) -> float:
    obs = _reset(task_id, base_url=base_url)
    terminal: float | None = None
    step = 0
    while not obs.done and step < max_steps:
        step += 1
        action = policy(obs, step)
        obs = _step(action, base_url=base_url)
        if obs.reward is not None:
            if obs.done:
                terminal = float(obs.reward)
    if terminal is None:
        return 0.0
    return terminal


def collect_returns(
    task_id: str,
    policy: Callable[[MembraneObservation, int], MembraneAction],
    n_episodes: int,
    base_url: Optional[str] = None,
) -> List[float]:
    return [run_episode(task_id, policy, base_url=base_url) for _ in range(n_episodes)]


def _dict_to_action(d: dict) -> MembraneAction:
    keys = (
        "verb",
        "surface",
        "content",
        "disclosure_tier",
        "target_agent",
        "reason",
        "refusal_kind",
        "acting_as",
        "metadata",
    )
    filtered = {k: d[k] for k in keys if k in d and d[k] is not None}
    if "verb" not in filtered:
        filtered["verb"] = "QUERY"
    if "content" not in filtered:
        filtered["content"] = ""
    surf = filtered.get("surface")
    if surf is not None and surf not in _ALLOWED_SURFACES:
        filtered.pop("surface", None)
    return MembraneAction(**filtered)


def run_episode_from_action_jsonl(
    task_id: str,
    action_jsonl: str,
    *,
    base_url: Optional[str] = None,
    max_steps: int = 64,
) -> float:
    """
    GRPO / Unsloth hook: one string per line, each line JSON for one ``MembraneAction``,
    replayed in order until ``done`` or lines or ``max_steps`` exhausted.

    If the model stops early without ``COMMIT``, returns **0.0** (sparse terminal reward).
    """
    lines = [ln.strip() for ln in action_jsonl.strip().splitlines() if ln.strip()]
    obs = _reset(task_id, base_url=base_url)
    terminal: float | None = None
    steps = 0
    for line in lines:
        if obs.done or steps >= max_steps:
            break
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            return 0.0
        if not isinstance(raw, dict):
            return 0.0
        try:
            action = _dict_to_action(raw)
            steps += 1
            obs = _step(action, base_url=base_url)
        except Exception:
            # Bad JSON fields, Pydantic validation, or env rejects the action — treat as failed rollout.
            return 0.0
        if obs.done and obs.reward is not None:
            terminal = float(obs.reward)
            break
    if terminal is not None:
        return terminal
    return 0.0


def _post_json(url: str, body: dict) -> dict:
    req = Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=10) as resp:  # noqa: S310 - user-supplied URL by design
            return json.loads(resp.read().decode("utf-8"))
    except URLError as exc:  # pragma: no cover - depends on runtime network
        raise RuntimeError(
            f"HTTP rollout failed for {url}. Is the server running and reachable?"
        ) from exc


def _reset(task_id: str, base_url: Optional[str]) -> MembraneObservation:
    if base_url:
        data = _post_json(base_url.rstrip("/") + "/reset", {"task_id": task_id})
        return _parse_observation_http(data)
    env = MembraneEnvironment()
    # Keep a process-global singleton by storing on function attribute.
    _reset._env = env  # type: ignore[attr-defined]
    return env.reset(task_id=task_id)


def _step(action: MembraneAction, base_url: Optional[str]) -> MembraneObservation:
    if base_url:
        data = _post_json(base_url.rstrip("/") + "/step", {"action": _to_dict(action)})
        return _parse_observation_http(data)
    env = getattr(_reset, "_env", None)
    if env is None:
        env = MembraneEnvironment()
        _reset._env = env  # type: ignore[attr-defined]
    return env.step(action)


def _to_dict(action: MembraneAction) -> dict:
    if hasattr(action, "model_dump"):
        return action.model_dump()  # type: ignore[no-any-return]
    return {
        "verb": action.verb,
        "surface": action.surface,
        "content": action.content,
        "disclosure_tier": action.disclosure_tier,
        "target_agent": action.target_agent,
        "reason": action.reason,
        "refusal_kind": action.refusal_kind,
        "acting_as": getattr(action, "acting_as", None),
        "metadata": action.metadata,
    }
