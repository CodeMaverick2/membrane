"""Load scenario specs from JSON/YAML and apply lightweight OOD mutations."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

from tasks.scenarios import FactSpec, RefusalGold, ScenarioSpec


def _read_data(path: Path) -> Dict[str, Any]:
    ext = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")
    if ext in (".json",):
        return json.loads(raw)
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML requested but PyYAML is not installed. Use JSON or install pyyaml."
            ) from exc
        return yaml.safe_load(raw) or {}
    raise ValueError(f"Unsupported scenario file extension: {ext}")


def _coerce_scenario(item: Dict[str, Any]) -> ScenarioSpec:
    facts = [FactSpec(**f) for f in item.get("facts", [])]
    rg = RefusalGold(**item.get("refusal_gold", {}))
    max_steps = max(1, int(item.get("max_steps", 20)))
    return ScenarioSpec(
        id=item["id"],
        difficulty=item.get("difficulty", "easy"),
        max_steps=max_steps,
        episode_goal=item.get("episode_goal", ""),
        actors=list(item.get("actors", ["PersonalAgent_A", "PersonalAgent_B"])),
        turn_mode=item.get("turn_mode", "single"),
        turn_order=list(item.get("turn_order", [])),
        commit_actor_id=item.get("commit_actor_id"),
        inbox_script=list(item.get("inbox_script", [])),
        require_fact_ids_queried_before_commit=list(
            item.get("require_fact_ids_queried_before_commit", [])
        ),
        require_user_reply_send_before_commit=bool(
            item.get("require_user_reply_send_before_commit", False)
        ),
        facts=facts,
        forbidden_substrings=list(item.get("forbidden_substrings", [])),
        forbidden_weights=item.get("forbidden_weights"),
        ood_axis=item.get("ood_axis"),
        commit_success_markers=dict(item.get("commit_success_markers", {"meeting_confirmed": True})),
        refusal_gold=rg,
    )


def load_scenarios_from_file(path: str | Path) -> Dict[str, ScenarioSpec]:
    p = Path(path)
    data = _read_data(p)
    items = data.get("scenarios", []) if isinstance(data, dict) else []
    out: Dict[str, ScenarioSpec] = {}
    for item in items:
        sc = _coerce_scenario(item)
        out[sc.id] = sc
    return out


def apply_ood_mutation(scenario: ScenarioSpec, axis: str) -> ScenarioSpec:
    """
    Lightweight deterministic OOD transformations.
    axis: TASK | POLICY | TOOL_PROTOCOL | SOCIAL_GRAPH
    """
    sc = copy.deepcopy(scenario)
    sc.ood_axis = axis
    a = axis.upper()
    if a == "TASK":
        sc.episode_goal = "[OOD TASK] " + sc.episode_goal + " Also include a backup plan."
    elif a == "POLICY":
        extra_tok = "POLICY-DELTA-OMEGA"
        if extra_tok not in sc.forbidden_substrings:
            sc.forbidden_substrings.append(extra_tok)
        fw = dict(sc.forbidden_weights or {})
        fw[extra_tok] = max(fw.values()) + 0.5 if fw else 1.5
        sc.forbidden_weights = fw
        sc.episode_goal = "[OOD POLICY] " + sc.episode_goal
    elif a == "TOOL_PROTOCOL":
        sc.episode_goal = (
            "[OOD TOOL_PROTOCOL] "
            + sc.episode_goal
            + " Use tool payload key meeting_slot_v2 instead of meeting_slot."
        )
    elif a == "SOCIAL_GRAPH":
        # Restrict visible facts to A only for half the facts (harder coordination).
        for idx, f in enumerate(sc.facts):
            if idx % 2 == 0:
                f.visible_to = ["PersonalAgent_A"]
        sc.episode_goal = "[OOD SOCIAL_GRAPH] " + sc.episode_goal
    return sc

