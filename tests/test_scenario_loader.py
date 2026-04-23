"""Scenario loader + OOD mutator behavior."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from tasks.scenario_loader import apply_ood_mutation, load_scenarios_from_file  # noqa: E402
from tasks.scenarios import get_scenario  # noqa: E402
from server.environment import MembraneEnvironment  # noqa: E402


def test_load_scenarios_from_json_fixture():
    path = _ROOT / "tests" / "fixtures" / "custom_scenarios.json"
    out = load_scenarios_from_file(path)
    assert "custom_refusal_case" in out
    sc = out["custom_refusal_case"]
    assert sc.max_steps == 18
    assert sc.refusal_gold.kind == "must_refuse_before_safe_commit"
    assert sc.forbidden_weights is not None
    assert sc.forbidden_weights["CUSTOM-TOKEN-999"] == 2.0
    assert sc.require_fact_ids_queried_before_commit == []
    assert sc.require_user_reply_send_before_commit is False


def test_apply_ood_policy_mutation_adds_token():
    base = get_scenario("dyad_schedule_v1")
    m = apply_ood_mutation(base, "POLICY")
    assert m.ood_axis == "POLICY"
    assert "POLICY-DELTA-OMEGA" in m.forbidden_substrings
    assert m.forbidden_weights is not None
    assert m.forbidden_weights["POLICY-DELTA-OMEGA"] >= 1.5


def test_environment_reset_with_scenario_path_and_ood_axis():
    path = _ROOT / "tests" / "fixtures" / "custom_scenarios.json"
    env = MembraneEnvironment()
    obs = env.reset(task_id="custom_refusal_case", scenario_path=str(path), ood_axis="TASK")
    assert obs.metadata["task_id"] == "custom_refusal_case"
    assert obs.metadata["ood_axis"] == "TASK"
    assert obs.episode_goal.startswith("[OOD TASK]")

