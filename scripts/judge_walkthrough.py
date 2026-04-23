#!/usr/bin/env python3
"""
Print a human-readable before/after for judges: bad trace vs good trace, same task.

Usage (from membrane/):
  python3 scripts/judge_walkthrough.py
  python3 scripts/judge_walkthrough.py --task triad_must_refuse_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import MembraneAction  # noqa: E402
from server.environment import MembraneEnvironment  # noqa: E402
from train.policies import policy_for_task  # noqa: E402


def _run_tagged(name: str, task_id: str) -> dict:
    env = MembraneEnvironment()
    obs = env.reset(task_id=task_id)
    step = 0
    trace = []
    while not obs.done and step < 48:
        step += 1
        pol = policy_for_task(task_id, name)
        act = pol(obs, step)
        trace.append({"step": step, "verb": act.verb, "surface": act.surface})
        obs = env.step(act)
    metrics = obs.metadata.get("metrics", {})
    return {"label": name, "trace": trace, "Total": metrics.get("Total"), "tags": obs.metadata.get("failure_tags", [])}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="dyad_must_refuse_v1")
    args = p.parse_args()
    task = args.task
    print(f"Task: {task}\n")
    bad = _run_tagged("baseline", task)
    good = _run_tagged("heuristic", task)
    print("=== Scripted “bad” policy (instant COMMIT or wrong pattern) ===")
    print(json.dumps(bad, indent=2))
    print("\n=== Scripted “good” policy (QUERY → REFUSE → USER_REPLY → COMMIT) ===")
    print(json.dumps(good, indent=2))
    print("\n--- Summary ---")
    print(f"Bad Total:  {bad['Total']}")
    print(f"Good Total: {good['Total']}")


if __name__ == "__main__":
    main()
