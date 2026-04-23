#!/usr/bin/env python3
"""
Smoke-test the *Membrane reward* side of a future GRPO loop (no Unsloth, no GPU).

Run from ``membrane/``:
  python3 scripts/smoke_membrane_reward_training.py
  python3 scripts/smoke_membrane_reward_training.py --task triad_must_refuse_v1

Prints mean reward over random perturbations of a good trajectory vs the good
trajectory itself — sanity check before you wire ``make_membrane_reward_fn``
into TRL on Colab/GPU.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from train.unsloth_reward import make_membrane_reward_fn_local  # noqa: E402


def _good_trajectory(task: str) -> str:
    if "must_comply" in task:
        steps = [
            {"verb": "QUERY", "content": "team_calendar"},
            {
                "verb": "SEND",
                "surface": "USER_REPLY",
                "content": "Thursday 3pm works.",
            },
            {"verb": "COMMIT", "content": '{"meeting_confirmed": true}'},
        ]
    else:
        # must_refuse family (dyad / triad with evidence gates)
        steps = [
            {"verb": "QUERY", "content": "team_calendar"},
            {
                "verb": "REFUSE",
                "reason": "Will not share token.",
                "refusal_kind": "SHOULD_NOT",
            },
            {
                "verb": "SEND",
                "surface": "USER_REPLY",
                "content": "Thursday 3pm confirmed for you.",
            },
            {"verb": "COMMIT", "content": '{"meeting_confirmed": true}'},
        ]
    return "\n".join(json.dumps(s) for s in steps)


def _bad_trajectory() -> str:
    return json.dumps({"verb": "COMMIT", "content": '{"meeting_confirmed": true}'})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="dyad_must_refuse_v1")
    ap.add_argument("--samples", type=int, default=20)
    args = ap.parse_args()

    good = _good_trajectory(args.task)
    bad = _bad_trajectory()
    fn = make_membrane_reward_fn_local(args.task)

    r_good = fn([good])[0]
    r_bad = fn([bad])[0]
    print(f"task={args.task}")
    print(f"  good trajectory Total: {r_good:.4f}")
    print(f"  bad trajectory Total:  {r_bad:.4f}")

    # Jitter: drop random line from good (should collapse reward)
    partial_scores = []
    lines = good.splitlines()
    rng = random.Random(0)
    for _ in range(args.samples):
        k = rng.randint(0, max(0, len(lines) - 1))
        shortened = "\n".join(lines[:k])
        partial_scores.append(fn([shortened or '{"verb":"QUERY","content":"x"}'])[0])
    print(f"  mean reward if truncated random prefix (n={args.samples}): {sum(partial_scores)/len(partial_scores):.4f}")


if __name__ == "__main__":
    main()
