#!/usr/bin/env python3
"""Roll out baseline vs heuristic; write CSV under docs/plots/ + optional summary JSON."""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.policies import policy_for_task  # noqa: E402
from train.rollout import run_episode  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Membrane P4 policy comparison (local env or deployed HTTP server)."
    )
    ap.add_argument("--task", default="dyad_must_refuse_v1")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument(
        "--url",
        default=None,
        help="Optional base URL (e.g. https://user-membrane-env.hf.space) to run via HTTP.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_ROOT / "docs" / "plots" / "episode_returns.csv",
    )
    ap.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write mean/std/min/max per policy for dashboards and judge packs.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed (local env only); makes scripted rollouts reproducible.",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)

    rows = []
    for label in ("baseline", "heuristic"):
        pol = policy_for_task(args.task, label)
        for ep in range(args.episodes):
            r = run_episode(args.task, pol, base_url=args.url)
            rows.append({"policy": label, "episode": ep, "return": r})

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "episode", "return"])
        w.writeheader()
        w.writerows(rows)

    def _stats(label: str) -> dict:
        vals = [x["return"] for x in rows if x["policy"] == label]
        if not vals:
            return {}
        return {
            "mean": sum(vals) / len(vals),
            "stdev": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    mean_b = sum(x["return"] for x in rows if x["policy"] == "baseline") / args.episodes
    mean_h = sum(x["return"] for x in rows if x["policy"] == "heuristic") / args.episodes
    print(f"Wrote {args.out}")
    if args.url:
        print(f"Mode: HTTP rollout via {args.url}")
    print(f"Mean return baseline={mean_b:.4f} heuristic={mean_h:.4f}")

    if args.summary_json:
        payload = {
            "task_id": args.task,
            "episodes_per_policy": args.episodes,
            "base_url": args.url,
            "seed": args.seed,
            "baseline": _stats("baseline"),
            "heuristic": _stats("heuristic"),
        }
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.summary_json}")


if __name__ == "__main__":
    main()
