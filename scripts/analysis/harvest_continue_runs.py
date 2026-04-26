#!/usr/bin/env python3
"""Pull per-step metrics from the four warm-start HF Jobs and write
training_metrics.csv + run_summary.json into docs/hf_runs/continue_warm_start/."""
from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "docs" / "hf_runs" / "continue_warm_start"

RUNS = [
    {
        "run_name": "continue_deep_seed_5821_lr2e-6",
        "job_id": "69ed3dd7d2c8bd8662bce8a9",
        "mode": "repeat",
        "seed": 5821,
        "learning_rate": 2e-6,
        "max_steps": 1500,
        "tasks": ["dyad_must_refuse_v1"],
        "label": "warm-deep / lr=2e-6 (conservative)",
    },
    {
        "run_name": "continue_multifull_seed_5822_lr3e-6",
        "job_id": "69ed3dd8d2c8bd8662bce8ab",
        "mode": "multi_full",
        "seed": 5822,
        "learning_rate": 3e-6,
        "max_steps": 2000,
        "tasks": "all_7_scenarios",
        "label": "warm-multi / lr=3e-6 (conservative)",
    },
    {
        "run_name": "continue_deep_seed_5823_lr5e-6",
        "job_id": "69ed4414d70108f37acdf171",
        "mode": "repeat",
        "seed": 5823,
        "learning_rate": 5e-6,
        "max_steps": 1500,
        "tasks": ["dyad_must_refuse_v1"],
        "label": "warm-deep / lr=5e-6 (aggressive)",
    },
    {
        "run_name": "continue_multifull_seed_5825_lr5e-6",
        "job_id": "69ed4418d2c8bd8662bce941",
        "mode": "multi_full",
        "seed": 5825,
        "learning_rate": 5e-6,
        "max_steps": 2000,
        "tasks": "all_7_scenarios",
        "label": "warm-multi / lr=5e-6 (aggressive)",
    },
]

METRIC_LINE_RE = re.compile(r"^\{'loss': .*'reward':.*\}$")


def parse_metric_line(line: str) -> dict | None:
    line = line.strip()
    if "'loss'" not in line or "'reward':" not in line:
        return None
    start = line.find("{'loss'")
    if start < 0:
        return None
    depth = 0
    end = -1
    for i in range(start, len(line)):
        ch = line[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return None
    blob = line[start:end]
    try:
        return ast.literal_eval(blob)
    except Exception:
        return None


def harvest(api: HfApi, run: dict) -> None:
    run_dir = OUT_ROOT / run["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"== {run['run_name']} ({run['job_id']}) ==")
    logs = list(api.fetch_job_logs(namespace="Tejasghatule", job_id=run["job_id"]))
    rows = []
    grad_steps = 0
    for line in logs:
        s = line.data if hasattr(line, "data") else str(line)
        for piece in s.splitlines():
            d = parse_metric_line(piece)
            if d is None:
                continue
            grad_steps += 1
            step = round(d.get("epoch", 0) * 256)  # not authoritative; we'll use grad_steps
            rows.append(
                {
                    "step": grad_steps * 20,  # logging_steps=20
                    "epoch": d.get("epoch"),
                    "reward_mean": d.get("reward"),
                    "reward_std": d.get("reward_std"),
                    "loss": d.get("loss"),
                    "grad_norm": d.get("grad_norm"),
                    "kl": d.get("kl"),
                    "frac_reward_zero_std": d.get("frac_reward_zero_std"),
                    "completion_mean_length": d.get("completions/mean_length"),
                    "entropy": d.get("entropy"),
                    "learning_rate": d.get("learning_rate"),
                }
            )

    if not rows:
        print("  no metrics rows parsed")
        return

    csv_path = run_dir / "training_metrics.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rewards = [r["reward_mean"] for r in rows if r["reward_mean"] is not None]
    summary = {
        "run_name": run["run_name"],
        "label": run["label"],
        "job_id": run["job_id"],
        "mode": run["mode"],
        "seed": run["seed"],
        "learning_rate": run["learning_rate"],
        "max_steps": run["max_steps"],
        "tasks": run["tasks"],
        "metrics": {
            "first_reward": rewards[0] if rewards else None,
            "final_reward": rewards[-1] if rewards else None,
            "best_reward": max(rewards) if rewards else None,
            "best_step": rows[max(range(len(rewards)), key=lambda i: rewards[i])]["step"]
            if rewards
            else None,
            "logged_points": len(rows),
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(
        f"  rows={len(rows)}  first={summary['metrics']['first_reward']:.3f}"
        f"  final={summary['metrics']['final_reward']:.3f}"
        f"  best={summary['metrics']['best_reward']:.3f} @ step {summary['metrics']['best_step']}"
    )


def main() -> None:
    api = HfApi()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        harvest(api, run)


if __name__ == "__main__":
    main()
