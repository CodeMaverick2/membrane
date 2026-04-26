#!/usr/bin/env python3
"""Plot GRPO saturation evidence for the Blog: frac_reward_zero_std + grad_norm vs step.

Reads the aggressive warm-start single-task run logged under docs/hf_runs/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = (
    ROOT / "docs/hf_runs/continue_warm_start/continue_deep_seed_5823_lr5e-6/training_metrics.csv"
)
OUT = ROOT / "docs" / "plots" / "grpo_aggressive_lr_saturation.svg"


def main() -> None:
    csv_path = DEFAULT_CSV
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}")

    steps: list[int] = []
    frac: list[float] = []
    grad: list[float] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            fz = row.get("frac_reward_zero_std") or ""
            frac.append(float(fz) if fz not in ("", "nan") else float("nan"))
            g = row.get("grad_norm") or ""
            grad.append(float(g) if g not in ("", "nan") else float("nan"))

    fig, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.set_xlabel("Training step", fontsize=11)
    ax1.set_ylabel("Fraction of prompt groups with zero reward variance", fontsize=10, color="#b45309")
    (ln1,) = ax1.plot(steps, frac, color="#b45309", linewidth=2, marker="o", markersize=3, label="frac_reward_zero_std")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gradient norm (policy update size)", fontsize=10, color="#1d4ed8")
    (ln2,) = ax2.plot(steps, grad, color="#1d4ed8", linewidth=1.8, linestyle="--", marker="s", markersize=2, label="grad_norm")
    ax2.set_ylim(-0.02, max(0.5, max((g for g in grad if g == g), default=0.1) * 1.2))

    fig.suptitle(
        "Aggressive warm-start run (continue_deep_seed_5823_lr5e-6): GRPO loses signal when all completions tie",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    fig.legend([ln1, ln2], [ln1.get_label(), ln2.get_label()], loc="upper right", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
