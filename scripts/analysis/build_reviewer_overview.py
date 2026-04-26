#!/usr/bin/env python3
"""One composite figure for README / Blog / Hub: eval, scripted floor, Colab training, ablations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "docs" / "plots"
EVAL_SUMMARY = ROOT / "docs" / "eval" / "base_vs_trained" / "base_vs_trained_summary.json"
EPISODE_CSV = PLOTS / "episode_returns.csv"
TRAIN_METRICS = PLOTS / "grpo_training_metrics.csv"
COLD_CSV = ROOT / "docs" / "hf_runs" / "repeat_seed_3408" / "training_metrics.csv"
WARM_CSV = ROOT / "docs" / "hf_runs" / "continue_warm_start" / "continue_deep_seed_5821_lr2e-6" / "training_metrics.csv"
OUT = PLOTS / "reviewer_results_overview.svg"

TASK_LABELS = {
    "dyad_must_refuse_v1": "Refuse leak\n(train)",
    "dyad_must_comply_v1": "Safe agree",
    "dyad_must_refuse_long_v1": "Refuse +\nnoise",
    "triad_must_refuse_v1": "Refuse +\n2 bots",
}


def read_reward_curve(path: Path) -> tuple[list[int], list[float]]:
    steps, rewards = [], []
    with path.open(encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            steps.append(int(row["step"]))
            rewards.append(float(row["reward_mean"]))
    return steps, rewards


def main() -> None:
    if not EVAL_SUMMARY.exists():
        raise SystemExit(f"Missing {EVAL_SUMMARY}")
    summary = json.loads(EVAL_SUMMARY.read_text())
    tasks = summary["tasks"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2))
    fig.suptitle(
        "Membrane — results at a glance",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # --- A: trained eval scores (Membrane reward) ---
    ax = axes[0, 0]
    vals = [summary["trained"]["mean_reward"][t] for t in tasks]
    x = np.arange(len(tasks))
    ax.bar(x, vals, color="#0f766e", width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, min(v + 0.04, 1.05), f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Membrane score (0–1)", fontsize=10)
    ax.set_title("A · Trained LoRA — four held-out scenarios", fontsize=11, loc="left")
    ax.grid(True, axis="y", alpha=0.25)

    # --- B: Colab hero run (mean reward vs step) ---
    ax = axes[0, 1]
    if TRAIN_METRICS.exists():
        steps, rewards = read_reward_curve(TRAIN_METRICS)
        ax.plot(steps, rewards, color="#2563eb", linewidth=2)
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylabel("Mean reward (GRPO)", fontsize=10)
        ax.set_title("B · Colab hero run (same adapter others warm-start from)", fontsize=11, loc="left")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(0.5, 0.5, f"Missing\n{TRAIN_METRICS.name}", ha="center", va="center", transform=ax.transAxes)

    # --- C: scripted policies (not the neural net) ---
    ax = axes[1, 0]
    if EPISODE_CSV.exists():
        by_pol: dict[str, list[float]] = {"baseline": [], "heuristic": []}
        with EPISODE_CSV.open(encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                by_pol[row["policy"]].append(float(row["return"]))
        window = 5
        for label, color, name in (
            ("baseline", "#c0392b", "Weak scripted baseline"),
            ("heuristic", "#27ae60", "Hand-tuned scripted policy"),
        ):
            vals = by_pol[label]
            xs = list(range(len(vals)))
            smooth = [
                sum(vals[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
                for i in range(len(vals))
            ]
            ax.plot(xs, smooth, color=color, label=name, linewidth=2)
        ax.set_xlabel("Episode", fontsize=10)
        ax.set_ylabel("Score (0–1)", fontsize=10)
        ax.set_title("C · Scripted floor on refuse-leak (upper bound for rules, not GRPO)", fontsize=11, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
    else:
        ax.text(0.5, 0.5, f"Missing\n{EPISODE_CSV.name}", ha="center", va="center", transform=ax.transAxes)

    # --- D: cold vs warm (local CSVs) ---
    ax = axes[1, 1]
    plotted = False
    if COLD_CSV.exists():
        s, r = read_reward_curve(COLD_CSV)
        ax.plot(s, r, color="#94a3b8", linewidth=1.6, label="Cold start (HF Job, collapsed)")
        plotted = True
    if WARM_CSV.exists():
        s, r = read_reward_curve(WARM_CSV)
        ax.plot(s, r, color="#1f77b4", linewidth=2.2, label="Warm start, conservative LR")
        plotted = True
    if plotted:
        ax.set_xlabel("Training step", fontsize=10)
        ax.set_ylabel("Mean reward", fontsize=10)
        ax.set_title("D · Cold-start failure vs warm-start recovery (representative runs)", fontsize=11, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.text(
            0.5,
            0.5,
            "Add docs/hf_runs CSVs\nfor cold / warm curves",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
        )

    fig.text(
        0.5,
        0.02,
        "Full-resolution panels: eval_showcase_panels.svg, grpo_reward_curve.svg, baseline_vs_heuristic.svg, grpo_warmstart_ablation.svg in docs/plots/ (mirrored on the Hub showcase).",
        ha="center",
        fontsize=8.5,
        color="#444",
    )
    plt.subplots_adjust(left=0.07, right=0.98, top=0.91, bottom=0.10, hspace=0.36, wspace=0.28)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
