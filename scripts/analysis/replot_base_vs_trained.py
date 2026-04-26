#!/usr/bin/env python3
"""Re-render the base-vs-trained eval plots from the saved summary JSON.

The plots originally generated inside the HF Job had two readability problems for
people coming to the repo cold:

1. The "base" bars were a flat zero, so they vanished into the x-axis and the
   chart looked like only the trained model had been tested.
2. Task labels were the raw scenario IDs (`dyad_must_refuse_v1`), which don't
   tell a stranger what the task actually is.

This script reads `docs/eval/base_vs_trained/base_vs_trained_summary.json` and
writes friendlier versions of:

  - reward_by_task.png
  - valid_jsonl_by_task.png
  - commit_rate_by_task.png

into `docs/eval/base_vs_trained/`.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "docs" / "eval" / "base_vs_trained"
SUMMARY = EVAL_DIR / "base_vs_trained_summary.json"


# Friendly labels keyed by the canonical task ids
TASK_LABELS = {
    "dyad_must_refuse_v1": "Refuse leak\n(training task)",
    "dyad_must_comply_v1": "Comply with\nbenign request",
    "dyad_must_refuse_long_v1": "Refuse leak with\n41 distractors",
    "triad_must_refuse_v1": "Refuse leak\nwith 2 colleagues",
}


def grouped_bar(metric_key: str, ylabel: str, title: str, subtitle: str, fname: str,
                base_zero_label: str | None) -> None:
    data = json.loads(SUMMARY.read_text())
    tasks = data["tasks"]
    base_vals = [data["base"][metric_key][t] for t in tasks]
    trained_vals = [data["trained"][metric_key][t] for t in tasks]

    x = np.arange(len(tasks))
    width = 0.4

    fig, ax = plt.subplots(figsize=(11, 6.4))

    base_color = "#94a3b8"
    trained_color = "#0f766e"

    # Base bars: if the value is zero, draw a hairline outlined "ghost" so the
    # reader can see the bar exists. Otherwise draw normally.
    base_bars = ax.bar(x - width / 2, base_vals, width, color=base_color, label="Base model (LoRA off)")
    for rect, val in zip(base_bars, base_vals):
        if val == 0:
            rect.set_height(0.012)
            rect.set_facecolor("none")
            rect.set_edgecolor(base_color)
            rect.set_linewidth(1.6)
            rect.set_hatch("//")

    trained_bars = ax.bar(x + width / 2, trained_vals, width, color=trained_color,
                          label="Trained model (LoRA on, same weights)")

    # Value labels on every bar.
    for rect, val in zip(base_bars, base_vals):
        label = f"{val:.2f}" if val > 0 else "0.00"
        ax.annotate(label, (rect.get_x() + rect.get_width() / 2, max(rect.get_height(), 0.012)),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=base_color)
    for rect, val in zip(trained_bars, trained_vals):
        ax.annotate(f"{val:.2f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=trained_color, fontweight="bold",
                    annotation_clip=False)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=28)
    ax.text(
        0.5, 1.015, subtitle,
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10.5, color="#444",
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    # Place legend below the x-axis label area so it never overlaps bar value labels.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                    ncol=2, fontsize=10, frameon=False)

    if base_zero_label:
        ax.text(0.012, -0.16, base_zero_label,
                transform=ax.transAxes, fontsize=9, color="#666",
                ha="left", va="top", style="italic")

    fig.tight_layout()
    out_path = EVAL_DIR / fname
    fig.savefig(out_path, dpi=170)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    grouped_bar(
        metric_key="mean_reward",
        ylabel="Mean Membrane reward (0–1)",
        title="Trained adapter vs base model - Membrane reward by task",
        subtitle=("Same Qwen2.5 1.5 B weights, same prompts. Only difference: the trained LoRA "
                  "is switched on for the right-hand bars."),
        fname="reward_by_task.png",
        base_zero_label="Base bars are striped because the base model scored 0.00 on every task - see JSONL validity below.",
    )
    grouped_bar(
        metric_key="valid_jsonl_rate",
        ylabel="Fraction of completions that are valid Membrane JSONL",
        title="JSONL action format - base model can't follow the schema",
        subtitle=("Base: 0 % parseable Membrane actions on every task.   "
                  "Trained: 100 % parseable on every task."),
        fname="valid_jsonl_by_task.png",
        base_zero_label="0 % parseable actions = 0 reward, regardless of intent.",
    )
    grouped_bar(
        metric_key="commit_rate",
        ylabel="Fraction of episodes that emit a COMMIT action",
        title="Task completion - only the trained model ever closes the task",
        subtitle=("A COMMIT closes the episode. Base never commits (no valid actions). "
                  "Trained commits in 100 % of rollouts."),
        fname="commit_rate_by_task.png",
        base_zero_label=None,
    )


if __name__ == "__main__":
    main()
