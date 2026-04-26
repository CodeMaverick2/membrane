#!/usr/bin/env python3
"""Base vs trained — one vertical figure (readable), not three cramped columns.

The base model scores 0 on Membrane because it cannot emit valid JSONL actions;
that is the comparison point (not a missing baseline). This figure keeps
labels short and repeats the legend once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Short x labels — full names live in README / eval summary table
TASK_LABELS = {
    "dyad_must_refuse_v1": "Refuse leak\n(train)",
    "dyad_must_comply_v1": "Safe agree",
    "dyad_must_refuse_long_v1": "Refuse +\nlong noise",
    "triad_must_refuse_v1": "Refuse +\n2 bots",
}


def build_from_summary(summary: dict, out_path: Path) -> None:
    tasks = summary["tasks"]
    metrics = [
        ("mean_reward", "Membrane score (0–1)", "After full grader"),
        ("valid_jsonl_rate", "Valid JSONL (0–1)", "Membrane accepts the syntax"),
        ("commit_rate", "COMMIT rate (0–1)", "Episode finished cleanly"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), constrained_layout=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.14, hspace=0.42)

    fig.suptitle(
        "Same Qwen2.5 1.5B checkpoint — LoRA off (hatched) vs Membrane-trained LoRA on (solid)",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    for row_i, (ax, (metric_key, ylabel, short_hint)) in enumerate(zip(axes, metrics)):
        base_vals = [summary["base"][metric_key][t] for t in tasks]
        trained_vals = [summary["trained"][metric_key][t] for t in tasks]
        x = np.arange(len(tasks))
        width = 0.36
        base_color, trained_color = "#94a3b8", "#0f766e"

        bl = "LoRA off (base)" if row_i == 0 else "_"
        tl = "LoRA on (trained)" if row_i == 0 else "_"
        base_bars = ax.bar(x - width / 2, base_vals, width, color=base_color, label=bl)
        for rect, val in zip(base_bars, base_vals):
            if val == 0:
                rect.set_height(0.03)
                rect.set_facecolor("none")
                rect.set_edgecolor(base_color)
                rect.set_linewidth(1.6)
                rect.set_hatch("//")
        ax.bar(x + width / 2, trained_vals, width, color=trained_color, label=tl)

        for i, (bv, tv) in enumerate(zip(base_vals, trained_vals)):
            ax.text(
                i - width / 2, max(bv, 0.03) + 0.04, f"{bv:.2f}",
                ha="center", fontsize=9, color=base_color,
            )
            ax.text(
                i + width / 2, tv + 0.04, f"{tv:.2f}",
                ha="center", fontsize=9, color=trained_color, fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(short_hint, fontsize=10, pad=6)
        ax.grid(True, axis="y", alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=10,
        frameon=True, bbox_to_anchor=(0.5, 0.02),
    )

    fig.text(
        0.5, 0.085,
        "Base at 0.00 is expected: the frozen model does not produce valid Membrane JSONL, "
        "so the grader never scores a successful episode. Compare the green bars to the hatched bars.",
        ha="center", fontsize=9, color="#444",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build eval_showcase_panels.svg from base_vs_trained_summary.json")
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "eval" / "base_vs_trained" / "base_vs_trained_summary.json",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "plots" / "eval_showcase_panels.svg",
    )
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    build_from_summary(summary, args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
