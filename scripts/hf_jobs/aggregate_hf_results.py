#!/usr/bin/env python3
"""Aggregate HF credit-run metrics into final judge-facing plots."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) if v not in ("", None) else 0.0 for k, v in row.items()})
    return rows


def maybe_metrics_paths(root: Path) -> list[Path]:
    paths = list(root.glob("**/training_metrics.csv"))
    paths += list(root.glob("**/grpo_training_metrics.csv"))
    return sorted(set(paths))


def label_for(path: Path) -> str:
    if path.name == "grpo_training_metrics.csv":
        return "colab_hero_1000"
    if path.name == "training_metrics.csv":
        for parent in path.parents:
            name = parent.name
            if name and name not in ("plots", "runs", ".", "hf_runs"):
                return name
    return path.stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="docs/plots")
    parser.add_argument("--out-dir", default="docs/plots")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_paths = maybe_metrics_paths(input_root)
    if not metric_paths:
        raise SystemExit(f"No metrics CSVs found under {input_root}")

    plt.rcParams.update({"figure.dpi": 160, "savefig.dpi": 220, "axes.grid": True, "grid.alpha": 0.25})

    summary = []
    fig, ax = plt.subplots(figsize=(9.5, 5))
    fig_loss, ax_loss = plt.subplots(figsize=(9.5, 5))
    fig_kl, ax_kl = plt.subplots(figsize=(9.5, 5))

    for path in metric_paths:
        rows = read_metrics(path)
        if not rows:
            continue
        steps = [r["step"] for r in rows]
        rewards = [r.get("reward_mean", 0.0) for r in rows]
        losses = [r.get("loss", 0.0) for r in rows]
        kls = [r.get("kl", 0.0) for r in rows]
        label = label_for(path)
        ax.plot(steps, rewards, linewidth=2.0, marker="o", markersize=2.5, label=label)
        ax_loss.plot(steps, losses, linewidth=1.8, marker="o", markersize=2.0, label=label)
        ax_kl.plot(steps, kls, linewidth=1.8, marker="o", markersize=2.0, label=label)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i])
        summary.append(
            {
                "label": label,
                "source": str(path),
                "first_step": steps[0],
                "first_reward": rewards[0],
                "final_step": steps[-1],
                "final_reward": rewards[-1],
                "best_step": steps[best_idx],
                "best_reward": rewards[best_idx],
            }
        )

    ax.set_title("Membrane GRPO Reward - All Runs")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean Membrane reward")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "combined_grpo_reward_curves.png")
    fig.savefig(out_dir / "combined_grpo_reward_curves.svg")
    plt.close(fig)

    ax_loss.set_title("Policy Loss - All Runs")
    ax_loss.set_xlabel("Training step")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(loc="upper right")
    fig_loss.tight_layout()
    fig_loss.savefig(out_dir / "combined_grpo_loss.png")
    plt.close(fig_loss)

    ax_kl.set_title("KL Divergence - All Runs")
    ax_kl.set_xlabel("Training step")
    ax_kl.set_ylabel("KL")
    ax_kl.legend(loc="upper right")
    fig_kl.tight_layout()
    fig_kl.savefig(out_dir / "combined_grpo_kl.png")
    plt.close(fig_kl)

    (out_dir / "combined_grpo_summary.json").write_text(json.dumps(summary, indent=2))

    md = ["# Combined GRPO Run Summary\n"]
    md.append("| run | first reward | final reward | best reward | best step | source |")
    md.append("|---|---|---|---|---|---|")
    for s in summary:
        md.append(
            f"| {s['label']} | {s['first_reward']:.4f} | {s['final_reward']:.4f} | "
            f"{s['best_reward']:.4f} | {s['best_step']} | `{s['source']}` |"
        )
    (out_dir / "combined_grpo_summary.md").write_text("\n".join(md))

    print(f"Wrote {out_dir / 'combined_grpo_reward_curves.png'}")
    print(f"Wrote {out_dir / 'combined_grpo_loss.png'}")
    print(f"Wrote {out_dir / 'combined_grpo_kl.png'}")
    print(f"Wrote {out_dir / 'combined_grpo_summary.md'}")


if __name__ == "__main__":
    main()
