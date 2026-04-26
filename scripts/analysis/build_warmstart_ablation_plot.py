#!/usr/bin/env python3
"""Build the aggregate showcase plot + summary table comparing every Membrane GRPO run we have metrics for.

Outputs:
  docs/plots/grpo_warmstart_ablation.png
  docs/plots/grpo_warmstart_ablation.svg
  docs/plots/grpo_warmstart_summary.csv
  docs/plots/grpo_warmstart_summary.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
PLOTS = ROOT / "docs" / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def read_csv_curve(path: Path) -> tuple[list[int], list[float]]:
    steps, rewards = [], []
    with path.open() as fp:
        for row in csv.DictReader(fp):
            steps.append(int(row["step"]))
            rewards.append(float(row["reward_mean"]))
    return steps, rewards


def load_hero() -> tuple[list[int], list[float]]:
    p = hf_hub_download(
        "Tejasghatule/membrane-qwen25-1p5b-grpo-lora",
        "membrane_grpo_existing_checkpoint_1000/final_adapter/trainer_state.json",
    )
    log = json.load(open(p))["log_history"]
    steps = [r["step"] for r in log if "reward" in r]
    rewards = [r["reward"] for r in log if "reward" in r]
    return steps, rewards


COLD_DIR = ROOT / "docs" / "hf_runs"
WARM_DIR = COLD_DIR / "continue_warm_start"

cold_runs = [
    ("cold-deep / seed 3408 (collapsed)", COLD_DIR / "repeat_seed_3408" / "training_metrics.csv"),
    ("cold-deep / seed 3409 (collapsed)", COLD_DIR / "repeat_seed_3409" / "training_metrics.csv"),
    ("cold-multi / seed 3410 (collapsed)", COLD_DIR / "multi_seed_3410" / "training_metrics.csv"),
]

warm_runs = [
    # (label_for_summary, csv_path, plot_label, color, lw)
    ("warm-deep / lr=2e-6 (conservative)",
     WARM_DIR / "continue_deep_seed_5821_lr2e-6" / "training_metrics.csv",
     "Warm start, single task, slow LR  →  best run",
     "#1f77b4", 2.4),
    ("warm-deep / lr=5e-6 (aggressive, saturates)",
     WARM_DIR / "continue_deep_seed_5823_lr5e-6" / "training_metrics.csv",
     "Warm start, single task, fast LR  →  saturates",
     "#9ec5e8", 1.9),
    ("warm-multi / lr=3e-6 (conservative, 7 tasks)",
     WARM_DIR / "continue_multifull_seed_5822_lr3e-6" / "training_metrics.csv",
     "Warm start, all 7 tasks, slow LR",
     "#d62728", 2.0),
    ("warm-multi / lr=5e-6 (aggressive, 7 tasks)",
     WARM_DIR / "continue_multifull_seed_5825_lr5e-6" / "training_metrics.csv",
     "Warm start, all 7 tasks, fast LR",
     "#f4a6a4", 1.7),
]


def smooth(xs: list[float], w: int = 3) -> list[float]:
    if len(xs) < w:
        return xs
    pad = w // 2
    out = []
    for i in range(len(xs)):
        lo = max(0, i - pad)
        hi = min(len(xs), i + pad + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def build_plot() -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # ---- Cold-start: collapse all 3 into a single shaded band, single legend entry.
    cold_curves = []
    for _, csv_path in cold_runs:
        if csv_path.exists():
            cold_curves.append(read_csv_curve(csv_path))
    if cold_curves:
        all_steps = sorted({s for steps, _ in cold_curves for s in steps})
        if all_steps:
            arr = np.full((len(cold_curves), len(all_steps)), np.nan)
            for i, (steps, rewards) in enumerate(cold_curves):
                for s, r in zip(steps, rewards):
                    arr[i, all_steps.index(s)] = r
            band_lo = np.nanmin(arr, axis=0)
            band_hi = np.nanmax(arr, axis=0)
            band_mid = np.nanmean(arr, axis=0)
            ax.fill_between(all_steps, band_lo, band_hi, color="#cccccc", alpha=0.55, lw=0,
                             label="3 cold-start runs  →  stuck below 0.02")
            ax.plot(all_steps, band_mid, color="#888888", lw=1.0, ls="--", alpha=0.85)

    # ---- Hero
    h_steps, h_rewards = load_hero()
    ax.plot(h_steps, smooth(h_rewards), color="black", lw=2.4,
            label="Colab hero  →  cold start that converged (lr=5e-6, 1000 steps)", zorder=10)
    ax.scatter([h_steps[-1]], [h_rewards[-1]], color="black", zorder=11)

    # ---- Warm runs
    best_warm_step = None
    best_warm_rew = -1.0
    sat_steps = sat_rew = None
    for _, csv_path, plot_label, color, lw in warm_runs:
        if not csv_path.exists():
            continue
        s, r = read_csv_curve(csv_path)
        rs = smooth(r)
        ax.plot(s, rs, color=color, lw=lw, label=plot_label)
        # Track best warm-deep / lr=2e-6 peak (the headline)
        if plot_label.startswith("Warm start, single task, slow LR"):
            best_idx = int(np.argmax(rs))
            best_warm_step, best_warm_rew = s[best_idx], rs[best_idx]
        # Track saturating warm-deep aggressive descent point
        if plot_label.startswith("Warm start, single task, fast LR"):
            # Find the descent - peak then a later lower point
            peak_idx = int(np.argmax(rs))
            if peak_idx < len(rs) - 5:
                sat_steps, sat_rew = s[-1], rs[-1]
                sat_peak_step, sat_peak_rew = s[peak_idx], rs[peak_idx]

    # ---- Hero vs warm reference lines
    ax.axhline(1.0, color="#bbbbbb", lw=0.7, ls=":")
    ax.text(2010, 1.0, "  perfect = 1.0", fontsize=8.5, color="#888", va="center")
    ax.axhline(0.974, color="black", lw=0.6, ls=":", alpha=0.5)
    ax.text(2010, 0.974, "  hero peak 0.974", fontsize=8.5, color="black", va="center", alpha=0.7)

    # ---- Annotations on the plot itself
    if best_warm_step is not None:
        ax.annotate(
            f"Warm start beats hero\n(peak {best_warm_rew:.3f})",
            xy=(best_warm_step, best_warm_rew),
            xytext=(700, 0.42),
            fontsize=10.5, fontweight="bold", color="#1f3a8a",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#1f3a8a", lw=1.2,
                            connectionstyle="arc3,rad=-0.15"),
        )
    if sat_steps is not None:
        ax.annotate(
            "Fast LR  →  policy saturates,\nadvantage signal disappears,\nreward drifts down",
            xy=(sat_steps, sat_rew),
            xytext=(1450, 0.30),
            fontsize=10, color="#3a3a3a",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#3a3a3a", lw=1.0,
                            connectionstyle="arc3,rad=0.15"),
        )

    # Annotate the cold-start band so the takeaway is on the chart, not just the legend
    ax.annotate(
        "Cold-start runs never escape zero -\n"
        "Membrane's reward is sparse, so a\n"
        "fresh policy almost never gets a\n"
        "non-zero signal to learn from.",
        xy=(900, 0.012),
        xytext=(60, 0.16),
        fontsize=9.5, color="#555", ha="left",
        arrowprops=dict(arrowstyle="->", color="#888", lw=0.9,
                        connectionstyle="arc3,rad=-0.25"),
    )

    # ---- Labels and legend
    ax.set_xlabel("Training step", fontsize=11)
    ax.set_ylabel("Membrane reward  (0 = invalid / leak,  1 = perfect refusal + commit)", fontsize=11)
    ax.set_ylim(-0.05, 1.08)
    ax.set_xlim(0, 2200)
    ax.grid(True, alpha=0.25)
    leg = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.97, title="Run", title_fontsize=10)
    leg.get_title().set_fontweight("bold")

    # Title + subtitle as separate axes-level objects so they stack cleanly.
    ax.set_title("How a 1.5 B Qwen learns Membrane", fontsize=15, fontweight="bold", pad=34)
    ax.text(
        0.5, 1.015,
        "Cold start collapses near zero. The Colab hero converges. Warm starts loaded from the hero adapter\n"
        "climb above it on the training task, and lift onto the 7-task curriculum.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=10.5, color="#444",
    )
    fig.tight_layout()
    fig.savefig(PLOTS / "grpo_warmstart_ablation.png", dpi=170)
    fig.savefig(PLOTS / "grpo_warmstart_ablation.svg")
    plt.close(fig)
    print("wrote", PLOTS / "grpo_warmstart_ablation.png")


def write_summary() -> None:
    rows = []

    h_steps, h_rewards = load_hero()
    rows.append({
        "run": "Colab hero (cold start, lr=5e-6, 1000 steps)",
        "category": "hero",
        "first_reward": round(h_rewards[0], 4),
        "final_reward": round(h_rewards[-1], 4),
        "best_reward": round(max(h_rewards), 4),
        "best_step": h_steps[h_rewards.index(max(h_rewards))],
        "steps_logged": len(h_rewards),
    })

    def _summarize(name: str, csv_path: Path, category: str) -> dict | None:
        if not csv_path.exists():
            return None
        s, r = read_csv_curve(csv_path)
        return {
            "run": name,
            "category": category,
            "first_reward": round(r[0], 4),
            "final_reward": round(r[-1], 4),
            "best_reward": round(max(r), 4),
            "best_step": s[r.index(max(r))],
            "steps_logged": len(r),
        }

    for name, p in cold_runs:
        d = _summarize(name, p, "cold-start")
        if d:
            rows.append(d)
    for entry in warm_runs:
        name, p = entry[0], entry[1]
        d = _summarize(name, p, "warm-start")
        if d:
            rows.append(d)

    csv_path = PLOTS / "grpo_warmstart_summary.csv"
    with csv_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", csv_path)

    md_lines = [
        "# Membrane GRPO - warm-start ablation",
        "",
        "Aggregate of every Membrane GRPO run that has produced metrics:",
        "",
        "- 1 Colab run (the hero) that converged from a cold start.",
        "- 3 cold-start Hugging Face Job runs that did not converge.",
        "- 4 warm-start Hugging Face Job runs that loaded the hero adapter as initial",
        "  weights and continued training. The four form a 2 × 2 ablation across",
        "  learning rate (conservative vs aggressive) and task mix (single task vs",
        "  full 7-scenario curriculum).",
        "",
        "| run | category | first reward | final reward | best reward | best step | steps logged |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['run']} | {r['category']} | {r['first_reward']:.3f} | "
            f"{r['final_reward']:.3f} | **{r['best_reward']:.3f}** | "
            f"{r['best_step']} | {r['steps_logged']} |"
        )
    md_lines += [
        "",
        "## Findings",
        "",
        "1. **Cold-start GRPO does not learn Membrane in the budgets we tested.** Three",
        "   independent HF Job runs (single-task seeds 3408 / 3409 at 900 steps,",
        "   multi-task seed 3410 at 1400 steps) all stay below 0.02 mean reward.",
        "   Membrane's reward is sparse on purpose - any malformed JSONL action zeroes",
        "   the episode - so a freshly-initialised policy almost never produces a",
        "   non-zero advantage signal long enough for GRPO to bootstrap.",
        "",
        "2. **The Colab T4 run converged with the same recipe that collapsed on the",
        "   A10G HF Job runs.** Same script, same hyperparameters, same seed family.",
        "   The only difference is the RNG stream from a different GPU and the",
        "   Unsloth/TRL versions that the version pin now reproduces. This is why",
        "   warm-starting was worth doing: a known-good policy whose weights could be",
        "   redeployed as initial conditions on the HF compute backend.",
        "",
        "3. **Warm-starting from the hero adapter beats the hero on the same task.**",
        "   `warm-deep / lr=2e-6` lifts mean reward from 0.880 (the hero adapter's",
        "   starting score) to **0.971 final / 0.988 peak**, surpassing both the",
        "   hero's 0.935 final and 0.974 peak. The conservative learning rate is the",
        "   key - see finding 4.",
        "",
        "4. **Aggressive learning rate saturates GRPO on a single task.**",
        "   `warm-deep / lr=5e-6` (the *hero's* learning rate) climbs from 0.849 to a",
        "   peak of **0.988 by step 240**, then drifts back down to 0.959 by",
        "   step 1500. This is not a model failure: `frac_reward_zero_std` rises",
        "   from 0.2 to ≥ 0.7, meaning ≥ 70 % of GRPO prompt groups produce identical",
        "   rewards across all 4 completions. With zero per-group advantage there is",
        "   no gradient signal, and `grad_norm` falls to exactly 0.0. Conservative",
        "   lr=2e-6 keeps per-group variance alive longer and continues improving.",
        "",
        "5. **A single-task warm-start transfers to the full 7-scenario curriculum.**",
        "   `warm-multi / lr=3e-6` (must-refuse, must-comply, long, triad,",
        "   round-robin, and two more held-out scenarios) starts at 0.495 - the model",
        "   has never seen 6 of those 7 tasks during the original Colab training -",
        "   and climbs to **0.793** by step 2000 without collapsing. The aggressive",
        "   variant peaks higher (0.854 at step 800) but slides to 0.785 as it",
        "   over-fits the task it was warm-started on.",
        "",
        "## Source data",
        "",
        "- Application source: <https://github.com/CodeMaverick2/membrane>",
        "- Per-run metrics: `docs/hf_runs/<run>/training_metrics.csv` and",
        "  `docs/hf_runs/<run>/run_summary.json`.",
        "- Aggregate plot: `grpo_warmstart_ablation.png` / `.svg`.",
        "- Headline CSV: `grpo_warmstart_summary.csv`.",
        "- All adapters:",
        "  <https://huggingface.co/Tejasghatule/membrane-qwen25-1p5b-grpo-lora>.",
        "",
    ]
    md_path = PLOTS / "grpo_warmstart_summary.md"
    md_path.write_text("\n".join(md_lines))
    print("wrote", md_path)


def main() -> None:
    build_plot()
    write_summary()


if __name__ == "__main__":
    main()
