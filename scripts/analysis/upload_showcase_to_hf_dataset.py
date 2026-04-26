#!/usr/bin/env python3
"""Upload curated SVG figures to the HF dataset under showcase/.

Hugging Face dataset viewer sometimes mis-classifies repos; a dedicated
``showcase/`` folder with SVG figures + README makes comparisons obvious.

Requires HF_TOKEN (or huggingface-cli login).

Usage:
  cd membrane && HF_TOKEN=... .venv/bin/python scripts/analysis/upload_showcase_to_hf_dataset.py
"""
from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = os.environ.get("MEMBRANE_RESULTS_REPO", "Tejasghatule/membrane-grpo-results")

FILES = [
    ROOT / "docs/plots/eval_showcase_panels.svg",
    ROOT / "docs/plots/baseline_vs_heuristic.svg",
    ROOT / "docs/plots/grpo_warmstart_ablation.svg",
    ROOT / "docs/plots/grpo_reward_curve.svg",
    ROOT / "docs/plots/grpo_training_dashboard.svg",
    ROOT / "docs/plots/grpo_aggressive_lr_saturation.svg",
    ROOT / "docs/eval/base_vs_trained/reward_by_task.svg",
]


README = """# Membrane results — showcase

All files here are **SVG** (vector) plots from the Membrane repo.

| File | What it shows |
|------|-----------------|
| `eval_showcase_panels.svg` | **Neural model:** same Qwen2.5 1.5B with LoRA **off** vs **on** — reward, valid JSONL rate, and COMMIT rate for four scenarios (plain-language labels). |
| `baseline_vs_heuristic.svg` | **Scripted policies only** (not the neural net): weak baseline vs hand-tuned heuristic on the refuse-leak task — an upper bound for simple rules. |
| `grpo_warmstart_ablation.svg` | **Training:** cold-start vs Colab hero vs warm-start GRPO curves. |
| `grpo_reward_curve.svg` | **Training:** Colab hero run mean reward over steps. |
| `grpo_training_dashboard.svg` | **Training:** reward, loss, KL, completion length on one page. |
| `grpo_aggressive_lr_saturation.svg` | **Training diagnostics:** `frac_reward_zero_std` vs `grad_norm` for run `continue_deep_seed_5823_lr5e-6` (CSV in repo under `docs/hf_runs/...`). |
| `reward_by_task.svg` | Same reward comparison as the top panel of `eval_showcase_panels.svg`, larger single chart. |

Per-run metrics and eval CSVs live under `runs/` and `eval/` in this dataset.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN")

    api = HfApi(token=token)
    repo = os.environ.get("MEMBRANE_RESULTS_REPO", DEFAULT_REPO)

    api.upload_file(
        path_or_fileobj=BytesIO(README.encode("utf-8")),
        path_in_repo="showcase/README.md",
        repo_id=repo,
        repo_type="dataset",
        commit_message="Showcase: README for results figures",
    )

    for path in FILES:
        if not path.exists():
            print("skip missing:", path)
            continue
        rel = f"showcase/{path.name}"
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Showcase: update {path.name}",
        )
        print("uploaded", rel)

    print(f"Done: https://huggingface.co/datasets/{repo}/tree/main/showcase")


if __name__ == "__main__":
    main()
