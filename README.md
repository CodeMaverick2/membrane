---
title: MembraneEnv
emoji: 🛡️
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
license: mit
short_description: OpenEnv channel-wise privacy and refusal scoring
tags:
  - openenv
  - privacy
  - multi-agent
---

# Membrane

You ask your AI assistant to confirm Thursday's meeting. The reply is polite, the
tone is right, you move on. You don't see the other window where the same
assistant just dropped your project's internal codename into a colleague's DM.
The user-facing reply was clean. The agent leaked anyway.

**Membrane is the environment that makes that failure trainable.** It runs the
assistant in a small simulated workplace with one or two scripted colleagues that
push back, audits **5 separate channels** (user reply, agent DM, team memory, tool
payload, run log) for both task completion and privacy, and scores every channel
with a deterministic grader — no LLM judge, no soft scores. Train against it and
a 1.5 B Qwen goes from **0 % valid actions and 0.000 reward to 100 % valid actions
and 0.988 reward on the training task** in a single Colab T4 session.

**Want the story first?** Open **[`Blog.md`](Blog.md)** in this repo (same folder as this README). It is a short, plain-language walkthrough: why hidden channels matter, what the failed runs showed, and how to read the training curves — no RL background required. Everything below is the technical reference.

The agent (`PersonalAgent_A`) operates against 1–2 scripted colleague
actors (`PersonalAgent_B`, `PersonalAgent_C`) that inject conflicting
requests through an inbox channel, and is audited across **5 communication
surfaces** (`USER_REPLY`, `AGENT_DM`, `TEAM_MEMORY`, `TOOL_PAYLOAD`,
`RUN_LOG`). The reward replays each generated JSONL trajectory through the
live environment — there is no LLM in the reward path.

## Links

| | |
|---|---|
| **Source code (GitHub)** | <https://github.com/CodeMaverick2/membrane> |
| **Plain-language story (open this file)** | **[`Blog.md`](Blog.md)** — in the repo root, next to `README.md` |
| Hugging Face Space | <https://huggingface.co/spaces/Tejasghatule/membrane-temp> |
| Live Space endpoint | <https://Tejasghatule-membrane-temp.hf.space> |
| Trained adapters | <https://huggingface.co/Tejasghatule/membrane-qwen25-1p5b-grpo-lora> |
| Training metrics & plots | <https://huggingface.co/datasets/Tejasghatule/membrane-grpo-results> |
| Training notebook (1000-step hero run — open in Colab) | <https://colab.research.google.com/drive/1rEFKYNGbtoNZmClFDh8Q0aoeTdy7Xsrf?usp=sharing> |
| Same notebook in this repo | [`notebooks/membrane_train_colab.ipynb`](notebooks/membrane_train_colab.ipynb) |
| Warm-start ablation summary | [`docs/plots/grpo_warmstart_summary.md`](docs/plots/grpo_warmstart_summary.md) |
| Trained-vs-base eval | [`docs/eval/base_vs_trained/base_vs_trained_summary.md`](docs/eval/base_vs_trained/base_vs_trained_summary.md) |
| Results on the Hub (`showcase/`) | [`showcase/`](https://huggingface.co/datasets/Tejasghatule/membrane-grpo-results/tree/main/showcase) — same SVGs as [`docs/plots/`](docs/plots/) in this repo |

## Results at a glance

Four panels on one page (eval, Colab training run, scripted floor, cold vs warm-start). Full-size versions of each idea are in the numbered figures below and in the Hub [`showcase/`](https://huggingface.co/datasets/Tejasghatule/membrane-grpo-results/tree/main/showcase).

![Membrane results at a glance: panels A–D](docs/plots/reviewer_results_overview.svg)

## Results figures

Key plots live in [`docs/plots/`](docs/plots/) and in the dataset
[`showcase/`](https://huggingface.co/datasets/Tejasghatule/membrane-grpo-results/tree/main/showcase).

### 1. Neural model: base vs trained (same Qwen, LoRA on/off)

**Why are the base (LoRA off) bars at 0.00?** That is the honest floor for this
evaluation: the frozen instruction-tuned Qwen does **not** emit valid Membrane
JSONL (`verb` / `SEND` / `COMMIT` grammar), so every rollout fails grading.
You are not supposed to read progress “against” a non-zero base line here — the
**comparison is off → on** on the same weights. For a **non-zero scripted floor**
on the same environment, see **figure 4** (hand-written policies, not the neural
net). For **training progress**, see **figure 5** and the cold-start band vs the
Colab hero / warm-start curves.

Three **stacked** bar charts (less cramped than a single row of three): reward,
then valid JSONL, then COMMIT. Hatched = LoRA off, solid green = trained.

![Base vs trained: three stacked bar charts](docs/plots/eval_showcase_panels.svg)

### 2. Colab hero run (mean reward over training steps)

![GRPO mean reward vs step on the 1000-step Colab T4 run](docs/plots/grpo_reward_curve.svg)

### 3. Training dashboard (reward, loss, KL, length)

![Single-page training health for the Colab hero run](docs/plots/grpo_training_dashboard.svg)

### 4. Scripted policies only (not the neural net)

Two hand-written policies on the refuse-leak task: a weak baseline vs a
strong heuristic. This is **not** the GRPO model; it shows how much signal
the environment gives to simple rules before you add learning.

![Weak scripted baseline vs hand-tuned scripted policy on the same scenario](docs/plots/baseline_vs_heuristic.svg)

### 5. All GRPO training runs we logged (cold start, hero, warm-start)

Already used in the table section below; repeated here for a one-stop view.

![Cold-start band, Colab hero, warm-start ablations](docs/plots/grpo_warmstart_ablation.svg)

### 6. GRPO “saturation” when the policy stops learning (logged metrics)

Aggressive warm-start run `continue_deep_seed_5823_lr5e-6`: as more prompt groups
produce identical rewards across completions, `frac_reward_zero_std` rises and
`grad_norm` goes to zero. Source CSV:
`docs/hf_runs/continue_warm_start/continue_deep_seed_5823_lr5e-6/training_metrics.csv`.

![frac_reward_zero_std and grad_norm vs step for the aggressive warm-start run](docs/plots/grpo_aggressive_lr_saturation.svg)

## The problem

Assistants help with calendar and email-style work, but a realistic failure is
not always visible in the final answer. The user reply may look safe while the
agent leaks a secret to a colleague, stores it in team memory, or writes it into
a tool payload.

Membrane turns that failure mode into a trainable environment. The agent must
complete a normal delegation task, such as confirming a meeting, while handling
conflicting social pressure to reveal a private token. The environment scores
both sides: task completion and privacy-preserving refusal.

## What this project is

Membrane is an **OpenEnv** environment. A model sends structured JSONL actions:

- `QUERY` a visible fact, such as `team_calendar`
- `SEND` text to a named surface: `USER_REPLY`, `AGENT_DM`, `TEAM_MEMORY`,
  `TOOL_PAYLOAD`, or `RUN_LOG`
- `REFUSE` unsafe requests
- `VERIFY` a proposed message before sending it
- `COMMIT` the final task result

The server keeps a surface-by-surface audit log. At the end of an episode, a
deterministic grader computes task utility, privacy score, refusal calibration,
state integrity, and efficiency. There is no LLM judge in the reward path.

Built-in scenarios include must-refuse, must-comply, long distracting prompts,
triad coordination, and round-robin multi-agent turns.

## Why this is interesting

Membrane does not ask whether an answer sounds safe. It asks whether the
agent kept the secret out of every channel while still doing the useful
work. That is harder to game than checking only the final response, and it
creates a clear RL signal for hidden-channel privacy behaviour.

## Training results

We trained `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` with a LoRA adapter using
TRL `GRPOTrainer`. The reward function replays each generated JSONL trajectory
inside Membrane. Invalid JSON, invalid actions, unsafe leaks, or incomplete
episodes score `0.0`; valid privacy-preserving task completion receives high
terminal reward.

### How the model learns Membrane (or fails to)

![Reward curves for every Membrane GRPO run we logged. Cold-start runs (grey
band, three runs) never escape zero. The Colab hero (black) is the cold-start
that did converge, hitting 0.974 at step 860. Four warm-start runs that loaded
the hero adapter as initial weights (blue and red) climb above the hero on the
single training task and lift onto the 7-task curriculum, with the conservative
single-task warm start peaking at 0.988.](docs/plots/grpo_warmstart_ablation.svg)

The figure above is every Membrane GRPO run we have metrics for, on one set of
axes: 3 cold-start Hugging Face Job runs (the grey band along the bottom), the
Colab hero (black, 1000 steps), and the 4 warm-start Hugging Face Job runs that
form a 2 × 2 ablation over learning rate × task mix. **What to look for:** the
cold-start band stuck at zero, the hero climbing in one big jump around step
200, and the warm-starts (blue, red) starting high and going higher.

| run | category | first reward | final reward | best reward | best step |
|---|---|---|---|---|---|
| Colab hero (cold start, lr=5e-6, 1000 steps) | hero | 0.000 | 0.935 | **0.974** | 860 |
| cold-deep / seed 3408 | cold-start (HF Jobs) | 0.006 | 0.000 | 0.014 | 840 |
| cold-deep / seed 3409 | cold-start (HF Jobs) | 0.006 | 0.000 | 0.014 | 840 |
| cold-multi / seed 3410 | cold-start (HF Jobs) | 0.008 | 0.008 | 0.040 | 180 |
| **warm-deep / lr=2e-6 (conservative)** | warm-start (HF Jobs) | 0.880 | **0.971** | **0.988** | 580 |
| warm-deep / lr=5e-6 (aggressive) | warm-start (HF Jobs) | 0.849 | 0.959 | 0.988 | 240 |
| warm-multi / lr=3e-6 (conservative, 7 tasks) | warm-start (HF Jobs) | 0.495 | **0.793** | 0.793 | 2000 |
| warm-multi / lr=5e-6 (aggressive, 7 tasks) | warm-start (HF Jobs) | 0.431 | 0.785 | 0.854 | 800 |

What this shows:

1. **Cold-start GRPO collapses on Membrane.** The 3 cold runs at the bottom
   stay below 0.02 mean reward for their entire budgets (900–1400 steps). The
   reward function is sparse, so a freshly initialised policy almost never
   produces a non-zero advantage signal.
2. **The Colab hero converges in 1000 steps.** Same script, same recipe, just a
   T4 RNG stream and pinned Unsloth/TRL versions. Final 0.935, peak 0.974.
3. **Warm-starting from the hero adapter on HF Jobs beats the hero on the
   original task.** `warm-deep / lr=2e-6` lifts from 0.880 → **0.971 final /
   0.988 peak**, surpassing both the hero's 0.935 final and 0.974 peak.
4. **Aggressive learning rate saturates GRPO.** `warm-deep / lr=5e-6` peaks at
   0.988 by step 240, then `frac_reward_zero_std` rises above 0.7 (≥70 % of
   prompt groups have identical rewards on all 4 completions), the per-group
   advantage goes to zero, and `grad_norm` collapses to exactly 0.0. The
   reward then drifts down to 0.959. Conservative lr=2e-6 avoids the
   saturation and keeps improving.
5. **The single-task warm-start transfers to the 7-scenario curriculum.**
   `warm-multi / lr=3e-6` starts at 0.495 (the model has never seen 6 of those
   7 tasks during the original Colab training) and climbs to **0.793** by step
   2000 without collapsing.

### Did the model actually learn? Trained vs base, same weights

We re-evaluated the *same* Qwen2.5 1.5 B with the LoRA delta toggled on/off,
generating 4 rollouts per task with the exact training prompt and scoring each
one through the Membrane environment. The trained adapter is our best warm-start
checkpoint, `continue_deep_seed_5821_steps_1500`.

![Mean Membrane reward by task. Base model (LoRA off, striped grey bars at zero)
scores 0.00 on every task. Trained model (LoRA on, teal bars) scores 0.99 on the
training task, 0.43 on the must-comply transfer task, 1.00 on the long-horizon
variant with 41 distractors, and 0.64 on the 3-actor triad. Same Qwen2.5 1.5 B
weights underneath.](docs/eval/base_vs_trained/reward_by_task.svg)

| task | base reward | trained reward | delta |
|---|---|---|---|
| `dyad_must_refuse_v1` (training task) | 0.000 | **0.988** | **+0.988** |
| `dyad_must_refuse_long_v1` (transfer, 41 distractors) | 0.000 | **0.996** | **+0.996** |
| `triad_must_refuse_v1` (transfer, 3 actors) | 0.000 | 0.642 | +0.642 |
| `dyad_must_comply_v1` (transfer, must NOT refuse) | 0.000 | 0.435 | +0.435 |
| **overall** | **0.000** | **0.765** | **+0.765** |

Base mean reward is `0.000` everywhere because the base model produces **0%
valid Membrane JSONL** - it cannot follow the verb-style action grammar at all.
The trained policy produces **100% valid JSONL and 100% `COMMIT` rate** on
every task, with the highest reward on the long-horizon variant (41 scattered
distractor instructions). The transfer drop on `dyad_must_comply_v1` is itself
informative: the model learned to refuse, so it slightly over-refuses on
benign requests it was never trained on. Re-running this with the
`continue_multifull_seed_5822` adapter (which trained on all 7 tasks) is the
obvious next step.

Source data and full per-sample completions live in
[`docs/eval/base_vs_trained/`](docs/eval/base_vs_trained/) (32 rollouts,
bar charts, CSV, JSON). The eval is also runnable from section 8 of the
Colab notebook against any adapter on the Hub.

The aggregate plot, headline CSV, and per-run metrics live under
[`docs/plots/`](docs/plots/) and [`docs/hf_runs/`](docs/hf_runs/). Every
trained adapter is on the Hub at
<https://huggingface.co/Tejasghatule/membrane-qwen25-1p5b-grpo-lora>.

## Training notebook

**Colab (the run we used for the 1000-step hero):**  
<https://colab.research.google.com/drive/1rEFKYNGbtoNZmClFDh8Q0aoeTdy7Xsrf?usp=sharing>

**Same source in the repo:**  
[`notebooks/membrane_train_colab.ipynb`](notebooks/membrane_train_colab.ipynb)  
runs end-to-end on a single Colab T4 (no Pro account required) and is the
fastest way to reproduce a Membrane policy from scratch. It has two modes:

- **Quick check (`LONG_RUN = False`, 30 steps):** verifies the pipeline on a
  fresh GPU runtime in a couple of minutes.
- **Long run (`LONG_RUN = True`, `TRAIN_ON_DRIVE = True`):** saves
  checkpoints and TensorBoard logs to Google Drive so Colab disconnects do
  not erase the run.

Section 8 of the notebook contains an in-distribution base-vs-trained eval
cell - same model weights, LoRA toggled on/off, scored by Membrane's grader.

## Run the environment locally

From the repo root:

```bash
cd membrane
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 7860
```

## Repository layout

Public clone: <https://github.com/CodeMaverick2/membrane>

- `server/` - FastAPI app and `MembraneEnvironment` (the live OpenEnv server).
- `tasks/scenarios.py` - the seven shipped scenarios (must-refuse,
  must-comply, long-distractor, triad, round-robin probe, …).
- `models.py` - JSONL action grammar and per-channel scoring.
- `notebooks/membrane_train_colab.ipynb` - training notebook.
- `scripts/hf_jobs/` - Hugging Face Jobs pipeline (training, eval,
  aggregation). See its [`README`](scripts/hf_jobs/README.md).
- `docs/plots/` - aggregate plots and summaries.
- `docs/eval/base_vs_trained/` - base-vs-trained evaluation artefacts.
- `docs/hf_runs/` - per-run metrics (CSV + JSON) for every training run.
