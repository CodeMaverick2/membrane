# Membrane

**Thesis:** train and score assistants on **channel-wise behavior** (who sees what) and **refusal under pressure**, not just the last line of text—because real failures hide in DMs and tool logs.

Problem statement for judges: **[`../membrance-docs/HACKATHON_PROBLEM_STATEMENT.md`](../membrance-docs/HACKATHON_PROBLEM_STATEMENT.md)** · demo script: **`python3 scripts/judge_walkthrough.py`** (from `membrane/`).

## The problem

Assistants help with calendar and email-style work, but they often leak sensitive details on the wrong channel: a DM to a colleague, a team scratchpad, or a tool log, while the message to the user looks fine. If you only read the user-facing reply, you miss most of that. We wanted a small environment where **where** each message goes matters, and where you can **score** privacy, refusals, and task completion in one place so you can train or evaluate a model on it.

## What this project is

Membrane is an **OpenEnv** environment: the model (or a script) sends structured actions—look up a fact, send text on a named surface (user reply, agent DM, team memory, etc.), refuse, double-check text, or commit a final confirmation. The server keeps an **audit log** and, at the end of an episode, runs a **fixed grader** (no LLM judge) to produce numbers like task success, leak rates, and refusal behavior. There are several built-in scenarios (dyad schedule, must-refuse, must-comply, a long distracting one, a three-agent one with inbox lines, round-robin turns).

Code lives here. Longer writeups and the full spec are in **`../membrance-docs/`** next to this folder.

## What the hackathon asked for (short)

They want: an **OpenEnv** environment people can run; a **Space on Hugging Face**; a **training path** (TRL or Unsloth, often Colab); **proof something was trained** (curves or before/after); a **short blog or under-2-minute video**; and a **README** that explains the problem, the environment, and results, with **links** to the Space and everything else—not big video files in the repo.

## What we actually have so far

- OpenEnv-style app and **`openenv.yaml`** in this directory; tests under `tests/`.
- Deterministic rewards / metrics (see **`../membrance-docs/06-reward-and-rubrics.md`** if you care about the math).
- Local **baseline vs heuristic** rollouts and a chart + CSV under **`docs/plots/`** (scripted policies, not an LLM yet—that is the gap to fill before submit).
- **Unsloth + TRL path:** Colab **`notebooks/membrane_train_colab.ipynb`**, GRPO reward bridge **`train/unsloth_reward.py`**, docs **`../membrance-docs/08-training-trl-unsloth.md`**. Unsloth **Studio** (browser) vs training script: **`../membrance-docs/UNSLOTH_STUDIO_VS_TRAINING.md`**. Local reward smoke: **`python3 scripts/smoke_membrane_reward_training.py`**.
- Space-oriented copy for Docker/HF: **`README.hf_space.md`**.

You still need to add your real **Space URL**, **Colab**, **blog or video link**, and ideally **one training plot from an actual LLM run** when you have them.

## Run it

From the repo root:

```bash
cd membrane
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests/ -q
.venv/bin/uvicorn server.app:app --host 127.0.0.1 --port 7860
```

Regenerate the local comparison plot: **`./scripts/refresh_plots.sh`**

## Links (fill in before you submit)

| What | Link |
|------|------|
| Hugging Face Space | |
| Colab (training) | |
| Blog (HF) or video (YouTube), under 2 min | |

Draft notes you do not want in git: **`../private/`** (only **`../private/README.md`** is meant to be tracked).
