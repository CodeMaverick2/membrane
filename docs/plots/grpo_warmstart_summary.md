# Membrane GRPO - warm-start ablation

Aggregate of every Membrane GRPO run that has produced metrics:

- 1 Colab run (the hero) that converged from a cold start.
- 3 cold-start Hugging Face Job runs that did not converge.
- 4 warm-start Hugging Face Job runs that loaded the hero adapter as initial
  weights and continued training. The four form a 2 × 2 ablation across
  learning rate (conservative vs aggressive) and task mix (single task vs
  full 7-scenario curriculum).

| run | category | first reward | final reward | best reward | best step | steps logged |
|---|---|---|---|---|---|---|
| Colab hero (cold start, lr=5e-6, 1000 steps) | hero | 0.000 | 0.935 | **0.974** | 860 | 50 |
| cold-deep / seed 3408 (collapsed) | cold-start | 0.006 | 0.000 | **0.014** | 840 | 45 |
| cold-deep / seed 3409 (collapsed) | cold-start | 0.006 | 0.000 | **0.014** | 840 | 45 |
| cold-multi / seed 3410 (collapsed) | cold-start | 0.008 | 0.008 | **0.040** | 180 | 70 |
| warm-deep / lr=2e-6 (conservative) | warm-start | 0.880 | 0.971 | **0.988** | 580 | 75 |
| warm-deep / lr=5e-6 (aggressive, saturates) | warm-start | 0.849 | 0.959 | **0.988** | 240 | 75 |
| warm-multi / lr=3e-6 (conservative, 7 tasks) | warm-start | 0.495 | 0.793 | **0.793** | 2000 | 100 |
| warm-multi / lr=5e-6 (aggressive, 7 tasks) | warm-start | 0.431 | 0.785 | **0.854** | 800 | 100 |

## Findings

1. **Cold-start GRPO does not learn Membrane in the budgets we tested.** Three
   independent HF Job runs (single-task seeds 3408 / 3409 at 900 steps,
   multi-task seed 3410 at 1400 steps) all stay below 0.02 mean reward.
   Membrane's reward is sparse on purpose - any malformed JSONL action zeroes
   the episode - so a freshly-initialised policy almost never produces a
   non-zero advantage signal long enough for GRPO to bootstrap.

2. **The Colab T4 run converged with the same recipe that collapsed on the
   A10G HF Job runs.** Same script, same hyperparameters, same seed family.
   The only difference is the RNG stream from a different GPU and the
   Unsloth/TRL versions that the version pin now reproduces. This is why
   warm-starting was worth doing: a known-good policy whose weights could be
   redeployed as initial conditions on the HF compute backend.

3. **Warm-starting from the hero adapter beats the hero on the same task.**
   `warm-deep / lr=2e-6` lifts mean reward from 0.880 (the hero adapter's
   starting score) to **0.971 final / 0.988 peak**, surpassing both the
   hero's 0.935 final and 0.974 peak. The conservative learning rate is the
   key - see finding 4.

4. **Aggressive learning rate saturates GRPO on a single task.**
   `warm-deep / lr=5e-6` (the *hero's* learning rate) climbs from 0.849 to a
   peak of **0.988 by step 240**, then drifts back down to 0.959 by
   step 1500. This is not a model failure: `frac_reward_zero_std` rises
   from 0.2 to ≥ 0.7, meaning ≥ 70 % of GRPO prompt groups produce identical
   rewards across all 4 completions. With zero per-group advantage there is
   no gradient signal, and `grad_norm` falls to exactly 0.0. Conservative
   lr=2e-6 keeps per-group variance alive longer and continues improving.

5. **A single-task warm-start transfers to the full 7-scenario curriculum.**
   `warm-multi / lr=3e-6` (must-refuse, must-comply, long, triad,
   round-robin, and two more held-out scenarios) starts at 0.495 - the model
   has never seen 6 of those 7 tasks during the original Colab training -
   and climbs to **0.793** by step 2000 without collapsing. The aggressive
   variant peaks higher (0.854 at step 800) but slides to 0.785 as it
   over-fits the task it was warm-started on.

## Source data

- Application source: <https://github.com/CodeMaverick2/membrane>
- Per-run metrics: `docs/hf_runs/<run>/training_metrics.csv` and
  `docs/hf_runs/<run>/run_summary.json`.
- Aggregate plot: `grpo_warmstart_ablation.png` / `.svg`.
- Headline CSV: `grpo_warmstart_summary.csv`.
- All adapters:
  <https://huggingface.co/Tejasghatule/membrane-qwen25-1p5b-grpo-lora>.
