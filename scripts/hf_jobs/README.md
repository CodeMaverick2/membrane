# Hugging Face Jobs pipeline

Application source: <https://github.com/CodeMaverick2/membrane>

Scripts that train, evaluate, and aggregate Membrane GRPO runs on Hugging Face
Jobs. Every run uploads its artifacts to the Hub before exit, so the local
`docs/` tree only needs the metrics CSVs and plots - not the multi-GB
checkpoints.

## Hub repos these scripts write to

- LoRA adapters: <https://huggingface.co/Tejasghatule/membrane-qwen25-1p5b-grpo-lora>
- Training metrics, plots, eval artifacts:
  <https://huggingface.co/datasets/Tejasghatule/membrane-grpo-results>

## Authenticate

```bash
cd membrane
.venv/bin/python - <<'PY'
from huggingface_hub import login
login()
PY
```

The model and dataset repos linked above already exist; if you fork the
project under a different namespace, create equivalent repos on the Hub
first and update the namespace constant in `submit_hf_jobs.py` and
`train_grpo_job.py`.

## Files in this directory

| script | what it does |
|---|---|
| `submit_hf_jobs.py` | CLI to submit training and eval runs as Hugging Face Jobs |
| `train_grpo_job.py` | The training entry point that runs inside each Job |
| `eval_base_vs_trained.py` | Eval entry point: same model, LoRA toggled on/off |
| `aggregate_hf_results.py` | Renders combined plots from multiple runs' CSVs |

## Submitting runs

The recommended path is **warm-start, not cold-start** - three independent
cold-start runs collapsed below 0.02 mean reward (see
`docs/plots/grpo_warmstart_summary.md`). Warm-start subcommands load the
Colab hero adapter as initial weights:

```bash
# Warm-start, single task, conservative lr (the run that beat the hero)
.venv/bin/python scripts/hf_jobs/submit_hf_jobs.py continue-deep \
  --seed 5821 --max-steps 1500 --lr 2e-6

# Warm-start, full 7-scenario curriculum
.venv/bin/python scripts/hf_jobs/submit_hf_jobs.py continue-multi-full \
  --seed 5822 --max-steps 2000 --lr 3e-6

# Base-vs-trained eval against any uploaded checkpoint run
.venv/bin/python scripts/hf_jobs/submit_hf_jobs.py eval \
  --checkpoint-run continue_deep_seed_5821_steps_1500
```

For provenance, the original cold-start subcommands (`smoke`, `repeat`,
`multi`, `deep`, `multi-full`) are still in `submit_hf_jobs.py`.

## Where each Job writes its artifacts

- Training jobs:
  - `Tejasghatule/membrane-qwen25-1p5b-grpo-lora` → `<run_name>/final_adapter/`
  - `Tejasghatule/membrane-grpo-results` → `runs/<run_name>/{training_metrics.csv, run_summary.json, plots/*}`
- Eval jobs:
  - `Tejasghatule/membrane-grpo-results` → `eval/<checkpoint_run>/{base_vs_trained.csv, base_vs_trained_examples.jsonl, base_vs_trained_summary.{json,md}, plots/*}`

## Aggregating runs locally

After downloading metrics CSVs from the dataset repo (or pointing at
`docs/plots/`), render the cross-run comparison:

```bash
.venv/bin/python scripts/hf_jobs/aggregate_hf_results.py \
  --input-root <folder containing training_metrics.csv files> \
  --out-dir docs/plots
```

This writes `combined_grpo_reward_curves.{png,svg}`,
`combined_grpo_loss.png`, `combined_grpo_kl.png`, and
`combined_grpo_summary.{json,md}`.

## Notes

- Jobs are ephemeral. Every script uploads to the Hub before exit; nothing
  persists on the Job runtime between runs.
- `train_grpo_job.py` pins `trl==0.22.2`, `vllm==0.10.2`, and ships
  `mergekit` to satisfy TRL's callbacks import path. It also stubs the
  optional `vllm_ascend` module on CUDA flavors.
- `submit_hf_jobs.py` pins `--namespace Tejasghatule` to avoid `whoami` rate
  limits.
