#!/usr/bin/env python3
"""Submit Membrane credit-spend jobs to Hugging Face Jobs."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import get_token, run_uv_job


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "scripts" / "hf_jobs" / "train_grpo_job.py"
EVAL_SCRIPT = ROOT / "scripts" / "hf_jobs" / "eval_base_vs_trained.py"
DEFAULT_NAMESPACE = "Tejasghatule"


def submit(script: Path, args: list[str], flavor: str, timeout: str, namespace: str) -> None:
    token = get_token()
    if not token:
        raise RuntimeError("Log in first with `from huggingface_hub import login; login()`.")
    print("Submitting with cached Hugging Face token")
    job = run_uv_job(
        str(script),
        script_args=args,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": token},
        namespace=namespace,
        token=token,
    )
    print("Job submitted")
    print("ID:", job.id)
    print("URL:", getattr(job, "url", f"https://huggingface.co/jobs/{job.id}"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Cheap 30-step run to verify the HF Job pipeline.")
    smoke.add_argument("--seed", type=int, default=4242)
    smoke.add_argument("--max-steps", type=int, default=30)
    smoke.add_argument("--flavor", default="l4x1")
    smoke.add_argument("--timeout", default="1h")
    smoke.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    repeat = sub.add_parser("repeat")
    repeat.add_argument("--seed", type=int, required=True)
    repeat.add_argument("--max-steps", type=int, default=900)
    repeat.add_argument("--flavor", default="l4x1")
    repeat.add_argument("--timeout", default="4h")
    repeat.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    multi = sub.add_parser("multi")
    multi.add_argument("--seed", type=int, default=3410)
    multi.add_argument("--max-steps", type=int, default=1400)
    multi.add_argument("--flavor", default="l4x1")
    multi.add_argument("--timeout", default="6h")
    multi.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    # The "deep" and "multi-full" recipes match the hero hyperparameters that
    # produced the 0.97 reward in Colab (lr=5e-6, num_gen=4, grad_accum=4) and
    # extend max_steps to 2000 for a longer/cleaner curve.
    deep = sub.add_parser("deep", help="Hero-aligned single-task replication run, longer schedule.")
    deep.add_argument("--seed", type=int, default=4711)
    deep.add_argument("--max-steps", type=int, default=2000)
    deep.add_argument("--learning-rate", type=float, default=5e-6)
    deep.add_argument("--num-generations", type=int, default=4)
    deep.add_argument("--grad-accum", type=int, default=4)
    deep.add_argument("--warmup-ratio", type=float, default=0.05)
    deep.add_argument("--max-completion-length", type=int, default=512)
    deep.add_argument("--flavor", default="l4x1")
    deep.add_argument("--timeout", default="8h")
    deep.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    multi_full = sub.add_parser("multi-full", help="Hero-aligned curriculum across all 7 Membrane scenarios.")
    multi_full.add_argument("--seed", type=int, default=4712)
    multi_full.add_argument("--max-steps", type=int, default=2000)
    multi_full.add_argument("--learning-rate", type=float, default=5e-6)
    multi_full.add_argument("--num-generations", type=int, default=4)
    multi_full.add_argument("--grad-accum", type=int, default=4)
    multi_full.add_argument("--warmup-ratio", type=float, default=0.05)
    multi_full.add_argument("--max-completion-length", type=int, default=512)
    multi_full.add_argument("--flavor", default="l4x1")
    multi_full.add_argument("--timeout", default="8h")
    multi_full.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    # The "continue-*" recipes warm-start from the published Colab hero adapter
    # (which already gets ~0.95 reward on dyad_must_refuse_v1) and then keep
    # training. This guarantees informative curves and exercises the remaining
    # HF compute budget on the *most useful* experiments: extended single-task
    # stability and multi-scenario transfer.
    HERO_ADAPTER_REPO = "Tejasghatule/membrane-qwen25-1p5b-grpo-lora"
    HERO_ADAPTER_SUB = "membrane_grpo_existing_checkpoint_1000/final_adapter"

    cont_deep = sub.add_parser(
        "continue-deep",
        help="Warm-start from hero adapter and keep training on dyad_must_refuse_v1.",
    )
    cont_deep.add_argument("--seed", type=int, default=5821)
    cont_deep.add_argument("--max-steps", type=int, default=1500)
    cont_deep.add_argument("--learning-rate", type=float, default=2e-6)
    cont_deep.add_argument("--num-generations", type=int, default=4)
    cont_deep.add_argument("--grad-accum", type=int, default=4)
    cont_deep.add_argument("--warmup-ratio", type=float, default=0.02)
    cont_deep.add_argument("--max-completion-length", type=int, default=512)
    cont_deep.add_argument("--init-adapter-repo", default=HERO_ADAPTER_REPO)
    cont_deep.add_argument("--init-adapter-subfolder", default=HERO_ADAPTER_SUB)
    cont_deep.add_argument("--flavor", default="l4x1")
    cont_deep.add_argument("--timeout", default="6h")
    cont_deep.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    cont_multi = sub.add_parser(
        "continue-multi-full",
        help="Warm-start from hero adapter and train on all 7 Membrane scenarios.",
    )
    cont_multi.add_argument("--seed", type=int, default=5822)
    cont_multi.add_argument("--max-steps", type=int, default=2000)
    cont_multi.add_argument("--learning-rate", type=float, default=3e-6)
    cont_multi.add_argument("--num-generations", type=int, default=4)
    cont_multi.add_argument("--grad-accum", type=int, default=4)
    cont_multi.add_argument("--warmup-ratio", type=float, default=0.03)
    cont_multi.add_argument("--max-completion-length", type=int, default=512)
    cont_multi.add_argument("--init-adapter-repo", default=HERO_ADAPTER_REPO)
    cont_multi.add_argument("--init-adapter-subfolder", default=HERO_ADAPTER_SUB)
    cont_multi.add_argument("--flavor", default="l4x1")
    cont_multi.add_argument("--timeout", default="8h")
    cont_multi.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    eval_cmd = sub.add_parser("eval")
    eval_cmd.add_argument("--flavor", default="t4-small")
    eval_cmd.add_argument("--timeout", default="2h")
    eval_cmd.add_argument("--checkpoint-run", default="membrane_grpo_existing_checkpoint_1000")
    eval_cmd.add_argument("--namespace", default=DEFAULT_NAMESPACE)

    args = parser.parse_args()
    if args.command == "smoke":
        submit(
            TRAIN_SCRIPT,
            [
                "--mode",
                "repeat",
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
                "--run-name",
                f"smoke_seed_{args.seed}_steps_{args.max_steps}",
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "repeat":
        run_name = f"repeat_seed_{args.seed}"
        submit(
            TRAIN_SCRIPT,
            [
                "--mode",
                "repeat",
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
                "--run-name",
                run_name,
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "multi":
        submit(
            TRAIN_SCRIPT,
            [
                "--mode",
                "multi",
                "--seed",
                str(args.seed),
                "--max-steps",
                str(args.max_steps),
                "--run-name",
                f"multi_seed_{args.seed}",
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "deep":
        run_name = f"deep_seed_{args.seed}_steps_{args.max_steps}"
        submit(
            TRAIN_SCRIPT,
            [
                "--mode", "repeat",
                "--seed", str(args.seed),
                "--max-steps", str(args.max_steps),
                "--learning-rate", str(args.learning_rate),
                "--num-generations", str(args.num_generations),
                "--grad-accum", str(args.grad_accum),
                "--warmup-ratio", str(args.warmup_ratio),
                "--max-completion-length", str(args.max_completion_length),
                "--run-name", run_name,
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "multi-full":
        run_name = f"multifull_seed_{args.seed}_steps_{args.max_steps}"
        submit(
            TRAIN_SCRIPT,
            [
                "--mode", "multi_full",
                "--seed", str(args.seed),
                "--max-steps", str(args.max_steps),
                "--learning-rate", str(args.learning_rate),
                "--num-generations", str(args.num_generations),
                "--grad-accum", str(args.grad_accum),
                "--warmup-ratio", str(args.warmup_ratio),
                "--max-completion-length", str(args.max_completion_length),
                "--run-name", run_name,
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "continue-deep":
        run_name = f"continue_deep_seed_{args.seed}_steps_{args.max_steps}"
        submit(
            TRAIN_SCRIPT,
            [
                "--mode", "repeat",
                "--seed", str(args.seed),
                "--max-steps", str(args.max_steps),
                "--learning-rate", str(args.learning_rate),
                "--num-generations", str(args.num_generations),
                "--grad-accum", str(args.grad_accum),
                "--warmup-ratio", str(args.warmup_ratio),
                "--max-completion-length", str(args.max_completion_length),
                "--init-adapter-repo", args.init_adapter_repo,
                "--init-adapter-subfolder", args.init_adapter_subfolder,
                "--run-name", run_name,
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "continue-multi-full":
        run_name = f"continue_multifull_seed_{args.seed}_steps_{args.max_steps}"
        submit(
            TRAIN_SCRIPT,
            [
                "--mode", "multi_full",
                "--seed", str(args.seed),
                "--max-steps", str(args.max_steps),
                "--learning-rate", str(args.learning_rate),
                "--num-generations", str(args.num_generations),
                "--grad-accum", str(args.grad_accum),
                "--warmup-ratio", str(args.warmup_ratio),
                "--max-completion-length", str(args.max_completion_length),
                "--init-adapter-repo", args.init_adapter_repo,
                "--init-adapter-subfolder", args.init_adapter_subfolder,
                "--run-name", run_name,
            ],
            args.flavor,
            args.timeout,
            args.namespace,
        )
    elif args.command == "eval":
        submit(
            EVAL_SCRIPT,
            ["--checkpoint-run", args.checkpoint_run],
            args.flavor,
            args.timeout,
            args.namespace,
        )


if __name__ == "__main__":
    main()
