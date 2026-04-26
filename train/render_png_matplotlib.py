#!/usr/bin/env python3
"""Optional: write baseline_vs_heuristic.png using matplotlib (pip install matplotlib)."""
from __future__ import annotations

import csv
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("Install matplotlib: pip install matplotlib") from e

    csv_path = _ROOT / "docs" / "plots" / "episode_returns.csv"
    by_pol: dict[str, list[float]] = {"baseline": [], "heuristic": []}
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_pol[row["policy"]].append(float(row["return"]))

    plt.figure(figsize=(8, 4.5))
    labels = {
        "baseline": "Weak scripted baseline (5-ep mean)",
        "heuristic": "Hand-tuned scripted policy (5-ep mean)",
    }
    for label, color in (("baseline", "#c0392b"), ("heuristic", "#27ae60")):
        vals = by_pol[label]
        xs = list(range(len(vals)))
        plt.plot(xs, vals, alpha=0.35, color=color, linewidth=1)
        window = min(5, len(vals))
        smooth = [
            sum(vals[max(0, i - window + 1) : i + 1]) / (i - max(0, i - window + 1) + 1)
            for i in range(len(vals))
        ]
        plt.plot(xs, smooth, color=color, label=labels[label])

    plt.xlabel("Episode index")
    plt.ylabel("Membrane episode score (0–1)")
    plt.title("Scripted policies on the refuse-leak scenario (not the neural model)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.25)
    out = _ROOT / "docs" / "plots" / "baseline_vs_heuristic.png"
    out_svg = out.with_suffix(".svg")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.savefig(out_svg, format="svg")
    plt.close()
    print(f"Wrote {out} and {out_svg}")


if __name__ == "__main__":
    main()
