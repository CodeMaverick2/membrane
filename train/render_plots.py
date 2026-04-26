#!/usr/bin/env python3
"""Read episode_returns.csv; write baseline_vs_heuristic.svg (stdlib only, no matplotlib)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from xml.sax.saxutils import escape


def rolling_mean(values: list[float], window: int) -> list[float]:
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def build_svg(
    baseline: list[float],
    heuristic: list[float],
    title: str,
    width: int = 720,
    height: int = 420,
) -> str:
    pad_l, pad_r, pad_t, pad_b = 56, 24, 28, 48
    w = width - pad_l - pad_r
    h = height - pad_t - pad_b
    allv = baseline + heuristic
    y_min = min(allv) * 0.95
    y_max = max(allv) * 1.05
    if y_max <= y_min:
        y_max = y_min + 1e-6

    def xpix(i: int, n: int) -> float:
        if n <= 1:
            return pad_l
        return pad_l + w * (i / (n - 1))

    def ypix(v: float) -> float:
        t = (v - y_min) / (y_max - y_min)
        return pad_t + h * (1.0 - t)

    def polyline(vals: list[float], stroke: str) -> str:
        n = len(vals)
        pts = " ".join(f"{xpix(i, n):.1f},{ypix(vals[i]):.1f}" for i in range(n))
        return f'<polyline fill="none" stroke="{stroke}" stroke-width="2" points="{pts}"/>'

    wb = rolling_mean(baseline, 5)
    wh = rolling_mean(heuristic, 5)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0f1419"/>',
        f'<text x="{width // 2}" y="20" text-anchor="middle" fill="#e6edf3" font-size="15" '
        f'font-family="system-ui,sans-serif">{escape(title)}</text>',
        f'<text x="{pad_l}" y="{height - 12}" fill="#8b949e" font-size="12" '
        'font-family="system-ui,sans-serif">Episode index</text>',
        f'<text transform="translate(16,{pad_t + h / 2}) rotate(-90)" text-anchor="middle" '
        'fill="#8b949e" font-size="12" font-family="system-ui,sans-serif">Mean episode return (Total)</text>',
        f'<line x1="{pad_l}" y1="{pad_t + h}" x2="{pad_l + w}" y2="{pad_t + h}" stroke="#30363d"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + h}" stroke="#30363d"/>',
        polyline(wb, "#f85149"),
        polyline(wh, "#3fb950"),
        '<text x="' + str(pad_l + w - 4) + '" y="' + str(pad_t + 14) + '" text-anchor="end" '
        'fill="#f85149" font-size="11" font-family="system-ui,sans-serif">baseline (5-ep mean)</text>',
        '<text x="' + str(pad_l + w - 4) + '" y="' + str(pad_t + 30) + '" text-anchor="end" '
        'fill="#3fb950" font-size="11" font-family="system-ui,sans-serif">heuristic (5-ep mean)</text>',
        "</svg>",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "plots" / "episode_returns.csv",
    )
    ap.add_argument(
        "--out-svg",
        type=Path,
        dest="out_svg",
        default=Path(__file__).resolve().parent.parent / "docs" / "plots" / "baseline_vs_heuristic.svg",
    )
    args = ap.parse_args()
    by_pol: dict[str, list[float]] = {"baseline": [], "heuristic": []}
    with args.csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_pol[row["policy"]].append(float(row["return"]))
    svg = build_svg(by_pol["baseline"], by_pol["heuristic"], "Membrane - baseline vs heuristic (local rollouts)")
    args.out_svg.parent.mkdir(parents=True, exist_ok=True)
    args.out_svg.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.out_svg}")


if __name__ == "__main__":
    main()
