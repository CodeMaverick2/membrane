#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Heuristic-vs-LLM rollout comparison (the small baseline plot).
python3 train/compare_policies.py --task dyad_must_refuse_v1 --episodes 48 \
  --seed 0 \
  --summary-json docs/plots/rollout_summary.json
python3 train/render_plots.py
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 train/render_png_matplotlib.py
fi

# 2. Aggregate GRPO ablation plot + summary (cold + hero + warm-start runs).
python3 scripts/analysis/build_warmstart_ablation_plot.py

# 3. Trained-vs-base bar charts re-rendered from the saved summary JSON.
python3 scripts/analysis/replot_base_vs_trained.py
