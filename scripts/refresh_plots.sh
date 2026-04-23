#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 train/compare_policies.py --task dyad_must_refuse_v1 --episodes 48 \
  --seed 0 \
  --summary-json docs/plots/rollout_summary.json
python3 train/render_plots.py
if python3 -c "import matplotlib" 2>/dev/null; then
  python3 train/render_png_matplotlib.py
fi
