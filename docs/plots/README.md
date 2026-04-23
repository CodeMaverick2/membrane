# Plots (P4)

Regenerate after changing policies or scenario mix:

```bash
cd membrane
python3 train/compare_policies.py --task dyad_must_refuse_v1 --episodes 48
python3 train/render_plots.py
```

Run against a deployed Space instead of local in-process env:

```bash
python3 train/compare_policies.py --task dyad_must_refuse_v1 --episodes 48 --url https://YOUR_SPACE.hf.space
python3 train/render_plots.py
```

- `episode_returns.csv` — per-episode `Total` reward from the local `MembraneEnvironment` (no LLM).
- `baseline_vs_heuristic.svg` — 5-episode rolling mean of `Total` for judge-facing READMEs (no matplotlib required).

Optional PNG (requires `pip install matplotlib`):

```bash
python3 train/render_png_matplotlib.py
```
