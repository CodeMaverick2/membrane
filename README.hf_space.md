---
title: MembraneEnv
emoji: shield
colorFrom: gray
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - privacy
  - multi-agent
---

# MembraneEnv

OpenEnv **FastAPI** server: multi-step episodes with **surface-audited** `SEND`, **refusal gold**, deterministic **terminal metrics** (`Total`, per-surface leak rates, `BlindSpotIndex`, `LeakSeverity`, `PrivacyHallucinationRate`, `RefusalCalibration`), and `VERIFY` dry-runs.

## Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Tests

```bash
python3 -m pytest tests/ -v
./scripts/run_tests.sh
```

## Training artifacts (no GPU required)

```bash
./scripts/refresh_plots.sh
```

Space rollout mode:

```bash
python3 train/compare_policies.py --task dyad_must_refuse_v1 --episodes 48 --url https://YOUR_SPACE.hf.space
python3 train/render_plots.py
```

See [`docs/plots/README.md`](docs/plots/README.md).

## Docs (repo)

- [Specification & phases](../membrance-docs/SPEC.md)
- [Implementation phases](../membrance-docs/07-implementation-phases.md)

## Docker / Space

```bash
docker build -t membrane .
docker run -p 7860:7860 membrane
```

`openenv.yaml` lists shipped `tasks` ids; keep them aligned with `tasks/scenarios.py`.
