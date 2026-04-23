"""
FastAPI entry for Membrane OpenEnv server.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Repo-style import when running from membrane/ (package root)
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from server.environment import MembraneEnvironment

# OpenEnv's HTTP handlers call the env factory for every /reset and /step and then
# close the instance. Membrane episodes are stateful, so we return one process-wide
# instance (close() is a no-op on the default Environment base).
_HTTP_MEMBRANE_SINGLETON: MembraneEnvironment | None = None


def membrane_http_singleton() -> MembraneEnvironment:
    global _HTTP_MEMBRANE_SINGLETON
    if _HTTP_MEMBRANE_SINGLETON is None:
        _HTTP_MEMBRANE_SINGLETON = MembraneEnvironment()
    return _HTTP_MEMBRANE_SINGLETON


def _make_app():
    try:
        from openenv.core.env_server import create_fastapi_app

        from models import MembraneAction, MembraneObservation

        return create_fastapi_app(
            membrane_http_singleton,
            MembraneAction,
            MembraneObservation,
        )
    except ImportError:
        from fastapi import FastAPI

        app = FastAPI(title="MembraneEnv", version="0.3.0")
        env = MembraneEnvironment()

        @app.get("/health")
        async def health():
            return {"status": "healthy", "env": "membrane", "mode": "stub"}

        @app.post("/reset")
        async def reset(body: Optional[dict] = None):
            obs = env.reset(**(body or {}))
            return obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

        @app.post("/step")
        async def step(body: dict):
            from models import MembraneAction

            action = MembraneAction(**body.get("action", body))
            obs = env.step(action)
            return obs.model_dump() if hasattr(obs, "model_dump") else dict(obs)

        @app.get("/state")
        async def state():
            st = env.state()
            return st.model_dump() if hasattr(st, "model_dump") else dict(st)

        return app


app = _make_app()


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
