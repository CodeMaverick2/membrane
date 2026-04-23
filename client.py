"""
HTTP client for MembraneEnv (stub).

When openenv-core is installed, prefer HTTPEnvClient / generated client
patterns from ../docs/02-reference-code-review-env.md
"""
from __future__ import annotations

from typing import Any, Dict, Optional

try:
    from openenv.core.http_env_client import HTTPEnvClient
except ImportError:
    HTTPEnvClient = None  # type: ignore[misc, assignment]


class MembraneEnvClient:
    """Thin placeholder; replace with official client subclass."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._inner: Optional[Any] = None
        if HTTPEnvClient is not None:
            self._inner = HTTPEnvClient(base_url=self.base_url)

    def sync(self) -> "MembraneEnvClient":
        return self

    def __enter__(self) -> "MembraneEnvClient":
        return self

    def __exit__(self, *args: object) -> None:
        pass
