"""Optional: pydantic Action/Observation subclasses when openenv-core is installed."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("openenv.core.env_server")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from models import MembraneAction, MembraneObservation  # noqa: E402


def test_pydantic_models_roundtrip():
    a = MembraneAction(verb="SEND", surface="AGENT_DM", content="hello")
    d = a.model_dump()
    a2 = MembraneAction(**d)
    assert a2.verb == "SEND"
    o = MembraneObservation(done=False, message="m")
    assert o.model_dump()["message"] == "m"
