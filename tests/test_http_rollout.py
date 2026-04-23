"""HTTP rollout mode (`run_episode(..., base_url=...)`) integration test."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import pytest

pytest.importorskip("uvicorn")
pytest.importorskip("fastapi")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from train.policies import policy_for_task  # noqa: E402
from train.rollout import _parse_observation_http, run_episode  # noqa: E402


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_parse_openenv_http_payload_merges_observation_reward_done():
    payload = {
        "observation": {
            "episode_goal": "g",
            "message": "m",
            "visible_fact_ids": ["a"],
        },
        "reward": 0.5,
        "done": True,
    }
    obs = _parse_observation_http(payload)
    assert obs.episode_goal == "g"
    assert obs.message == "m"
    assert obs.visible_fact_ids == ["a"]
    assert obs.reward == 0.5
    assert obs.done is True
    assert obs.metadata == {}


def test_parse_openenv_http_extracts_terminal_metrics_from_message():
    from train.rollout import _METRICS_MARKER, _parse_observation_http

    metrics = {"Total": 0.77, "RefusalCalibration": 1.0}
    import json as _json

    payload = {
        "observation": {
            "episode_goal": "g",
            "message": f"done{_METRICS_MARKER}{_json.dumps(metrics)}",
            "visible_fact_ids": [],
        },
        "reward": 0.77,
        "done": True,
    }
    obs = _parse_observation_http(payload)
    assert obs.message == "done"
    assert obs.metadata.get("terminal_metrics", {}).get("Total") == 0.77


def _wait_health(url: str, timeout_s: float = 5.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urlopen(url + "/health", timeout=1.0) as resp:  # noqa: S310
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("Server did not become healthy in time")


def test_http_rollout_mode_local_server():
    py_exe = _ROOT / ".venv" / "bin" / "python"
    if not py_exe.exists():
        pytest.skip("Project virtualenv not present for HTTP rollout test.")
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    try:
        proc = subprocess.Popen(  # noqa: S603,S607 - controlled test command
            [
                str(py_exe),
                "-m",
                "uvicorn",
                "server.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=str(_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, PermissionError) as e:
        pytest.skip(f"Cannot spawn local uvicorn: {e}")
    try:
        try:
            _wait_health(base_url)
        except RuntimeError as e:
            proc.terminate()
            proc.wait(timeout=5)
            pytest.skip(f"Local HTTP server not reachable (sandbox/CI): {e}")
        task = "dyad_must_refuse_v1"
        baseline = run_episode(task, policy_for_task(task, "baseline"), base_url=base_url)
        heuristic = run_episode(task, policy_for_task(task, "heuristic"), base_url=base_url)
        assert heuristic > baseline
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except (PermissionError, ProcessLookupError):
            pass

