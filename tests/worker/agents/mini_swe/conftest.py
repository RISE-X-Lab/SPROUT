"""Fixtures for the tests of the mini slot: no network at mini's first import, and no backoff sleep."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def local_litellm_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    """LiteLLM reads its bundled prices, and does not fetch them, when the slot first imports it."""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")


@pytest.fixture(autouse=True)
def backoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[float]:
    """The seconds that mini's model waits before each retry. It records them and does not wait.

    A scripted model answers 500 after its last reply, and mini retries a 500 for minutes.
    """
    if importlib.util.find_spec("minisweagent") is None:
        return []
    # This import of the model module is mini's first import if no slot ran yet. Like the slot,
    # give mini an empty config directory, so that it loads no developer's .env.
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    delays: list[float] = []
    clock = SimpleNamespace(monotonic=time.monotonic, sleep=delays.append)
    monkeypatch.setattr("sprout.worker.agents.mini_swe.model.time", clock)
    return delays
