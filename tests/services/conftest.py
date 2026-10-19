"""Each services test that asks for ``paths`` runs with HOME and the two roots of the CLI in its ``tmp_path``.

So a wrong path in the code cannot reach the real ``~/.config/sprout``. When the test ends, the fixture
kills each process that names ``tmp_path`` in its argv, so a failed test leaves no service running.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout.services.state import CONFIG_DIR_VAR, STATE_DIR_VAR, Paths
from tests.services.helpers import kill_processes_naming


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Paths]:
    """The paths of a host whose config, state and home are ``config/``, ``state/`` and ``home/``."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv(CONFIG_DIR_VAR, str(tmp_path / "config"))
    monkeypatch.setenv(STATE_DIR_VAR, str(tmp_path / "state"))
    yield Paths.from_env()
    kill_processes_naming(str(tmp_path))
