"""Fixtures of the job tests: git without the host's config.

``tests/worker/conftest.py`` gives ``run_sandboxes``, the directory sandboxes for each run.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def git_without_host_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Git in each test reads no global or system config, which could change a diff or refuse a repository."""
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
