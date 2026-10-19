"""The suite runs against the source tree, and each test can use the fixtures of ``tests/support``.

``src/`` and the repository root are on the import path (``pyproject.toml``), and on PYTHONPATH for
the child processes that some tests start.
"""

import os

import pytest

# Before the first import of tests.support, so that pytest rewrites the asserts of its helpers.
pytest.register_assert_rewrite("tests.support")

from tests.support.paths import REPO  # noqa: E402

_current = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), str(REPO)] + ([_current] if _current else []))

pytest_plugins = (
    "tests.support.tools",
    "tests.support.live",
    "tests.support.runstore",
    "tests.support.process",
)
