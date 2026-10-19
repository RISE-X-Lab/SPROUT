"""Each module that another package imports from the agents or the gateway imports first in a fresh process.

A module-level import that closes a cycle fails only when its module is the first one imported,
so the suite, which imports the runner early, cannot show it. Each case starts its own interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from tests.support.paths import REPO


@pytest.mark.parametrize(
    "module",
    [
        "sprout.worker.agents.slot",
        "sprout.worker.agents.nop",
        "sprout.worker.agents.oracle",
        "sprout.worker.agents.mini_swe.slot",
        "sprout.worker.agents.mini_swe.history",
        "sprout.worker.agents.mini_swe.tools",
        "sprout.worker.gateway.auth",
    ],
)
def test_the_module_imports_as_the_first_import_of_a_process(module):
    completed = subprocess.run(
        [sys.executable, "-c", "import %s" % module],
        cwd=REPO,
        env={"PYTHONPATH": str(REPO / "src"), "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
