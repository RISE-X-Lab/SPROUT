"""Harbor's own trials on ``PodmanEnvironment``: the rewards that the worker's ``harbor`` grade must equal.

Harbor runs the trial of its ``oracle`` or ``nop`` agent as it runs any trial. Sprout gives only the
environment. Import this module after ``pytest.importorskip("harbor")``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from tests.sprout_sandbox.live import NO_RUNTIME
from tests.support.live import live_podman_problem, live_store
from tests.support.paths import RUNTIME_BIN


def live_harbor_store() -> Path:
    """The store of the large tests, for a module fixture. Skips with the reason when it is not usable."""
    store = live_store()
    problem = live_podman_problem(store) or (None if RUNTIME_BIN.is_file() else NO_RUNTIME)
    if problem:
        pytest.skip(problem)
    return store.resolve()


def harbor_trial(output: Path, agent: str, *, task: Path, store: Path) -> Path:
    """The directory of the trial of Harbor's ``agent`` on ``task``, in ``PodmanEnvironment`` on ``store``."""
    config = TrialConfig.model_validate(
        {
            "task": {"path": str(task)},
            "trial_name": "%s-%s" % (task.name, agent),
            "trials_dir": str(output),
            "agent_timeout_multiplier": 1.0,
            "agent": {"name": agent},
            "environment": {
                "import_path": "sprout.eval.harbor.podman:PodmanEnvironment",
                "delete": True,
                "kwargs": {"runtime_bin": str(RUNTIME_BIN), "store": str(store)},
            },
        }
    )
    trial = asyncio.run(Trial.create(config))
    asyncio.run(trial.run())
    return output / ("%s-%s" % (task.name, agent))


def harbor_reward(trial: Path) -> float | None:
    """The reward in the ``result.json`` of a Harbor trial. Raises ``AssertionError`` if the trial raised."""
    result = json.loads((trial / "result.json").read_text())
    assert not result.get("exception_info"), result["exception_info"]
    reward: float | None = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return reward


def receipt(trial: Path) -> dict:
    """The one receipt that ``PodmanEnvironment`` wrote into the directory of a trial."""
    [path] = sorted(trial.glob("*.podman.json"))
    record: dict = json.loads(path.read_text())
    return record
