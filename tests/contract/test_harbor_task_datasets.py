"""``load_task`` reads each task of the three datasets as Harbor 0.22.0's own task model reads it.

The test needs ``harbor==0.22.0`` and the deep-swe, terminal-bench-2-1 and SWE-bench_Pro-os datasets
under ``$SPROUT_HARBOR_DATASETS`` (default ``~/ash-data/datasets``). It reads the datasets and starts no
sandbox:

    pytest tests/contract/test_harbor_task_datasets.py -m slow
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

from sprout.contract.harbor_task import CollectCommand, HarborTask, load_task

harbor_task = pytest.importorskip("harbor.models.task.task", reason="needs harbor==0.22.0: pip install -e .")

from harbor.models.task.verifier_mode import (  # noqa: E402
    VerifierEnvironmentMode,
    resolve_effective_verifier_env_config,
    resolve_task_verifier_mode,
)
from harbor.models.trial.config import AgentConfig, EnvironmentConfig  # noqa: E402
from harbor.trial.network_policy import resolve_trial_network_plan  # noqa: E402

DATASETS = Path(os.environ.get("SPROUT_HARBOR_DATASETS", "~/ash-data/datasets")).expanduser()
TASK_ROOTS = ("deep-swe/tasks", "terminal-bench-2-1/tasks", "SWE-bench_Pro-os/v2/tasks")
TASK_DIRS = sorted(path.parent for root in TASK_ROOTS for path in (DATASETS / root).glob("*/task.toml"))

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all((DATASETS / root).is_dir() for root in TASK_ROOTS),
        reason="needs %s under $SPROUT_HARBOR_DATASETS (default ~/ash-data/datasets)" % ", ".join(TASK_ROOTS),
    ),
]


def harbor_reading(task_dir: Path) -> HarborTask:
    """The ``HarborTask`` that Harbor's task model and its trial resolution give for ``task_dir``."""
    # Harbor warns about the deprecated allow_internet and about overlapping artifact paths.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        task = harbor_task.Task(task_dir)
    config = task.config
    mode = resolve_task_verifier_mode(config)
    shares = mode == VerifierEnvironmentMode.SHARED
    network = resolve_trial_network_plan(config, AgentConfig(), EnvironmentConfig(), None, verifier_mode=mode)
    verifier_environment = resolve_effective_verifier_env_config(config, None)
    dockerfile = task.paths.environment_dir / "Dockerfile"
    return HarborTask(
        task_dir=task.task_dir,
        name=task.name,
        instruction=task.instruction,
        docker_image=config.environment.docker_image,
        dockerfile=dockerfile if dockerfile.is_file() else None,
        workdir=config.environment.workdir,
        cpus=config.environment.cpus,
        memory_mb=config.environment.memory_mb,
        storage_mb=config.environment.storage_mb,
        agent_timeout_s=config.agent.timeout_sec,
        agent_network=network.agent_phase.network_mode.value,
        verifier_timeout_s=config.verifier.timeout_sec,
        verifier_network=network.verifier_phase.network_mode.value,
        verifier_shares_environment=shares,
        verifier_dockerfile=None if shares else task.paths.tests_dir / "Dockerfile",
        verifier_cpus=None if shares else verifier_environment.cpus,
        verifier_memory_mb=None if shares else verifier_environment.memory_mb,
        verifier_storage_mb=None if shares else verifier_environment.storage_mb,
        collect=tuple(
            CollectCommand(command=hook.command, timeout_s=hook.timeout_sec)
            for hook in config.verifier.collect
        ),
        artifacts=tuple(config.artifacts),
        solution_dir=task.paths.solution_dir,
        tests_dir=task.paths.tests_dir,
    )


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=[str(path.relative_to(DATASETS)) for path in TASK_DIRS])
def test_load_task_reads_each_dataset_task_as_harbor_does(task_dir):
    expected = harbor_reading(task_dir)

    task = load_task(task_dir)

    assert task == expected
