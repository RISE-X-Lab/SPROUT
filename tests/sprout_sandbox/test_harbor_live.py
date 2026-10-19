"""Harbor's own trials, its phase policies and a non-root image on ``PodmanEnvironment``, on the real store.

These large tests skip with a reason when the store, podman or the runtime binary cannot be used, or
when Harbor is not installed. Run the file alone, while no experiment uses the store:

    pytest tests/sprout_sandbox/test_harbor_live.py -m slow
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.sprout_sandbox.live import Leftovers, nothing, remove_leftovers
from tests.support.paths import HARBOR_SHELL_TASK

pytest.importorskip("harbor", reason="install the harbor extra: pip install -e '.[harbor]'")

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy  # noqa: E402
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from sprout.eval.harbor.podman import PodmanEnvironment  # noqa: E402
from tests.sprout_sandbox.harbor_trials import harbor_reward, harbor_trial, live_harbor_store, receipt  # noqa: E402

pytestmark = pytest.mark.slow

#: A small public image whose User is not root: Debian's haproxy 3.0.30, with the User haproxy and the
#: WorkingDir /var/lib/haproxy. A digest pins it, as ``PodmanEnvironment`` pins each image.
USER_IMAGE = (
    "docker.io/library/haproxy@sha256:836f65081bc532d2d024641ca868122105cca2f150effd2e04f8c155258b9ef8"
)
#: Prints ``online`` when the container has egress, and ``offline`` when it has none.
EGRESS = (
    "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\" 2>/dev/null "
    "&& echo online || echo offline"
)


@dataclass(frozen=True)
class Trials:
    """The directory of each agent's trial, the store, and what the trials left on it and in podman."""

    directories: dict[str, Path]
    store: Path
    left: Leftovers


@pytest.fixture(scope="module")
def trials(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Trials]:
    """Harbor's trials of its ``oracle`` and ``nop`` agents on the fixture task, run once for the module."""
    store = live_harbor_store()
    output = tmp_path_factory.mktemp("trials")
    before = Leftovers.now(store)
    try:
        directories = {
            agent: harbor_trial(output, agent, task=HARBOR_SHELL_TASK, store=store)
            for agent in ("oracle", "nop")
        }
        yield Trials(directories=directories, store=store, left=Leftovers.now(store).appeared_since(before))
    finally:
        asyncio.run(remove_leftovers(store, before))


@pytest.fixture
async def started() -> AsyncIterator[Callable[[PodmanEnvironment], Awaitable[PodmanEnvironment]]]:
    """A starter of environments. After the test, each environment stops and deletes its sandbox."""
    running: list[PodmanEnvironment] = []

    async def start(environment: PodmanEnvironment) -> PodmanEnvironment:
        running.append(environment)
        await environment.start()
        return environment

    yield start
    for environment in running:
        await environment.stop(delete=True)


def podman_environment(
    *, store: Path, runtime_bin: Path, trial_dir: Path, environment_dir: Path, image: str, **policies: object
) -> PodmanEnvironment:
    """A ``PodmanEnvironment`` on ``store`` for one session, with the shape of the fixture task."""
    return PodmanEnvironment(
        environment_dir=environment_dir,
        environment_name=environment_dir.parent.name,
        session_id="%s__env" % trial_dir.name,
        trial_paths=TrialPaths(trial_dir=trial_dir),
        task_env_config=EnvironmentConfig(docker_image=image, cpus=1, memory_mb=512),
        runtime_bin=str(runtime_bin),
        store=str(store),
        **policies,
    )


def overlay_task(root: Path, image: str) -> Path:
    """A ``tests/`` directory whose Dockerfile is an overlay on ``image``: a COPY, and a RUN as root."""
    tests = root / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (tests / "Dockerfile").write_text(
        "FROM %s\nCOPY test.sh /tests/test.sh\n"
        "RUN chmod +x /tests/test.sh && touch /usr/local/root-only && id -u > /tmp/run-as\n" % image
    )
    return tests


def test_harbors_oracle_scores_1_and_nop_scores_0_on_podman(trials):
    rewards = {agent: harbor_reward(directory) for agent, directory in trials.directories.items()}

    assert rewards == {"oracle": 1.0, "nop": 0.0}


def test_harbors_trials_on_podman_leave_no_container_network_sandbox_snapshot_or_socket(trials):
    """Harbor deletes each environment. The base of the image may stay, for the next run."""
    assert trials.left == nothing()


def test_the_receipt_of_a_trial_pins_the_image_and_records_its_shape_its_store_and_its_network(trials):
    record = receipt(trials.directories["oracle"])

    assert re.fullmatch(r"docker\.io/library/python@sha256:[0-9a-f]{64}", record["image"]), record["image"]
    assert (record["backend"], record["store"], record["resources"], record["network"]["network_mode"]) == (
        "podman",
        str(trials.store),
        {"cpu": 1, "memory_mb": 512},
        "public",
    )


@pytest.mark.anyio
async def test_harbors_phase_policy_toggles_the_egress_of_the_running_container(
    started, live_podman_store, live_runtime_bin, tmp_path
):
    """The task's agent phase is no-network. Its other phases are public."""
    env = await started(
        podman_environment(
            store=live_podman_store,
            runtime_bin=live_runtime_bin,
            trial_dir=tmp_path,
            environment_dir=HARBOR_SHELL_TASK / "environment",
            image="python:3.11-slim",
            network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
            phase_network_policies=[NetworkPolicy(network_mode=NetworkMode.PUBLIC)],
        )
    )

    offline = await env.exec(EGRESS, timeout_sec=30)
    await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC))
    online = await env.exec(EGRESS, timeout_sec=30)
    await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    again = await env.exec(EGRESS, timeout_sec=30)

    assert (offline.stdout, online.stdout, again.stdout) == ("offline\n", "online\n", "offline\n")


@pytest.mark.anyio
async def test_on_a_non_root_image_harbors_root_steps_run_as_root_and_other_commands_as_the_images_user(
    started, live_podman_store, live_runtime_bin, tmp_path
):
    """``start`` lays out ``/logs`` as root, and the overlay's RUN runs as root. A command that names no
    user runs as the image's User, in its WorkingDir, as in a microVM."""
    tests = overlay_task(tmp_path / "task", USER_IMAGE)
    env = await started(
        podman_environment(
            store=live_podman_store,
            runtime_bin=live_runtime_bin,
            trial_dir=tmp_path,
            environment_dir=tests,
            image=USER_IMAGE,
        )
    )

    agent = await env.exec("id -un && pwd && cat /tmp/run-as && ls /tests/test.sh /usr/local/root-only")
    root = await env.exec("id -un", user="root")
    logs = await env.exec("stat -c '%a %U' /logs/verifier")

    assert agent.stdout == "haproxy\n/var/lib/haproxy\n0\n/tests/test.sh\n/usr/local/root-only\n"
    assert (root.stdout, logs.stdout) == ("root\n", "777 root\n")
