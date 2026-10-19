"""The worker's ``harbor`` grade on podman gives the rewards of Harbor's own verifier, on the real store.

Harbor runs its ``oracle`` and ``nop`` agents on the fixture task (``harbor_trial``). The worker runs
the same agents as rollouts, through the attempt entry point, and grades each one from its final
snapshot and in the sandbox that it kept. These large tests skip with a reason when the store, podman
or the runtime binary cannot be used, or when Harbor is not installed. Run the file alone:

    pytest tests/sprout_sandbox/test_harbor_grade_live.py -m slow
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from tests.sprout_sandbox.live import Leftovers, nothing, remove_leftovers
from tests.support.paths import HARBOR_SHELL_TASK, RUNTIME_BIN
from tests.support.process import kept_process_state

pytest.importorskip("harbor", reason="install the harbor extra: pip install -e '.[harbor]'")

from sprout.worker.jobs.rollout import execute  # noqa: E402
from tests.sprout_sandbox.harbor_trials import harbor_reward, harbor_trial, live_harbor_store  # noqa: E402

pytestmark = pytest.mark.slow

#: A small public image whose User is not root: Debian's haproxy 3.0.30, with the User haproxy, the
#: WorkingDir /var/lib/haproxy, and HAPROXY_VERSION in its Env.
USER_IMAGE = (
    "docker.io/library/haproxy@sha256:836f65081bc532d2d024641ca868122105cca2f150effd2e04f8c155258b9ef8"
)
#: A solve.sh and a test.sh that record who ran them, and the HAPROXY_VERSION that each one saw.
WHO_SOLVES = '#!/bin/bash\necho "$(id -un) $HAPROXY_VERSION" > /var/lib/haproxy/solved-by\n'
WHO_TESTS = (
    '#!/bin/bash\necho "$(cat /var/lib/haproxy/solved-by), $(id -un) $HAPROXY_VERSION" '
    "> /logs/verifier/identity.txt\necho 1 > /logs/verifier/reward.txt\n"
)


@dataclass(frozen=True)
class Grades:
    """Harbor's rewards, the worker's grades of the same agents, and what the jobs left behind."""

    harbor: dict[str, float | None]
    from_snapshot: dict[str, dict[str, Any]]
    in_kept_sandbox: dict[str, dict[str, Any]]
    #: A second grade of the sandbox that the oracle's rollout kept, after the first grade destroyed it.
    again: dict[str, Any]
    left: Leftovers


def backend(store: Path, **options: object) -> dict[str, Any]:
    """The backend section of a job on the podman ``store``."""
    return {"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(RUNTIME_BIN), **options}}


def rollout(
    directory: Path, agent: str, *, task: Path, image: str, backend: dict, keep: bool = False
) -> dict:
    """The result of a rollout of Harbor's ``agent`` on ``task``. Raises ``AssertionError`` if it fails."""
    directory.mkdir(parents=True)
    spec = {
        "prompt": "Solve the task.",
        "agent": agent,
        "harbor_task": str(task),
        "backend": backend,
        "sandbox_image": image,
        "timeout_s": 600,
        "sandbox_resources": {"cpu": 1, "memory_mb": 512},
        "keep_sandbox": keep,
    }
    request = {
        "kind": "rollout",
        "attempt_id": "rollout-" + agent,
        "effective_spec": spec,
        "profile_config": {},
    }
    result = execute(request, directory)
    assert result["status"] == "completed" and result["final_snapshot_id"], result
    return result


def harbor_grade(
    directory: Path, *, task: Path, backend: dict, kept: dict | None = None, **source: str
) -> dict:
    """The result of the ``harbor`` grade of ``task``, from a snapshot or in the ``kept`` sandbox."""
    directory.mkdir(parents=True)
    spec = {"benchmark": "harbor", "task_dir": str(task), "backend": backend, **source}
    request = {
        "kind": "grade",
        "attempt_id": directory.name,
        "effective_spec": spec,
        "profile_config": {},
        "kept_sandbox": kept,
    }
    return execute(request, directory)


def by_agent(results: dict[str, dict[str, Any]], *keys: str) -> dict[str, Any]:
    """The value at ``keys`` in the result of each agent."""
    values = {}
    for agent, result in results.items():
        for key in keys:
            result = result[key]
        values[agent] = result
    return values


def identity_task(root: Path) -> Path:
    """A task on ``USER_IMAGE`` whose verifier shares the agent's sandbox, and records who ran what."""
    (root / "solution").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "task.toml").write_text(
        '[agent]\ntimeout_sec = 300\n[verifier]\ntimeout_sec = 300\n[environment]\ndocker_image = "%s"\n'
        % USER_IMAGE
    )
    (root / "instruction.md").write_text("Say who you are.\n")
    (root / "solution" / "solve.sh").write_text(WHO_SOLVES)
    (root / "tests" / "test.sh").write_text(WHO_TESTS)
    return root


@pytest.fixture(scope="module")
def grades(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Grades]:
    """Harbor's trials, and the worker's rollouts and grades of the same agents, run once for the module."""
    store = live_harbor_store()
    root = tmp_path_factory.mktemp("grades")
    task, image, podman = HARBOR_SHELL_TASK, "docker.io/library/python:3.11-slim", backend(store)
    agents = ("oracle", "nop")
    before = Leftovers.now(store)
    try:
        # A rollout sets the umask and the environment of the process. The tests run in the state before.
        with kept_process_state():
            trials = tmp_path_factory.mktemp("harbor")
            harbor = {
                agent: harbor_reward(harbor_trial(trials, agent, task=task, store=store)) for agent in agents
            }
            rollouts = {
                agent: rollout(root / agent, agent, task=task, image=image, backend=podman)
                for agent in agents
            }
            from_snapshot = {
                agent: harbor_grade(
                    root / ("grade-" + agent),
                    task=task,
                    backend=podman,
                    snapshot_id=rollouts[agent]["final_snapshot_id"],
                )
                for agent in agents
            }
            kept = {
                agent: rollout(
                    root / ("kept-" + agent), agent, task=task, image=image, backend=podman, keep=True
                )
                for agent in agents
            }
            in_kept_sandbox = {
                agent: harbor_grade(
                    root / ("kept-grade-" + agent),
                    task=task,
                    backend=podman,
                    kept=kept[agent]["kept_sandbox"],
                    in_place_job="rollout-job",
                )
                for agent in agents
            }
            again = harbor_grade(
                root / "grade-again",
                task=task,
                backend=podman,
                kept=kept["oracle"]["kept_sandbox"],
                in_place_job="rollout-job",
            )
            left = Leftovers.now(store).appeared_since(before)
        yield Grades(
            harbor=harbor,
            from_snapshot=from_snapshot,
            in_kept_sandbox=in_kept_sandbox,
            again=again,
            left=left,
        )
    finally:
        asyncio.run(remove_leftovers(store, before))


@pytest.fixture
def identity_store(live_podman_store: Path, live_runtime_bin: Path) -> Iterator[Path]:
    """The live store. What the test's jobs leave on it goes after the test."""
    before = Leftovers.now(live_podman_store)
    yield live_podman_store
    asyncio.run(remove_leftovers(live_podman_store, before))


def test_harbors_own_trials_score_1_for_the_oracle_and_0_for_nop(grades):
    assert grades.harbor == {"oracle": 1.0, "nop": 0.0}


def test_the_harbor_grade_from_the_final_snapshot_gives_harbors_rewards(grades):
    assert by_agent(grades.from_snapshot, "reward") == grades.harbor
    assert by_agent(grades.from_snapshot, "reason") == {"oracle": None, "nop": None}, (
        "the verifier wrote each reward"
    )


def test_the_harbor_grade_in_the_kept_sandbox_gives_harbors_rewards(grades):
    assert by_agent(grades.in_kept_sandbox, "reward") == grades.harbor
    assert by_agent(grades.in_kept_sandbox, "verifier", "in_place") == {"oracle": True, "nop": True}


def test_a_second_grade_of_a_kept_sandbox_finds_it_lost_because_the_first_destroyed_it(grades):
    assert (grades.again["reward"], grades.again["reason"]) == (0.0, "sandbox_lost")


def test_the_rollouts_and_grades_leave_no_container_network_sandbox_or_socket_behind(grades):
    """The final snapshots of the rollouts stay, for a later grade or branch."""
    assert dataclasses.replace(grades.left, snapshots=frozenset()) == nothing()


@pytest.mark.parametrize(
    "image_options, identity",
    [
        pytest.param(
            {"image_user": True, "image_env": True}, "haproxy 3.0.30, haproxy 3.0.30", id="image-user-and-env"
        ),
        pytest.param({}, "root 3.0.30, root 3.0.30", id="defaults"),
    ],
)
def test_solve_and_test_run_as_the_images_user_when_the_backend_asks(
    identity_store, tmp_path, image_options, identity
):
    """As Harbor runs them on an image whose User is not root. A podman container starts with the
    image's Env, so each command sees it, whatever ``image_env`` says."""
    task = identity_task(tmp_path / "task")
    podman = backend(identity_store, **image_options)
    run = rollout(tmp_path / "rollout", "oracle", task=task, image=USER_IMAGE, backend=podman)

    grade = harbor_grade(tmp_path / "grade", task=task, backend=podman, snapshot_id=run["final_snapshot_id"])

    assert (grade["reward"], grade["reason"]) == (1.0, None), grade
    assert (tmp_path / "grade" / "verifier" / "identity.txt").read_text() == identity + "\n"
