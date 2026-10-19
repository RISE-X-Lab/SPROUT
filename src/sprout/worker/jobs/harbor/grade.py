"""The ``harbor`` grade: a Harbor task's verifier, run the way Harbor 0.22.0 runs it (ADR-0003).

``grade_harbor`` starts the agent's sandbox from the grade's ``snapshot_id``, or attaches to the
one that the rollout ``in_place_job`` kept (ADR-0004). It runs the task's ``[[verifier.collect]]``
commands there and copies out ``/logs/artifacts`` and the task's ``artifacts``. A verifier that
shares the agent's environment then runs in that sandbox. A separate verifier runs in a sandbox
of its own, which starts from ``tests/Dockerfile`` (``overlay``) and gets the artifacts at their
own paths. ``tests/test.sh`` runs within the task's verifier timeout, under its verifier network
policy, and as the image's USER when the grade's backend turns ``image_user`` on
(``worker.sandbox.image_shell``). The reward comes from ``/logs/verifier/reward.json`` or else
``reward.txt``. The grade destroys every sandbox that it starts or attaches to before it returns.

The result is ``{"status": "completed", "reward": float, "reason": str | None, "verifier": {...}}``.
``reason`` is None when the verifier wrote the reward. Otherwise the reward is 0, and ``reason``
is ``sandbox_lost``, ``verifier_timed_out``, ``reward_missing`` or ``reward_unreadable``. A grade
that cannot be made raises: a sandbox that does not start, a file that does not move. The
attempt directory keeps ``verifier/`` (the sandbox's ``/logs/verifier``) and ``artifacts/``.
"""

from __future__ import annotations

import json
import math
import shlex
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, NoReturn, TypeGuard

from sprout.contract.harbor_task import HarborTask, load_task
from sprout.runstore.specs import GradeSpec
from sprout.worker.jobs.harbor.files import download_dir, download_file, run, upload_dir, upload_file
from sprout.worker.jobs.harbor.overlay import ImageOverlay, parse_overlay, replay
from sprout.worker.sandbox.backends import with_egress
from sprout.worker.sandbox.commands import checked
from sprout.worker.sandbox.image_shell import ImageShell, image_shell

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession

#: The network policy of a Harbor task phase that may reach the internet.
_PUBLIC = "public"
#: Harbor's paths in a sandbox (``harbor.models.trial.paths.EnvironmentPaths``).
_TESTS = "/tests"
_VERIFIER_LOGS = "/logs/verifier"
_ARTIFACTS = "/logs/artifacts"
#: Harbor's verifier command for ``tests/test.sh`` (``harbor.verifier.verifier``).
_TEST = "(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1"
#: The seconds that a preparing command (mkdir, chmod) may take. Arbitrary.
_STEP_TIMEOUT_S = 300


def grade_harbor(
    spec: GradeSpec,
    directory: Path,
    *,
    session_factory: Callable[..., SandboxSession],
    kept_sandbox: dict[str, Any] | None,
) -> dict[str, Any]:
    """The reward that the task's verifier gives the work of the grade's rollout (module docstring).

    ``kept_sandbox`` is the ``kept_sandbox`` of the rollout ``in_place_job``, None when it is gone.
    """
    if spec.task_dir is None:
        raise ValueError("A harbor grade needs the task_dir of its Harbor task")
    task = load_task(Path(spec.task_dir))
    agent: SandboxSession | None
    if spec.in_place_job is None:
        agent, lost = _agent_sandbox(spec, task, session_factory=session_factory), ""
    else:
        agent, lost = _kept_sandbox(spec, task, session_factory=session_factory, kept_sandbox=kept_sandbox)
    if agent is None:
        return {
            "status": "completed",
            "reward": 0.0,
            "reason": "sandbox_lost",
            "verifier": {"in_place": True, "error": lost},
        }
    try:
        record = {"in_place": spec.in_place_job is not None, **_collect(agent, task, directory / "artifacts")}
        if task.verifier_shares_environment:
            shell = image_shell(agent, backend=spec.backend, image=task.docker_image)
            return _verify(
                agent,
                task,
                directory,
                record=record,
                upload_tests=True,
                working_dir=task.workdir,
                shell=shell,
            )
    finally:
        agent.destroy()
    # HarborTask gives each verifier that does not share the environment its tests/Dockerfile.
    assert task.verifier_dockerfile is not None
    overlay = parse_overlay(task.verifier_dockerfile)
    verifier = _verifier_sandbox(spec, task, overlay=overlay, session_factory=session_factory)
    try:
        _restore(verifier, source=directory / "artifacts", paths=record["artifacts"])
        shell = image_shell(verifier, backend=spec.backend, image=overlay.base)
        return _verify(
            verifier, task, directory, record=record, upload_tests=False, working_dir=None, shell=shell
        )
    finally:
        verifier.destroy()


def _agent_sandbox(
    spec: GradeSpec, task: HarborTask, *, session_factory: Callable[..., SandboxSession]
) -> SandboxSession:
    """A fresh sandbox from the agent's snapshot, on the network of the phase that runs in it."""
    if spec.snapshot_id is None:
        raise ValueError("A harbor grade names exactly one of snapshot_id and in_place_job")
    network = task.verifier_network if task.verifier_shares_environment else task.agent_network
    return _started(
        session_factory(quiet=True, backend=with_egress(spec.backend, allow=network == _PUBLIC)),
        image=spec.snapshot_id,
        resources=_shape(spec.resources, cpus=task.cpus, memory_mb=task.memory_mb),
        failure="Could not restore the agent's snapshot",
    )


def _kept_sandbox(
    spec: GradeSpec,
    task: HarborTask,
    *,
    session_factory: Callable[..., SandboxSession],
    kept_sandbox: dict[str, Any] | None,
) -> tuple[SandboxSession | None, str]:
    """``(the sandbox that the rollout kept, attached, "")``, or ``(None, why it is gone)``."""
    if kept_sandbox is None:
        return None, "this worker keeps no such sandbox, or its keep_until has passed"
    session = session_factory(quiet=True, backend=spec.backend)
    if not session.attach(kept_sandbox["sandbox_id"]):
        return None, session.create_error
    if task.verifier_shares_environment:
        # The kept sandbox is on the agent's network. The verifier runs on the verifier's network.
        try:
            session.set_network(task.verifier_network == _PUBLIC)
        except BaseException:
            session.destroy()
            raise
    return session, ""


def _verifier_sandbox(
    spec: GradeSpec,
    task: HarborTask,
    *,
    overlay: ImageOverlay,
    session_factory: Callable[..., SandboxSession],
) -> SandboxSession:
    """A separate verifier's sandbox: the FROM image of ``tests/Dockerfile``, its COPY and RUN replayed."""
    session = _started(
        session_factory(
            quiet=True, backend=with_egress(spec.backend, allow=task.verifier_network == _PUBLIC)
        ),
        image=overlay.base,
        resources=_shape(spec.resources, cpus=task.verifier_cpus, memory_mb=task.verifier_memory_mb),
        failure=f"Could not start the verifier sandbox from {overlay.base}",
    )
    try:
        # Harbor runs a RUN step in the image's WORKDIR, where the runtime starts.
        workdir = checked(session, "pwd -P", _STEP_TIMEOUT_S).strip()
        replay(session, overlay, cwd=workdir, env={}, timeout_s=int(spec.timeout_s))
    except BaseException:
        session.destroy()
        raise
    return session


def _started(
    session: SandboxSession, *, image: str, resources: dict[str, Any], failure: str
) -> SandboxSession:
    if not session.create(image, resources):
        raise RuntimeError(f"{failure}: {session.create_error}")
    return session


def _shape(resources: dict[str, Any], *, cpus: int | None, memory_mb: int | None) -> dict[str, Any]:
    """The grade's sandbox shape, with the CPUs and memory that the task asks for."""
    return {
        **resources,
        **({"cpu": cpus} if cpus is not None else {}),
        **({"memory_mb": memory_mb} if memory_mb is not None else {}),
    }


def _collect(session: SandboxSession, task: HarborTask, destination: Path) -> dict[str, Any]:
    """Run the collect commands, then copy the artifacts under ``destination``. Best effort, as in Harbor.

    Returns ``{"collect": [{command, exit_code, timed_out}], "artifacts": [the sandbox paths copied]}``.
    """
    commands: list[dict[str, object]] = []
    for collect in task.collect:
        try:
            code, timed_out, _ = run(
                session, collect.command, timeout_s=collect.timeout_s, working_dir=task.workdir
            )
        except RuntimeError:
            code, timed_out = None, False
        commands.append({"command": collect.command, "exit_code": code, "timed_out": timed_out})
    copied = []
    for path in _artifact_paths(task.artifacts):
        host = destination / path.lstrip("/")
        try:
            is_dir, _, _ = run(session, "test -d " + shlex.quote(path), timeout_s=_STEP_TIMEOUT_S)
            if is_dir == 0:
                download_dir(session, path, host)
            else:
                download_file(session, path, host)
        except RuntimeError:
            continue
        copied.append(path)
    return {"collect": commands, "artifacts": copied}


def _artifact_paths(artifacts: tuple[str, ...]) -> list[str]:
    """``/logs/artifacts`` and then each listed path, without one that overlaps a path before it."""
    paths: list[str] = []
    for path in (_ARTIFACTS, *artifacts):
        candidate = PurePosixPath(path.rstrip("/") or "/")
        if not any(
            candidate == PurePosixPath(kept)
            or PurePosixPath(kept) in candidate.parents
            or candidate in PurePosixPath(kept).parents
            for kept in paths
        ):
            paths.append(str(candidate))
    return paths


def _restore(session: SandboxSession, *, source: Path, paths: list[str]) -> None:
    """Copy each artifact collected under ``source`` into the sandbox at its own path."""
    for path in paths:
        host = source / path.lstrip("/")
        if host.is_dir():
            checked(session, f"rm -rf {shlex.quote(path)} && mkdir -p {shlex.quote(path)}", _STEP_TIMEOUT_S)
            upload_dir(session, host, path)
        else:
            upload_file(session, host, path)


def _verify(
    session: SandboxSession,
    task: HarborTask,
    directory: Path,
    *,
    record: dict[str, Any],
    upload_tests: bool,
    working_dir: str | None,
    shell: ImageShell | None,
) -> dict[str, Any]:
    """Run ``tests/test.sh`` in the sandbox of ``session`` and read the reward.

    ``test.sh`` runs as the image's user when ``shell`` names one, else as root. The result's
    ``verifier`` holds ``record`` and the outcome of ``test.sh``.
    """
    # An empty /logs/verifier, as Harbor gives a separate verifier: a planted reward file does not count.
    checked(
        session,
        f"rm -rf {_VERIFIER_LOGS} && mkdir -p {_VERIFIER_LOGS} && chmod 777 {_VERIFIER_LOGS}",
        _STEP_TIMEOUT_S,
    )
    if upload_tests:
        upload_dir(session, task.tests_dir, _TESTS)
    checked(session, f"chmod +x {_TESTS}/test.sh", _STEP_TIMEOUT_S)
    test = _TEST if shell is None else shell.command(_TEST, working_dir=working_dir)
    code, timed_out, _ = run(session, test, timeout_s=task.verifier_timeout_s, working_dir=working_dir)
    logs = directory / "verifier"
    try:
        download_dir(session, _VERIFIER_LOGS, logs)
    except RuntimeError:
        # After a timeout the logs are only for a reader; without one the reward is in them.
        if not timed_out:
            raise
    reward, reason, rewards = (0.0, "verifier_timed_out", None) if timed_out else _reward(logs)
    return {
        "status": "completed",
        "reward": reward,
        "reason": reason,
        "verifier": {
            **record,
            "shared": task.verifier_shares_environment,
            "exit_code": code,
            "timed_out": timed_out,
            "rewards": rewards,
            "logs": str(logs),
        },
    }


def _reward(logs: Path) -> tuple[float, str | None, dict[str, Any] | None]:
    """``(reward, reason, rewards)`` from the reward file that ``test.sh`` wrote, read as Harbor reads it."""
    if (logs / "reward.json").is_file():
        try:
            rewards = json.loads((logs / "reward.json").read_text(), parse_constant=_refuse)
        except ValueError:
            return 0.0, "reward_unreadable", None
        value = rewards.get("reward") if isinstance(rewards, dict) else None
        if not _finite(value):
            return 0.0, "reward_unreadable", None
        return float(value), None, rewards
    if (logs / "reward.txt").is_file():
        try:
            value = float((logs / "reward.txt").read_text())
        except ValueError:
            return 0.0, "reward_unreadable", None
        return (value, None, {"reward": value}) if _finite(value) else (0.0, "reward_unreadable", None)
    return 0.0, "reward_missing", None


def _finite(value: object) -> TypeGuard[float]:
    # Exact types: a JSON true or false is a bool, which is an int but not a reward.
    return (type(value) is int or type(value) is float) and math.isfinite(value)


def _refuse(constant: str) -> NoReturn:
    # A job result is stored as JSON, which has no NaN or Infinity.
    raise ValueError(f"{constant} is not a reward")
