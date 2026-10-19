"""``PodmanEnvironment``: Harbor's environment over one Sprout sandbox session (``docs/HARBOR_PLUGIN.md``).

The session is ``FakeSession``: it starts no sandbox, records what the environment asks of it and
answers each command with a scripted result. The image record comes from the fake podman of
``tests.support.tools``, so the environment pins the image as it does on a real host.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import subprocess
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("harbor", reason="install the harbor extra: pip install -e '.[harbor]'")

from harbor.models.task.config import (  # noqa: E402
    EnvironmentConfig,
    MCPServerConfig,
    NetworkMode,
    NetworkPolicy,
)
from harbor.models.trial.paths import TrialPaths  # noqa: E402

from sprout.eval.harbor import environment  # noqa: E402
from sprout.eval.harbor.podman import PodmanEnvironment  # noqa: E402
from sprout.worker.tools.interceptors import shell_command  # noqa: E402
from sprout.worker.tools.result import CommandOutcome, ToolResult  # noqa: E402
from sprout_sandbox.podman_cli import PodmanError  # noqa: E402
from sprout_sandbox.podman_images import digest_reference  # noqa: E402
from tests.support.tools import FakePodman  # noqa: E402

IMAGE = "public.ecr.aws/org/task:v1"
#: The digest that the fake podman reports for a reference: the sha256 of the reference.
PINNED = "public.ecr.aws/org/task@sha256:" + hashlib.sha256(IMAGE.encode()).hexdigest()
#: The OCI config that the fake podman reports for every image whose User the test does not set.
IMAGE_CONFIG = {
    "Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
    "WorkingDir": "/testbed",
    "User": "",
    "Entrypoint": None,
    "Cmd": ["python3"],
}
IMAGE_ENV = {"PATH": "/usr/local/bin:/usr/bin", "LANG": "C.UTF-8", "FOO": "a=b c"}
LOG_LAYOUT = (
    "mkdir -p /logs/agent /logs/user-agent /logs/verifier /logs/artifacts && "
    "chmod 777 /logs/agent /logs/user-agent /logs/verifier /logs/artifacts"
)


class FakePool:
    """The pool of a ``FakeSession``: what ``stop`` asks of it, in order."""

    def __init__(self, *, fail_destroy: bool) -> None:
        self.calls: list[str] = []
        self._fail_destroy = fail_destroy

    async def destroy(self, sandbox: object) -> None:
        self.calls.append("destroy")
        if self._fail_destroy:
            raise RuntimeError("the container is gone but its rootfs is not")

    async def forget(self, sandbox: object) -> None:
        self.calls.append("forget")

    async def close(self) -> None:
        self.calls.append("close")


class FakeSession:
    """A ``SandboxSession`` without a sandbox: it records each call and answers each command with ``answer``.

    ``stop`` also reaches the session's private pool, its loop and its destroy and release, so the fake
    has them, with the behavior of ``SandboxSession``: they act only on a sandbox that ``create`` made."""

    def __init__(self, *, backend: dict[str, Any], quiet: bool, behavior: Sessions) -> None:
        self.backend = backend
        self.sandbox_id = "sprout-0123456789ab"
        self.create_error = behavior.create_error or ""
        self.transfer_error = ""
        self.created: tuple[str, dict[str, int]] | None = None
        #: Each tool call as its arguments and the session timeout.
        self.calls: list[tuple[dict[str, Any], float]] = []
        #: Each upload as the destination and the bytes.
        self.uploads: list[tuple[str, bytes]] = []
        #: The source of each download, and the bytes that each download gives.
        self.downloads: list[str] = []
        self.network: list[bool] = []
        self.answer: Callable[[dict[str, Any]], ToolResult] = lambda arguments: ToolResult(True, "")
        self.download: bytes = b""
        self._pool = FakePool(fail_destroy=behavior.fail_destroy)
        self._sandbox: object | None = None
        self._loop = SimpleNamespace(drive=asyncio.run)

    def create(self, image: str, resources: dict[str, int]) -> bool:
        self.created = (image, resources)
        if self.create_error:
            return False
        self._sandbox = object()
        return True

    def execute(self, tool: str, arguments: dict[str, Any], timeout: float) -> ToolResult:
        self.calls.append((arguments, timeout))
        return self.answer(arguments)

    def upload_file(self, source: str | Path, destination: str) -> bool:
        self.uploads.append((destination, Path(source).read_bytes()))
        return True

    def download_file(self, source: str, destination: str | Path) -> bool:
        self.downloads.append(source)
        Path(destination).write_bytes(self.download)
        return True

    def set_network(self, allow: bool) -> None:
        self.network.append(allow)

    async def _destroy_async(self) -> None:
        if self._sandbox is not None:
            await self._pool.destroy(self._sandbox)
            self._sandbox = None

    async def _release_async(self) -> None:
        if self._sandbox is not None:
            await self._pool.forget(self._sandbox)
            self._sandbox = None

    def commands(self) -> list[str]:
        """The shell line of each command, in call order."""
        return [arguments["command"] for arguments, _ in self.calls]


@dataclass
class Sessions:
    """The fake sessions that the environments of one test made, and how each new one behaves."""

    made: list[FakeSession] = field(default_factory=list)
    #: The reason that ``create`` gives when it fails. None: ``create`` succeeds.
    create_error: str | None = None
    fail_destroy: bool = False


@pytest.fixture
def sessions(monkeypatch: pytest.MonkeyPatch, fake_podman: FakePodman) -> Sessions:
    """Every ``SandboxSession`` that an environment makes is a ``FakeSession``. The fake podman is on PATH."""
    behavior = Sessions()

    def make(**settings: Any) -> FakeSession:
        session = FakeSession(**settings, behavior=behavior)
        behavior.made.append(session)
        return session

    monkeypatch.setattr(environment, "SandboxSession", make)
    return behavior


def podman_environment(tmp_path: Path, fake_runtime_bin: Path, **changes: Any) -> PodmanEnvironment:
    """The environment that Harbor makes of a task with ``IMAGE``, 2 CPUs and 8 GB, with ``changes``."""
    settings: dict[str, Any] = {
        "environment_dir": tmp_path / "environment",
        "environment_name": "task",
        "session_id": "task__env",
        "trial_paths": TrialPaths(trial_dir=tmp_path / "trial"),
        "task_env_config": EnvironmentConfig(docker_image=IMAGE, cpus=2, memory_mb=8192, storage_mb=20480),
        "runtime_bin": str(fake_runtime_bin),
        "store": str(tmp_path / "store"),
        **changes,
    }
    settings["environment_dir"].mkdir(parents=True, exist_ok=True)
    (tmp_path / "trial").mkdir(exist_ok=True)
    (tmp_path / "store").mkdir(exist_ok=True)
    return PodmanEnvironment(**settings)


def started(env: PodmanEnvironment, sessions: Sessions) -> FakeSession:
    """Start ``env``, and return its session."""
    asyncio.run(env.start())
    return sessions.made[-1]


# --- start ----------------------------------------------------------------------------------


def test_start_creates_the_sandbox_on_the_store_from_the_pinned_image_in_the_tasks_shape(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """The rootfs is a subvolume of the store, so the shape has no disk."""
    session = started(podman_environment(tmp_path, fake_runtime_bin), sessions)

    assert session.backend == {
        "backend": "podman",
        "podman": {
            "store": str((tmp_path / "store").resolve()),
            "runtime_bin": str(fake_runtime_bin.resolve()),
            "podman_bin": "podman",
            "allow_internet": True,
        },
    }
    assert session.created == (PINNED, {"cpu": 2, "memory_mb": 8192})


@pytest.mark.parametrize(
    "image, pinned",
    [
        (
            "python:3.11-slim",
            "docker.io/library/python@sha256:"
            + hashlib.sha256(b"docker.io/library/python:3.11-slim").hexdigest(),
        ),
        (
            "localhost:5000/python:3",
            "localhost:5000/python@sha256:" + hashlib.sha256(b"localhost:5000/python:3").hexdigest(),
        ),
    ],
    ids=["a-docker-short-name", "a-registry-with-a-port"],
)
def test_the_sandbox_starts_from_the_digest_of_the_repository_that_the_task_names(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, image: str, pinned: str
) -> None:
    """Harbor tasks name Docker images, so a short name is a repository on docker.io."""
    env = podman_environment(
        tmp_path, fake_runtime_bin, task_env_config=EnvironmentConfig(docker_image=image)
    )

    assert started(env, sessions).created == (pinned, {"cpu": 2, "memory_mb": 1024})


@pytest.mark.parametrize(
    "image, info",
    [
        ("built-here:latest", {"RepoDigests": []}),
        ("mypython:latest", {"RepoDigests": ["docker.io/library/python@sha256:" + "ef" * 32]}),
    ],
    ids=["a-local-build", "the-digest-of-another-repository"],
)
def test_an_image_without_a_digest_of_its_own_repository_cannot_be_pinned(
    image: str, info: dict[str, Any]
) -> None:
    """A tag can move, so a trial that records a tag does not record what it ran."""
    with pytest.raises(PodmanError, match="no repository digest"):
        digest_reference(image, info)


def test_a_task_without_a_shape_gets_two_cpus_and_1024_mb(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(
        tmp_path, fake_runtime_bin, task_env_config=EnvironmentConfig(docker_image=IMAGE)
    )

    assert started(env, sessions).created == (PINNED, {"cpu": 2, "memory_mb": 1024})


@pytest.mark.parametrize("mode, allowed", [(NetworkMode.NO_NETWORK, False), (NetworkMode.PUBLIC, True)])
def test_the_sandbox_starts_with_egress_only_under_a_public_baseline_policy(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, mode: NetworkMode, allowed: bool
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin, network_policy=NetworkPolicy(network_mode=mode))

    assert started(env, sessions).backend["podman"]["allow_internet"] is allowed


def test_start_lays_out_the_log_directories_as_root_before_anything_else(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    session = started(podman_environment(tmp_path, fake_runtime_bin), sessions)

    assert session.calls[0] == (
        {
            "command": shell_command(LOG_LAYOUT, "/", IMAGE_ENV, "root"),
            "working_dir": "/",
            "timeout": 300,
            "max_output_bytes": 16 * 1024 * 1024,
        },
        360,
    )


def test_the_receipt_in_the_trial_directory_says_what_the_sandbox_is(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """A branch must run on the store that holds its parent's snapshots, so the receipt names the store."""
    policy = NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)

    started(podman_environment(tmp_path, fake_runtime_bin, network_policy=policy), sessions)

    assert json.loads((tmp_path / "trial" / "task__env.podman.json").read_text()) == {
        "backend": "podman",
        "sandbox_id": "sprout-0123456789ab",
        "image": PINNED,
        "image_config": IMAGE_CONFIG,
        "requested_disk_size_mb": 20480,
        "resources": {"cpu": 2, "memory_mb": 8192},
        "network": policy.model_dump(mode="json"),
        "overlay": None,
        "store": str((tmp_path / "store").resolve()),
    }


def test_a_tests_dockerfile_on_the_task_image_is_replayed_as_root_in_the_fresh_sandbox(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """Harbor builds a separate verifier's image from tests/Dockerfile. Sprout repeats its steps instead."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (tests / "Dockerfile").write_text(
        f"FROM {IMAGE}\nCOPY test.sh /tests/test.sh\nRUN chmod +x /tests/test.sh\n"
    )
    env = podman_environment(
        tmp_path,
        fake_runtime_bin,
        environment_dir=tests,
        session_id="task__verifier",
        task_env_config=EnvironmentConfig(cpus=2, memory_mb=8192),
    )

    session = started(env, sessions)

    assert session.created == (PINNED, {"cpu": 2, "memory_mb": 8192})
    assert session.uploads == [("/tests/test.sh", b"#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")]
    assert session.commands()[-1] == shell_command("chmod +x /tests/test.sh", "/testbed", IMAGE_ENV, "root")
    receipt = json.loads((tmp_path / "trial" / "task__verifier.podman.json").read_text())
    assert receipt["overlay"] == {
        "copies": [[str(tests / "test.sh"), "/tests/test.sh"]],
        "runs": ["chmod +x /tests/test.sh"],
    }


def test_the_environment_directory_of_a_prebuilt_image_arrives_in_the_workdir(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """As in Harbor: a task with a docker_image and no Dockerfile gets its environment/ files uploaded."""
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "data.txt").write_text("rows\n")
    config = EnvironmentConfig(docker_image=IMAGE, workdir="/app")

    session = started(podman_environment(tmp_path, fake_runtime_bin, task_env_config=config), sessions)

    ((remote, archive),) = session.uploads
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as stream:
        assert sorted(stream.getnames()) == [".", "./data.txt"]
    assert session.commands()[-2:] == [
        shell_command(f"mkdir -p /app && tar -xzf {remote} -C /app", "/", IMAGE_ENV, "root"),
        shell_command(f"rm -f {remote}", "/", IMAGE_ENV, "root"),
    ]


def test_a_sandbox_that_does_not_start_raises_its_reason_and_closes_its_pool(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    sessions.create_error = "PodmanError: image not known"
    env = podman_environment(tmp_path, fake_runtime_bin)

    with pytest.raises(RuntimeError, match="^PodmanError: image not known$"):
        asyncio.run(env.start())

    (session,) = sessions.made
    assert (session._pool.calls, env.session) == (["close"], None)


def test_start_with_a_build_request_is_refused_before_any_sandbox(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)

    with pytest.raises(ValueError, match="Sprout boots the task's prebuilt image; it does not build one"):
        asyncio.run(env.start(force_build=True))

    assert sessions.made == []


# --- commands -------------------------------------------------------------------------------


@pytest.mark.parametrize("image_user, user", [("", "root"), ("appuser", "appuser")])
def test_a_command_runs_as_the_images_user_in_its_working_directory_with_its_environment(
    tmp_path: Path,
    fake_runtime_bin: Path,
    sessions: Sessions,
    fake_podman: FakePodman,
    image_user: str,
    user: str,
) -> None:
    """An image that names no user runs its commands as root. ``FOO`` keeps the ``=`` in its value."""
    fake_podman.image_user(image_user)
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions)

    assert env.command_args("make test") == {
        "command": shell_command("make test", "/testbed", IMAGE_ENV, user),
        "working_dir": "/",
    }


def test_harbors_own_cwd_env_and_user_win_over_the_images(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions)

    arguments = env.command_args("make", cwd="/work", env={"FOO": "x", "MODE": "test"}, user="nobody")

    assert arguments["command"] == shell_command(
        "make",
        "/work",
        {"PATH": "/usr/local/bin:/usr/bin", "LANG": "C.UTF-8", "FOO": "x", "MODE": "test"},
        "nobody",
    )


def test_a_value_in_harbors_env_reaches_the_command_as_text_and_never_as_a_command(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """The shell line runs here, as root runs it in the sandbox: in a directory whose name has a space."""
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions)
    (tmp_path / "work space").mkdir()
    value = "x'; touch injected; '"

    line = env.command_args(
        'printf %s "$TOKEN"', cwd=str(tmp_path / "work space"), env={"TOKEN": value}, user="root"
    )
    result = subprocess.run(["bash", "-c", line["command"]], capture_output=True, text=True, timeout=30)

    assert (result.returncode, result.stdout, (tmp_path / "work space" / "injected").exists()) == (
        0,
        value,
        False,
    )


def test_the_tasks_workdir_and_harbors_default_user_win_over_the_images(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(
        tmp_path, fake_runtime_bin, task_env_config=EnvironmentConfig(docker_image=IMAGE, workdir="/app")
    )
    started(env, sessions)
    env.default_user = "agent"

    assert env.command_args("make")["command"] == shell_command("make", "/app", IMAGE_ENV, "agent")


@pytest.mark.parametrize(
    "result, answer",
    [
        (
            ToolResult(
                False, "", outcome=CommandOutcome(exit_code=7, stdout='{"stdout": "data"}', stderr="failed")
            ),
            {"stdout": '{"stdout": "data"}', "stderr": "failed", "return_code": 7},
        ),
        (ToolResult(True, "plain output"), {"stdout": "plain output", "stderr": "", "return_code": 0}),
    ],
    ids=["exit-status", "plain-success"],
)
def test_a_command_keeps_its_exit_status_and_its_output(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, result: ToolResult, answer: dict[str, Any]
) -> None:
    """A non-zero exit is an answer for Harbor, and output that looks like JSON is still output."""
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions).answer = lambda arguments: result

    assert asyncio.run(env.exec("pytest")).model_dump() == answer


@pytest.mark.parametrize(
    "result, error, message",
    [
        (
            ToolResult(False, "", outcome=CommandOutcome(timed_out=True)),
            TimeoutError,
            "^Environment command timed out after 50s$",
        ),
        (
            ToolResult(False, "", outcome=CommandOutcome(exit_code=None)),
            RuntimeError,
            "^Environment command has no exit status$",
        ),
        (ToolResult(False, "", error="connection lost"), RuntimeError, "^connection lost$"),
    ],
    ids=["timed-out", "no-exit-status", "transport-failure"],
)
def test_a_command_without_an_exit_status_raises_instead_of_answering(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, result: ToolResult, error: type, message: str
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions).answer = lambda arguments: result

    with pytest.raises(error, match=message):
        asyncio.run(env.exec("pytest", timeout_sec=50))


@pytest.mark.parametrize("timeout_sec, timeouts", [(None, (36000, 36060)), (50, (50, 110))])
def test_a_command_without_a_timeout_of_its_own_gets_the_bound_of_an_eight_hour_actor(
    tmp_path: Path,
    fake_runtime_bin: Path,
    sessions: Sessions,
    timeout_sec: int | None,
    timeouts: tuple[int, int],
) -> None:
    """The session waits a minute longer than the command, so the runtime's own timeout answers first."""
    env = podman_environment(tmp_path, fake_runtime_bin)
    session = started(env, sessions)

    asyncio.run(env.exec("pytest", timeout_sec=timeout_sec))

    arguments, session_timeout = session.calls[-1]
    assert (arguments["timeout"], session_timeout) == timeouts


def test_harbors_output_callback_gets_the_stdout_then_the_stderr_of_a_command(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)
    started(env, sessions).answer = lambda arguments: ToolResult(
        False, "", outcome=CommandOutcome(exit_code=1, stdout="out", stderr="err")
    )
    chunks: list[tuple[str, str]] = []

    async def collect(text: str, stream: str) -> None:
        chunks.append((stream, text))

    async def run() -> None:
        with env.scoped_output_callback(collect):
            await env.exec("pytest")

    asyncio.run(run())

    assert chunks == [("stdout", "out"), ("stderr", "err")]


def test_download_dir_copies_a_sandbox_directory_into_the_target_and_removes_the_archive(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    """Harbor reads the verifier's reward this way."""
    env = podman_environment(tmp_path, fake_runtime_bin)
    session = started(env, sessions)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as stream:
        info = tarfile.TarInfo("reward.txt")
        info.size = 2
        stream.addfile(info, io.BytesIO(b"1\n"))
    session.download = archive.getvalue()

    asyncio.run(env.download_dir("/logs/verifier", tmp_path / "verifier"))

    assert (tmp_path / "verifier" / "reward.txt").read_text() == "1\n"
    (remote,) = session.downloads
    assert session.commands()[-2:] == [
        shell_command(f"tar -czf {remote} -C /logs/verifier .", "/", IMAGE_ENV, "root"),
        shell_command(f"rm -f {remote}", "/", IMAGE_ENV, "root"),
    ]


# --- the network policy ---------------------------------------------------------------------


def test_harbors_phase_policies_switch_the_egress_of_the_running_sandbox(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    phases = [NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)]
    env = podman_environment(tmp_path, fake_runtime_bin, phase_network_policies=phases)
    session = started(env, sessions)

    asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)))
    asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC)))

    assert session.network == [False, True]


def test_a_network_policy_before_the_sandbox_starts_is_refused(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)

    with pytest.raises(RuntimeError, match="^Cannot update network policy before sandbox start$"):
        asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)))


# --- stop -----------------------------------------------------------------------------------


@pytest.mark.parametrize("delete, calls", [(True, ["destroy", "close"]), (False, ["forget", "close"])])
def test_stop_destroys_the_sandbox_unless_harbor_keeps_it_and_then_closes_the_pool(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, delete: bool, calls: list[str]
) -> None:
    """Closing a pool destroys each sandbox that it holds, so a kept sandbox leaves the pool first."""
    env = podman_environment(tmp_path, fake_runtime_bin)
    session = started(env, sessions)

    asyncio.run(env.stop(delete=delete))

    assert (session._pool.calls, env.session) == (calls, None)


def test_a_second_stop_does_nothing(tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions) -> None:
    env = podman_environment(tmp_path, fake_runtime_bin)
    session = started(env, sessions)
    asyncio.run(env.stop(delete=True))

    asyncio.run(env.stop(delete=True))

    assert session._pool.calls == ["destroy", "close"]


def test_a_destroy_that_fails_still_closes_the_pool_and_raises_what_is_left(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions
) -> None:
    sessions.fail_destroy = True
    env = podman_environment(tmp_path, fake_runtime_bin)
    session = started(env, sessions)

    with pytest.raises(RuntimeError, match="^the container is gone but its rootfs is not$"):
        asyncio.run(env.stop(delete=True))

    assert (session._pool.calls, env.session) == (["destroy", "close"], None)


# --- admission ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, message",
    [
        ("ENV X=1", "ENV is not replayed in a sandbox"),
        ("WORKDIR /tests", "WORKDIR is not replayed in a sandbox"),
        ("USER nobody", "USER is not replayed in a sandbox"),
        ("COPY --chown=1 test.sh /tests/test.sh", "only `COPY <src> <dst>` is replayed"),
        ("FROM other:1", "multi-stage Dockerfiles are not replayed"),
    ],
    ids=["env", "workdir", "user", "copy-with-options", "a-second-stage"],
)
def test_an_overlay_step_that_a_sandbox_cannot_repeat_exactly_is_refused_before_any_sandbox(
    tmp_path: Path, fake_runtime_bin: Path, sessions: Sessions, line: str, message: str
) -> None:
    """A replayed environment must never differ from the one that the task's author built."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("exit 0\n")
    (tests / "Dockerfile").write_text(f"FROM {IMAGE}\nCOPY test.sh /tests/test.sh\n{line}\n")

    with pytest.raises(ValueError, match=message):
        podman_environment(tmp_path, fake_runtime_bin, environment_dir=tests)

    assert sessions.made == []


def compose(tmp_path: Path) -> dict[str, Any]:
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "docker-compose.yaml").write_text("services: {}\n")
    return {}


def dockerfile(text: str, *, docker_image: str | None) -> Callable[[Path], dict[str, Any]]:
    """A task whose environment/Dockerfile is ``text``, with ``docker_image``."""

    def make(tmp_path: Path) -> dict[str, Any]:
        (tmp_path / "environment").mkdir()
        (tmp_path / "environment" / "Dockerfile").write_text(text)
        return {"task_env_config": EnvironmentConfig(docker_image=docker_image)}

    return make


@pytest.mark.parametrize(
    "setup, error, message",
    [
        (lambda tmp_path: {"store": str(tmp_path / "nowhere")}, ValueError, "^podman store does not exist: "),
        (
            lambda tmp_path: {"runtime_bin": str(tmp_path / "absent-runtime")},
            ValueError,
            "^Missing sprout-runtime: ",
        ),
        (compose, ValueError, "^podman multi-service composition is not implemented; no Docker fallback$"),
        (
            lambda tmp_path: {
                "task_env_config": EnvironmentConfig(
                    docker_image=IMAGE,
                    mcp_servers=[MCPServerConfig(name="tools", url="http://tools.test/sse")],
                )
            },
            ValueError,
            "^Task-side MCP networking requires a podman service adapter$",
        ),
        (
            lambda tmp_path: {"task_env_config": EnvironmentConfig()},
            ValueError,
            "^Task needs a docker_image or environment/Dockerfile$",
        ),
        (
            dockerfile("FROM base:latest\nWORKDIR /app\nRUN make\n", docker_image=None),
            ValueError,
            "^A task without a docker_image needs a Dockerfile a sandbox can replay; Sprout boots prebuilt",
        ),
        (
            lambda tmp_path: {"mounts": [{"type": "bind", "source": "/path/to/data", "target": "/data"}]},
            ValueError,
            "^podman cannot reproduce host mount /data$",
        ),
        (
            lambda tmp_path: {"task_env_config": EnvironmentConfig(docker_image=IMAGE, gpus=1)},
            RuntimeError,
            "GPU",
        ),
        (
            lambda tmp_path: {
                "network_policy": NetworkPolicy(
                    network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"]
                )
            },
            ValueError,
            "allowlist",
        ),
    ],
    ids=[
        "no-store",
        "no-runtime",
        "compose",
        "mcp-servers",
        "no-image",
        "a-recipe-that-needs-a-build",
        "another-mount",
        "a-gpu",
        "an-allowlist",
    ],
)
def test_a_task_that_the_sandbox_cannot_reproduce_is_refused_before_any_sandbox(
    tmp_path: Path,
    fake_runtime_bin: Path,
    sessions: Sessions,
    setup: Callable[[Path], dict[str, Any]],
    error: type,
    message: str,
) -> None:
    """The environment never falls back to Docker, and never builds an image."""
    changes = setup(tmp_path)

    with pytest.raises(error, match=message):
        podman_environment(tmp_path, fake_runtime_bin, **changes)

    assert sessions.made == []
