"""PodmanEnvironment: Harbor's environment over the podman backend, with a fake session.

What differs from the microVM environment is checked here -- the backend
section, the shape without a disk, the image pinned through podman rather than
regctl, egress toggled through the session, the store on the receipt -- and
what must not differ: the overlay replay, the retained stop, and that the
agent classes and the branch environment accept it unchanged.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.agent.context import AgentContext
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from sprout.harbor_plugin import environment as sandbox_environment
from sprout.harbor_plugin import podman as podman_environment
from sprout.harbor_plugin.agentenv_agent import AgentENVOrchestrator
from sprout.harbor_plugin.agentenv_mini import AgentENVMini
from sprout.harbor_plugin.branching import PodmanSnapshotEnvironment, read_branch_context
from sprout.harbor_plugin.environment import SandboxEnvironment
from sprout.harbor_plugin.podman import PodmanEnvironment
from sprout.harness.core.journal import JournalWriter
from sprout.harness.core.result import ToolResult
from sprout.harness.orchestrator.run import RunOutcome
from sprout_sandbox.podman import PodmanError, digest_reference

IMAGE = "public.ecr.aws/org/task:v1"
INFO = {"Id": "ab" * 32,
        "RepoDigests": ["public.ecr.aws/org/task@sha256:" + "cd" * 32],
        "Config": {"Env": ["PATH=/root/go/bin:/usr/bin", "GOFLAGS=-mod=mod"], "WorkingDir": "/app",
                   "User": "", "Entrypoint": None, "Cmd": ["bash"], "Labels": {"ignored": "1"}}}


def environment(tmp_path: Path, **changes) -> PodmanEnvironment:
    directory = changes.pop("environment_dir", tmp_path / "environment")
    directory.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "runtime"
    runtime.touch()
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    config = changes.pop("task_env_config", EnvironmentConfig(docker_image=IMAGE, cpus=2, memory_mb=8192,
                                                              storage_mb=20480))
    return PodmanEnvironment(environment_dir=directory, environment_name="task",
                             session_id=changes.pop("session_id", "t__env"),
                             task_env_config=config, trial_paths=TrialPaths(trial_dir=tmp_path),
                             runtime_bin=str(runtime), store=changes.pop("store", str(store)), **changes)


class FakePool:
    """What stop() asks of the session's pool, in order: the order is the point."""

    def __init__(self):
        self.calls: list = []
        self.fail_destroy = False

    async def destroy(self, sandbox):
        self.calls.append("destroy")
        if self.fail_destroy:
            raise RuntimeError("container removed but its rootfs is not")

    async def forget(self, sandbox):
        self.calls.append("forget")

    async def close(self):
        self.calls.append("close")


class FakeSession:
    """What the environment asks of a SandboxSession, recorded.

    ``_destroy_async`` / ``_release_async`` are the real session's four lines
    each: the pool's ``destroy`` or ``forget`` on the sandbox.
    """

    instances: list = []

    def __init__(self, **kwargs):
        self.backend = kwargs["backend"]
        self.sandbox_id = "sprout-0123456789ab"
        self.created = None
        self.commands: list = []
        self.uploads: list = []
        self.network: list = []
        self._pool = FakePool()
        self._sandbox = object()
        self.on_swap: list = []
        FakeSession.instances.append(self)

    def _drive(self, coroutine):
        return asyncio.run(coroutine)

    async def _destroy_async(self):
        await self._pool.destroy(self._sandbox)

    async def _release_async(self):
        await self._pool.forget(self._sandbox)

    def create(self, image, resources):
        self.created = (image, resources)
        return True

    def execute(self, name, arguments, timeout):
        self.commands.append(arguments["command"])
        return ToolResult(True, "")

    def upload_file(self, source, destination):
        self.uploads.append((Path(source).name, destination))
        return True

    def set_network(self, allow):
        self.network.append(allow)


@pytest.fixture
def fake_session(monkeypatch):
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "podman")   # the default podman is `podman` on PATH here
    FakeSession.instances = []
    monkeypatch.setattr(sandbox_environment, "SandboxSession", FakeSession)

    async def info(image, *, podman_bin="podman", cwd=None):
        assert podman_bin == "podman"
        return INFO

    monkeypatch.setattr(podman_environment, "image_info", info)
    return FakeSession


def test_start_creates_a_shaped_offline_container_from_the_pinned_image(tmp_path, fake_session):
    env = environment(tmp_path, network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    asyncio.run(env.start())
    [session] = fake_session.instances
    store = str((tmp_path / "store").resolve())
    assert session.backend == {"backend": "podman", "podman": {
        "store": store, "runtime_bin": str((tmp_path / "runtime").resolve()),
        "podman_bin": "podman", "allow_internet": False}}
    # Pinned through podman's own record, no regctl; no disk in the shape (a
    # subvolume has none to size).
    assert session.created == ("public.ecr.aws/org/task@sha256:" + "cd" * 32, {"cpu": 2, "memory_mb": 8192})
    assert env.image_workdir == "/app" and env.image_user == "root"
    assert env.image_env == {"PATH": "/root/go/bin:/usr/bin", "GOFLAGS": "-mod=mod"}
    assert env.shell_context() == {"name": "shell_context", "cwd": "/app", "env": env.image_env, "user": "root"}
    assert any("mkdir -p /logs/agent" in command for command in session.commands)
    record = json.loads((tmp_path / "t__env.podman.json").read_text())
    assert record["backend"] == "podman" and record["store"] == store
    assert record["sandbox_id"] == "sprout-0123456789ab" and record["image"] == env.image
    assert record["image_config"] == {"Env": INFO["Config"]["Env"], "WorkingDir": "/app", "User": "",
                                      "Entrypoint": None, "Cmd": ["bash"]}
    assert record["resources"] == {"cpu": 2, "memory_mb": 8192} and record["requested_disk_size_mb"] == 20480
    assert record["network"]["network_mode"] == "no-network" and record["overlay"] is None


def test_a_public_task_starts_connected_and_phases_toggle_through_the_session(tmp_path, fake_session):
    env = environment(tmp_path, phase_network_policies=[NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)])
    assert env.capabilities.dynamic_network_policy and env.capabilities.disable_internet
    asyncio.run(env.start())
    [session] = fake_session.instances
    assert session.backend["podman"]["allow_internet"] is True
    asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK)))
    asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC)))
    assert session.network == [False, True]
    with pytest.raises(ValueError):
        asyncio.run(env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST,
                                                         allowed_hosts=["example.com"])))


def test_the_verifier_overlay_is_replayed_into_the_fresh_container(tmp_path, fake_session):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (tests / "Dockerfile").write_text(f"FROM {IMAGE}\nCOPY test.sh /tests/test.sh\nRUN chmod +x /tests/test.sh\n")
    env = environment(tmp_path, environment_dir=tests, session_id="t__verifier__grade",
                      task_env_config=EnvironmentConfig(cpus=2, memory_mb=8192))
    assert env.overlay is not None and env.overlay.base == IMAGE
    asyncio.run(env.start())
    [session] = fake_session.instances
    assert session.uploads == [("test.sh", "/tests/test.sh")]
    assert any("chmod +x /tests/test.sh" in command and "cd /app" in command for command in session.commands)
    record = json.loads((tmp_path / "t__verifier__grade.podman.json").read_text())
    assert record["overlay"]["runs"] == ["chmod +x /tests/test.sh"]


def test_stop_destroys_unless_the_sandbox_is_retained(tmp_path, fake_session):
    """Closing a pool destroys every sandbox it still holds, so a kept sandbox
    is released from the pool first -- retained, or kept by Harbor's own
    delete=False -- and never destroyed; an ordinary stop destroys, then closes."""
    env = environment(tmp_path)
    asyncio.run(env.start())
    [session] = fake_session.instances
    env.retain_reason = "snapshot_pending"
    asyncio.run(env.stop(delete=True))
    assert session._pool.calls == ["forget", "close"], "the container holds the only copy of the rollout"
    assert json.loads((tmp_path / "t__env.retained-sandbox.json").read_text()) == {
        "sandbox_id": "sprout-0123456789ab", "reason": "snapshot_pending"}
    assert env.session is None
    env = environment(tmp_path)
    asyncio.run(env.start())
    session = fake_session.instances[-1]
    asyncio.run(env.stop(delete=True))
    assert session._pool.calls == ["destroy", "close"], "no reason to retain: destroyed, then the pool closed"
    asyncio.run(env.stop(delete=True))
    assert session._pool.calls == ["destroy", "close"], "a second stop has nothing left to do"
    env = environment(tmp_path)
    asyncio.run(env.start())
    session = fake_session.instances[-1]
    asyncio.run(env.stop(delete=False))
    assert session._pool.calls == ["forget", "close"], "Harbor's own keep: released, so the pool's close leaves it"


def test_a_destroy_that_fails_still_closes_the_pool_and_reports(tmp_path, fake_session):
    env = environment(tmp_path)
    asyncio.run(env.start())
    [session] = fake_session.instances
    session._pool.fail_destroy = True
    with pytest.raises(RuntimeError, match="rootfs is not"):
        asyncio.run(env.stop(delete=True))
    assert session._pool.calls == ["destroy", "close"] and env.session is None, \
        "the pool is closed and the environment finished either way; the error names the leftover"


def test_a_missing_store_or_a_build_request_is_refused(tmp_path, fake_session):
    with pytest.raises(ValueError, match="store does not exist"):
        environment(tmp_path, store=str(tmp_path / "nowhere"))
    env = environment(tmp_path)
    with pytest.raises(ValueError, match="does not build"):
        asyncio.run(env._prepare_image(True))


def test_the_pinned_reference_is_the_digest_podman_pulled():
    assert digest_reference(IMAGE, INFO) == "public.ecr.aws/org/task@sha256:" + "cd" * 32
    short = {"RepoDigests": ["docker.io/library/python@sha256:" + "ef" * 32]}
    assert digest_reference("python:3.11-slim", short) == "docker.io/library/python@sha256:" + "ef" * 32
    assert digest_reference("docker.io/library/python:3.11-slim", short) == short["RepoDigests"][0]
    assert digest_reference("localhost:5000/python:3", {"RepoDigests": ["localhost:5000/python@sha256:" + "ab" * 32]}) \
        == "localhost:5000/python@sha256:" + "ab" * 32
    with pytest.raises(PodmanError, match="no repository digest"):
        digest_reference("mypython:latest", short)
    with pytest.raises(PodmanError, match="no repository digest"):
        digest_reference("built-here:latest", {"RepoDigests": []})


def test_the_mini_agent_runs_on_a_podman_environment_unchanged(tmp_path, fake_session, monkeypatch):
    """The agent classes read only SandboxEnvironment's surface, so a podman
    trial runs the same actor with the same spec, and the record says which
    backend it ran on."""
    env = environment(tmp_path, task_env_config=EnvironmentConfig(docker_image=IMAGE, workdir="/app"))
    asyncio.run(env.start())
    assert isinstance(env, SandboxEnvironment)
    seen = []

    def run(self, spec):
        seen.append(spec)
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed",
                          final_snapshot_id="sprout-0123456789ab-0007")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_KEY", "test-only")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                         inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_KEY")
    asyncio.run(agent.setup(env))
    context = AgentContext()
    asyncio.run(agent.run("Solve it.", env, context))
    assert seen[0].backend == env.backend and seen[0].backend["backend"] == "podman"
    assert seen[0].sandbox_image == env.image
    assert seen[0].interceptors == [env.shell_context()]
    assert context.metadata["backend"] == "podman"
    assert context.metadata["final_snapshot_id"] == "sprout-0123456789ab-0007"
    with pytest.raises(TypeError, match="SandboxEnvironment"):
        asyncio.run(agent.setup(SimpleNamespace(session=object())))


def test_a_podman_branch_restores_the_subvolume_snapshot_by_id(tmp_path, fake_session):
    parent = tmp_path / "parent.jsonl"
    parent.write_text('{"type":"run.started","slot":"mini-swe-agent"}\n')
    (tmp_path / "task" / "environment").mkdir(parents=True)
    (tmp_path / "task" / "tests").mkdir(parents=True)
    context = {"task_dir": str(tmp_path / "task"), "branch_guidance": "none", "checkpoint_mode": "disk_only",
               "snapshot_id": "sprout-0123456789ab-0003", "base_image": "public.ecr.aws/org/task@sha256:" + "cd" * 32,
               "image_config": {"User": "", "WorkingDir": "/app", "Env": ["GOFLAGS=-mod=mod"]},
               "parent_journal": str(parent), "parent_journal_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
               "step": 3, "native_session_id": "session", "conversation_cut": "turn"}
    path = tmp_path / "branch.json"
    path.write_text(json.dumps(context))
    assert read_branch_context(path)["snapshot_id"] == "sprout-0123456789ab-0003"
    for bad in ("../escape", "a/b", "", ".hidden"):
        path.write_text(json.dumps({**context, "snapshot_id": bad}))
        with pytest.raises(ValueError, match="snapshot id"):
            read_branch_context(path)
    path.write_text(json.dumps(context))
    runtime = tmp_path / "runtime"
    runtime.touch()
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    branch = PodmanSnapshotEnvironment(
        environment_dir=tmp_path / "task" / "environment", environment_name="task", session_id="t-r1b1__env",
        task_env_config=EnvironmentConfig(docker_image=IMAGE, cpus=2, memory_mb=8192),
        trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(runtime), store=str(store),
        branch_context=str(path))
    assert branch.restores_snapshot is True
    asyncio.run(branch.start())
    [session] = fake_session.instances
    # The snapshot id is the image; the shape is passed again (a subvolume carries none).
    assert session.created == ("sprout-0123456789ab-0003", {"cpu": 2, "memory_mb": 8192})
    assert session.uploads == [], "restored files are already in the snapshot"
    assert branch.image_workdir == "/app" and branch.image_env == {"GOFLAGS": "-mod=mod"}
    receipt = json.loads((tmp_path / "restored-state.json").read_text())
    assert receipt["snapshot_id"] == "sprout-0123456789ab-0003" and receipt["sandbox_id"] == "sprout-0123456789ab"
    verifier = PodmanSnapshotEnvironment(
        environment_dir=tmp_path / "task" / "tests", environment_name="task", session_id="t-r1b1__verifier__grade",
        task_env_config=EnvironmentConfig(docker_image=IMAGE, cpus=2, memory_mb=8192),
        trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(runtime), store=str(store),
        branch_context=str(path))
    assert verifier.restores_snapshot is False
    assert asyncio.run(verifier._prepare_image(False))[0] == "public.ecr.aws/org/task@sha256:" + "cd" * 32
