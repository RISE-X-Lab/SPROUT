from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from sprout_sandbox.pool import Snapshot
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from harbor.models.agent.context import AgentContext
from sprout.harness.core.journal import JournalWriter, read_journal
from sprout.harness.core.result import CommandOutcome, ToolResult
from sprout.harness.orchestrator.run import OwnedSandbox, RunOutcome, RunSpec
from sprout.harness.tools.interceptors import shell_command
from sprout.harbor_plugin import environment as sandbox_environment
from sprout.harbor_plugin.agentenv import AgentENVEnvironment
from sprout.harbor_plugin.agentenv_agent import AgentENVOrchestrator
from sprout.harbor_plugin.agentenv_mini import AgentENVMini
from sprout.harness.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.harbor_plugin.environment import overlay_for


def environment(tmp_path: Path, **changes) -> AgentENVEnvironment:
    directory = tmp_path / "environment"
    directory.mkdir(exist_ok=True)
    runtime = tmp_path / "runtime"
    runtime.touch()
    config = changes.pop("task_env_config", EnvironmentConfig(docker_image="image:tag", cpus=2, memory_mb=1024))
    return AgentENVEnvironment(environment_dir=directory, environment_name="task", session_id="test",
                               task_env_config=config, trial_paths=TrialPaths(trial_dir=tmp_path),
                               runtime_bin=str(runtime), **changes)


@pytest.mark.parametrize("change,files,error,match", [
    ({"task_env_config": EnvironmentConfig(docker_image="image", gpus=1)}, {}, RuntimeError, "GPU"),
    ({"network_policy": NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["example.com"])}, {},
     ValueError, "allowlist"),
    ({"task_env_config": EnvironmentConfig(docker_image="image", storage_mb=1536)}, {}, ValueError, "divisible by 1024"),
    # a compose task would otherwise boot its docker_image alone and silently drop the other services
    ({}, {"docker-compose.yaml": "services: {}"}, ValueError, "multi-service"),
])
def test_unsupported_requirements_fail_before_start(tmp_path, change, files, error, match):
    directory = tmp_path / "environment"
    directory.mkdir()
    for name, text in files.items():
        (directory / name).write_text(text)
    with pytest.raises(error, match=match):
        environment(tmp_path, **change)


def test_the_server_is_the_one_the_memory_snapshot_release_assumes(tmp_path, monkeypatch):
    """methods.evaluate releases a trial's memory snapshot through
    AgentEnvClient; named nowhere, the two once defaulted to different ports."""
    from sprout.harness.orchestrator.reap import AgentEnvClient

    monkeypatch.delenv("AENV_SERVER_URL", raising=False)
    assert environment(tmp_path).server_url == AgentEnvClient().base_url == "http://127.0.0.1:8000"
    monkeypatch.setenv("AENV_SERVER_URL", "http://bench:18000")
    assert environment(tmp_path).server_url == AgentEnvClient().base_url == "http://bench:18000"
    assert environment(tmp_path, server_url="http://named:1").server_url == "http://named:1"


def test_shell_quotes_arguments_and_honors_user_environment():
    wrapped = shell_command("printf '%s' \"$TOKEN\"", "/work space", {"TOKEN": "x'; touch /wrong; '"}, "123")
    assert "runuser" in wrapped and "getent passwd 123" in wrapped
    assert "cd '/work space'" in wrapped.replace("'\"'\"'", "'")
    assert "TOKEN=" in wrapped


def test_exec_preserves_exit_status_and_does_not_confuse_json_stdout(tmp_path):
    env = environment(tmp_path)
    calls = []

    def execute(name, arguments, timeout):
        calls.append((name, arguments, timeout))
        return ToolResult(success=False, output="", outcome=CommandOutcome(exit_code=7, stdout='{"stdout":"data"}', stderr="failed"))

    env.session = SimpleNamespace(execute=execute)
    env.default_user = "nobody"
    result = asyncio.run(env.exec("test command", cwd="/work", env={"MODE": "test"}, timeout_sec=50))
    assert result.return_code == 7 and result.stdout == '{"stdout":"data"}'
    assert "nobody" in calls[0][1]["command"]
    assert calls[0][1]["timeout"] == 50 and calls[0][2] == 110


def test_transport_failure_is_not_an_exit_code(tmp_path):
    env = environment(tmp_path)
    env.session = SimpleNamespace(execute=lambda *args: ToolResult(False, "", error="connection lost"))
    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(env.exec("test"))


def test_the_tools_run_through_the_task_context_and_each_turn_is_captured(tmp_path):
    """The plugin's tool access: the task's cwd/user wrap every command, a turn
    is snapshotted disk-only when the next begins, and the last by the one FULL
    final snapshot."""
    env = environment(tmp_path, checkpoint_mode="disk_only",
                      task_env_config=EnvironmentConfig(docker_image="image", workdir="/app"))
    ran, captured = [], []

    class Session:
        sandbox_id = "sandbox"

        def supports_snapshot(self):
            return True

        def call(self, name, args, timeout=300):
            ran.append(args["command"])
            return ToolResult(True, "ok")

        def snapshot(self, disk_only=True, name=None):
            captured.append((len(ran), disk_only))
            return Snapshot("snapshot-%d" % len(captured))

    session = Session()
    orchestrator = AgentENVOrchestrator(env, {}, out_dir=tmp_path)
    spec = RunSpec(prompt="x", interceptors=[env.shell_context()])
    with JournalWriter(tmp_path / "trajectory.jsonl") as journal:
        tools = orchestrator._tool_access(spec, journal, OwnedSandbox(session=session, sandbox_id="sandbox"), None)
        journal.emit("session.ref", native_session_id="native-test")
        for number in range(2):
            turn_id, call_id = "t%d" % number, "call-%d" % number
            tools.call("shell", {"command": "write-%d" % number, "timeout": 5},
                       call_id=call_id, turn_id=turn_id)
            journal.emit("model.turn.completed", turn_id=turn_id, call_ids=[call_id], step=number + 1)
            tools.close_turn(turn_id)
        final = tools.finalize()
    assert captured == [(1, True), (2, True)], "the final snapshot is disk-only by default"
    assert final["snapshot_id"] == "snapshot-2"
    assert all("/app" in command for command in ran), ran
    records = read_journal(tmp_path / "trajectory.jsonl")
    points = [row for row in records if row["type"] == "checkpoint.captured"]
    assert [(row["turn_id"], row["session_ckpt"]) for row in points] == [
        ("t0", "native-test"), ("t1", "native-test")]


@pytest.mark.parametrize("requested_disk,effective_disk", [
    (None, 65536), (8192, 65536), (10240, 65536), (65536, 65536), (131072, 131072),
])
def test_cpu_memory_disk_are_sent_through_template_and_cold_start(tmp_path, monkeypatch,
                                                                requested_disk, effective_disk):
    env = environment(tmp_path, task_env_config=EnvironmentConfig(
        docker_image="image:tag", cpus=3, memory_mb=2048, storage_mb=requested_disk))
    calls = []

    class FakeSession:
        sandbox_id = "sandbox"

        def __init__(self, **kwargs):
            self.backend = kwargs["backend"]

        def create(self, image, resources):
            calls.append((image, resources))
            return True

        def execute(self, *args):
            return ToolResult(True, "")

    monkeypatch.setattr(sandbox_environment, "SandboxSession", FakeSession)

    async def prepared(force):
        return "image@sha256:digest", {"WorkingDir": "/work", "User": "nobody"}

    async def no_upload():
        pass

    monkeypatch.setattr(env, "_prepare_image", prepared)

    monkeypatch.setattr(env, "_upload_environment_dir_after_start", no_upload)
    asyncio.run(env.start())
    assert calls == [("image@sha256:digest", {"cpu": 3, "memory_mb": 2048, "disk_size_mb": effective_disk})]
    record = json.loads((tmp_path / "test.agentenv.json").read_text())
    assert record["requested_disk_size_mb"] == requested_disk
    assert record["resources"] == calls[0][1]
    assert record["backend"] == "microvm" and record["image"] == "image@sha256:digest"
    assert record["image_config"] == {"WorkingDir": "/work", "User": "nobody"}, "a branch reads the config from here"
    assert env.task_env_config.storage_mb == requested_disk
    assert env.image_user == "nobody" and env.image_workdir == "/work"


def harbor_session():
    return SimpleNamespace(on_swap=[], sandbox_id="sandbox", _pool=None)


@pytest.mark.parametrize("paired", [True, False])
def test_actor_requires_a_snapshot_for_each_closed_turn(tmp_path, monkeypatch, paired):
    env = environment(tmp_path)
    env.session = harbor_session()
    original_session = env.session

    def run(self, spec):
        assert spec.slot == "mini-swe-agent"
        assert spec.extra["rollout_contract"]["model"] == "test"
        assert spec.extra["rollout_contract"]["api_key_env"] == "EVAL_BRIDGE_KEY"
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.started", slot="test", task_prompt=spec.prompt)
            journal.emit("tool.started", call_id="call", turn_id="t1", name="shell", args={"command": "write"})
            journal.emit("tool.finished", call_id="call", turn_id="t1", name="shell", status="ok", output="")
            journal.emit("model.turn.completed", turn_id="t1", call_ids=["call"], step=1)
            if paired:
                journal.emit("checkpoint.captured", turn_id="t1", step=1, snapshot_id="final",
                             reason="captured", final=True)
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed",
                          final_snapshot_id="final")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "secret-value")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252",
                        api_key_env="EVAL_BRIDGE_KEY")
    context = AgentContext()
    if paired:
        asyncio.run(agent.run("task", env, context))
        assert context.metadata["final_snapshot_id"] == "final"
    else:
        with pytest.raises(RuntimeError, match="lack snapshots"):
            asyncio.run(agent.run("task", env, context))
    assert env.session is original_session
    assert (tmp_path / "agent/trajectory.json").exists()


def test_the_command_timeout_reaches_every_command_and_is_validated(tmp_path, monkeypatch):
    """The model's bash tool has no timeout argument: the one limit every command runs
    under is the agent's command_timeout_s, handed to mini as environment.timeout."""
    env = environment(tmp_path)
    env.session = harbor_session()
    seen = {}

    def run(self, spec):
        seen["timeout"] = spec.extra["mini"]["environment"]["timeout"]
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.started", slot="mini-swe-agent", task_prompt=spec.prompt)
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed",
                          final_snapshot_id="final")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "secret-value")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY",
                        command_timeout_s=240)
    asyncio.run(agent.run("task", env, AgentContext()))
    assert seen["timeout"] == 240
    default = AgentENVMini(logs_dir=tmp_path / "agent2", model_name="test",
                          inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY")
    assert default.command_timeout_s == DEFAULT_COMMAND_TIMEOUT_S == 600, "the one default limit"
    for bad in (0, -5, 60.0, "60"):
        with pytest.raises(ValueError, match="command_timeout_s"):
            AgentENVMini(logs_dir=tmp_path / "agent3", model_name="test",
                         inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY",
                         command_timeout_s=bad)


def test_harbor_deadline_finalizes_snapshot_and_usage_before_verification(tmp_path, monkeypatch):
    import time
    from sprout.harness.core.control import RunControl

    env = environment(tmp_path)
    env.session = harbor_session()

    def run(self, spec):
        control = RunControl()
        env.actor_control = control
        while not control.reason:
            time.sleep(0.002)
        assert control.stop_reason == "timeout"
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.started", slot="mini-swe-agent", task_prompt=spec.prompt)
            journal.emit("run.finished", status="timeout", usage={"input_tokens": 7, "output_tokens": 3})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="timeout",
                          usage={"input_tokens": 7, "output_tokens": 3}, final_snapshot_id="final")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "test-only")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY",
                        actor_timeout_s=0.02)
    context = AgentContext()
    async def trial():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(agent.run("task", env, context), timeout=0.05)
    asyncio.run(trial())
    assert context.n_input_tokens == 7 and context.n_output_tokens == 3
    assert context.metadata["final_snapshot_id"] == "final"
    assert json.loads((tmp_path / "agent/execution.json").read_text())["status"] == "timeout"
    assert (tmp_path / "agent/snapshot.json").exists()


def test_mini_prompt_is_task_plus_the_pinned_mini_tool_primer(tmp_path, monkeypatch):
    from sprout.harness.agents.mini_swe.tools import mini_tool_primer

    env = environment(tmp_path, task_env_config=EnvironmentConfig(docker_image="image", workdir="/app"))
    env.session = harbor_session()
    seen = []

    def run(self, spec):
        seen.append(spec)
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed",
                          final_snapshot_id="final")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "test-only")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY")
    asyncio.run(agent.run("Solve the task.", env, AgentContext()))
    assert seen[0].prompt == "Solve the task.\n\n" + mini_tool_primer("/app")
    assert seen[0].extra["prompt_primer"] == "mini_tool_primer"
    # The task's cwd/env/user reach every command through a declared interceptor,
    # not an orchestrator override.
    assert seen[0].interceptors == [{"name": "shell_context", "cwd": "/app", "env": {}, "user": "root"}]
    assert seen[0].final_memory_snapshot is False, "memory is the benchmark's call, off by default"
    # No wording written for other agents' tools reaches mini.
    assert "MCP" not in seen[0].prompt and "built-in" not in seen[0].prompt


def test_the_memory_snapshot_request_reaches_the_run_and_its_id_the_trial_record(tmp_path, monkeypatch):
    env = environment(tmp_path)
    env.session = harbor_session()
    seen = []

    def run(self, spec):
        seen.append(spec)
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="completed",
                          final_snapshot_id="disk", final_memory_snapshot_id="memory")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "test-only")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY",
                        final_memory_snapshot=True)
    context = AgentContext()
    asyncio.run(agent.run("task", env, context))
    assert seen[0].final_memory_snapshot is True
    assert context.metadata["final_snapshot_id"] == "disk"
    assert context.metadata["final_memory_snapshot_id"] == "memory"
    assert json.loads((tmp_path / "agent/snapshot.json").read_text())["final_memory_snapshot_id"] == "memory"


def test_a_failed_final_snapshot_retains_the_sandbox_instead_of_destroying_it(tmp_path, monkeypatch):
    """The final full snapshot is the only durable copy of the rollout. Until it
    exists the VM must survive Harbor's teardown."""
    env = environment(tmp_path)
    env.session = harbor_session()

    class RecordingPool:
        calls: list = []

        async def destroy(self, sandbox):
            self.calls.append("destroy")

        async def forget(self, sandbox):
            self.calls.append("forget")

        async def close(self):
            self.calls.append("close")

    pool = RecordingPool()
    env.session._pool = pool
    env.session._drive = asyncio.run
    env.session._destroy_async = lambda: pool.destroy("sandbox")
    env.session._release_async = lambda: pool.forget("sandbox")

    def run(self, spec):
        with JournalWriter(spec.journal_path) as journal:
            journal.emit("run.finished", status="completed", usage={})
        return RunOutcome(run_id="test", journal_path=spec.journal_path, status="error",
                          error="Final snapshot failed (snapshot failed); sandbox sandbox retained")

    monkeypatch.setattr(AgentENVOrchestrator, "run", run)
    monkeypatch.setenv("EVAL_BRIDGE_KEY", "test-only")
    agent = AgentENVMini(logs_dir=tmp_path / "agent", model_name="test",
                        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_BRIDGE_KEY")
    with pytest.raises(RuntimeError, match="retained"):
        asyncio.run(agent.run("task", env, AgentContext()))
    assert env.retain_reason == "snapshot_pending"
    asyncio.run(env.stop(delete=True))
    assert pool.calls == ["forget", "close"], \
        "Harbor asked for deletion; the VM must stay: released from the pool before the pool closes"
    receipt = json.loads((tmp_path / "test.retained-sandbox.json").read_text())
    assert receipt == {"sandbox_id": "sandbox", "reason": "snapshot_pending"}
    assert env.session is None


# --- tests/Dockerfile as an overlay on the task image ------------------------
def write_overlay(directory: Path, base: str = "image:tag", extra: str = "") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (directory / "config.json").write_text("{}")
    (directory / "Dockerfile").write_text(
        f"# verifier image\nFROM {base}\n\nCOPY test.sh /tests/test.sh\n"
        f"COPY config.json /tests/config.json\nRUN chmod +x /tests/test.sh\n{extra}")
    return directory


RECIPE = "FROM base:latest\nWORKDIR /app\nRUN git clone x .\n"   # DeepSWE's environment/Dockerfile


@pytest.mark.parametrize("dockerfile,docker_image,expected", [
    # None: write_overlay's own tests/Dockerfile, FROM the task image
    (None, "image:tag", ("image:tag", [("test.sh", "/tests/test.sh"), ("config.json", "/tests/config.json")],
                         ("chmod +x /tests/test.sh",))),
    (RECIPE, "image:tag", None),
    (RECIPE, None, None),
])
def test_a_tests_dockerfile_on_the_task_image_is_replayed_and_a_prebuilt_images_recipe_is_not(
        tmp_path, dockerfile, docker_image, expected):
    """DeepSWE's environment/Dockerfile builds the task image FROM a base; the
    image already holds its result, so replaying it would run the clone twice."""
    tests = write_overlay(tmp_path / "tests")
    if dockerfile:
        (tests / "Dockerfile").write_text(dockerfile)
    overlay = overlay_for(tests, docker_image)
    assert (overlay and (overlay.base, [(s.name, d) for s, d in overlay.copies], overlay.runs)) == expected


@pytest.mark.parametrize("line", ["ENV X=1", "WORKDIR /tests", "USER nobody", "COPY --chown=1 a b", "FROM other"])
def test_an_overlay_instruction_a_sandbox_cannot_honour_is_refused(tmp_path, line):
    tests = write_overlay(tmp_path / "tests", extra=line + "\n")
    with pytest.raises(ValueError):
        overlay_for(tests, "image:tag")
    (tmp_path / "runtime").touch()
    with pytest.raises(ValueError):   # and before any VM exists
        AgentENVEnvironment(environment_dir=tests, environment_name="task", session_id="t__verifier__trial",
                            task_env_config=EnvironmentConfig(docker_image="image:tag", cpus=1, memory_mb=512),
                            trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(tmp_path / "runtime"))


def test_start_replays_the_overlay_into_the_fresh_vm(tmp_path, monkeypatch):
    tests = write_overlay(tmp_path / "tests")
    runtime = tmp_path / "runtime"
    runtime.touch()
    env = AgentENVEnvironment(environment_dir=tests, environment_name="task", session_id="t__verifier__trial",
                              task_env_config=EnvironmentConfig(docker_image="image:tag", cpus=1, memory_mb=512),
                              trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(runtime))
    commands, uploads = [], []

    class FakeSession:
        sandbox_id = "verifier"

        def __init__(self, **kwargs):
            pass

        def create(self, image, resources):
            return True

        def execute(self, name, arguments, timeout):
            commands.append(arguments["command"])
            return ToolResult(True, "")

        def upload_file(self, source, destination):
            uploads.append((Path(source).name, destination))
            return True

    async def prepared(force):
        return "image@sha256:digest", {"WorkingDir": "/app"}

    monkeypatch.setattr(sandbox_environment, "SandboxSession", FakeSession)
    monkeypatch.setattr(env, "_prepare_image", prepared)
    asyncio.run(env.start())
    assert uploads == [("test.sh", "/tests/test.sh"), ("config.json", "/tests/config.json")]
    assert any("chmod +x /tests/test.sh" in command and "cd /app" in command for command in commands)
    record = json.loads((tmp_path / "t__verifier__trial.agentenv.json").read_text())
    assert record["overlay"]["runs"] == ["chmod +x /tests/test.sh"]


def test_a_verifier_config_without_an_image_boots_the_overlays_from_image(tmp_path, monkeypatch):
    """Harbor's separate verifier for DeepSWE is `[verifier.environment]` with only
    resources; tests/Dockerfile says which image it is built on."""
    from sprout.harbor_plugin import agentenv

    tests = write_overlay(tmp_path / "tests")
    (tmp_path / "runtime").touch()
    env = AgentENVEnvironment(environment_dir=tests, environment_name="task", session_id="t__verifier__trial",
                              task_env_config=EnvironmentConfig(cpus=2, memory_mb=8192),
                              trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(tmp_path / "runtime"))
    assert env.overlay is not None and env.overlay.base == "image:tag"
    resolved = []
    monkeypatch.setattr(agentenv.subprocess, "check_output",
                        lambda argv, **kw: (resolved.append(argv) or ("sha256:digest" if "digest" in argv else "{}")))
    monkeypatch.setattr(agentenv, "find_regctl", lambda: Path("/usr/bin/regctl"))
    image, _ = asyncio.run(env._prepare_image(False))
    assert image == "image:tag@sha256:digest"
    # A Dockerfile that would need a real build is refused: Sprout never builds an image.
    recipe = tmp_path / "environment"
    recipe.mkdir()
    (recipe / "Dockerfile").write_text("FROM base:latest\nWORKDIR /app\nRUN make\n")
    with pytest.raises(ValueError, match="never builds"):
        AgentENVEnvironment(environment_dir=recipe, environment_name="task", session_id="t",
                            task_env_config=EnvironmentConfig(cpus=2, memory_mb=8192),
                            trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(tmp_path / "runtime"))
    with pytest.raises(ValueError, match="does not build"):
        asyncio.run(env._prepare_image(True))
