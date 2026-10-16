from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from sprout.eval.harbor import branching
from sprout.worker.agents.mini_swe.tools import mini_tool_schema
from tests.harness.test_assistant_turn import assistant_turn


SNAPSHOT = "01a0a960-85f0-75b1-9d54-14da424d2d61"


def context_file(tmp_path: Path) -> Path:
    parent = tmp_path / "parent.jsonl"
    parent.write_text('{"type":"run.started","slot":"mini-swe-agent"}\n')
    (tmp_path / "task" / "environment").mkdir(parents=True, exist_ok=True)
    (tmp_path / "task" / "tests").mkdir(parents=True, exist_ok=True)
    context = {
        "task_dir": str(tmp_path / "task"),
        "branch_guidance": "assistant-turn", "checkpoint_mode": "disk_only",
        "snapshot_id": SNAPSHOT, "base_image": "image@sha256:digest",
        "image_config": {"User": "nobody", "WorkingDir": "/work", "Env": []},
        "parent_journal": str(parent),
        "parent_journal_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
        "step": 2, "native_session_id": "session", "conversation_cut": "turn-id",
        "assistant_turn": assistant_turn("pytest -q tests/test_boundary.py"),
    }
    path = tmp_path / "branch-context.json"
    path.write_text(json.dumps(context))
    return path


def snapshot_environment(tmp_path: Path, context: Path, **changes):
    environment_dir = changes.pop("environment_dir", tmp_path / "task" / "environment")
    environment_dir.mkdir(parents=True, exist_ok=True)
    runtime = tmp_path / "runtime"
    runtime.touch()
    return branching.SnapshotEnvironment(
        environment_dir=environment_dir, environment_name="task",
        session_id=changes.pop("session_id", "branch"),
        task_env_config=EnvironmentConfig(docker_image="image@sha256:digest"),
        trial_paths=TrialPaths(trial_dir=tmp_path), runtime_bin=str(runtime),
        branch_context=str(context), checkpoint_mode="disk_only", **changes,
    )


def test_branch_context_needs_per_call_snapshots_and_an_unchanged_parent(tmp_path):
    path = context_file(tmp_path)
    data = json.loads(path.read_text())
    assert branching.read_branch_context(path)["checkpoint_mode"] == "disk_only"
    data["checkpoint_mode"] = "none"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="per-call snapshots"):
        branching.read_branch_context(path)
    data["checkpoint_mode"] = "disk_only"
    path.write_text(json.dumps(data))
    Path(data["parent_journal"]).write_text("changed\n")
    with pytest.raises(ValueError, match="changed"):
        branching.read_branch_context(path)


def test_branch_mini_rechecks_snapshot_and_injects_only_exact_prefix(tmp_path, monkeypatch):
    path = context_file(tmp_path)
    environment = snapshot_environment(tmp_path, path)
    environment.session = SimpleNamespace()
    environment.backend = {"backend": "microvm"}
    environment.image = SNAPSHOT
    agent = branching.BranchMini(
        logs_dir=tmp_path / "agent", model_name="fixture",
        inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_KEY",
        branch_context=str(path),
    )
    point = SimpleNamespace(snapshot_id=SNAPSHOT, session_ckpt="session")
    monkeypatch.setattr(branching, "available_branch_points", lambda _: {2: point})
    reference = {"slot": "mini-swe-agent", "session_id": "session", "cut": "turn-id",
                 "path": "native.jsonl", "byte_length": 10, "sha256": "digest"}
    monkeypatch.setattr(branching, "reference_at", lambda *_: reference)
    monkeypatch.setattr(branching, "load_prefix", lambda _: [{"type": "mini.session"}])
    monkeypatch.setattr(branching, "actor_tools_at", lambda *_: mini_tool_schema())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = agent._make_spec("must not repeat parent prompt", workspace, tmp_path / "child.jsonl", environment)
    assert spec.prompt == "" and spec.slot == "mini-swe-agent"
    assert spec.extra["native_prefix"] == reference
    assert spec.extra["assistant_turn"] == json.loads(path.read_text())["assistant_turn"]
    assert spec.origin["snapshot_id"] == SNAPSHOT

    point.snapshot_id = "wrong"
    with pytest.raises(ValueError, match="does not match"):
        agent._make_spec("", workspace, tmp_path / "child.jsonl", environment)


def test_the_verifier_built_from_the_same_config_boots_the_pristine_image(tmp_path, monkeypatch):
    """Harbor creates a trial's separate verifier from the trial's environment
    config -- for a branch, SnapshotEnvironment -- but from the task's tests/
    directory rather than environment/. The verifier must grade on the task
    image plus tests/Dockerfile, not on the branch's snapshot. (Session ids are
    not a usable signal: Harbor truncates long ones.) The agent-side
    environment, by contrast, boots the selected snapshot under the recorded
    image config and refuses a forced rebuild."""
    context = context_file(tmp_path)
    verifier = snapshot_environment(tmp_path, context, environment_dir=tmp_path / "task" / "tests",
                                    session_id="task-r1b1-a-very-long-name__veri__a1b90ec2")
    assert verifier.restores_snapshot is False

    async def pristine(self, force):
        return "image@sha256:digest", {"WorkingDir": "/app"}

    monkeypatch.setattr(branching.AgentENVEnvironment, "_prepare_image", pristine)
    assert asyncio.run(verifier._prepare_image(False)) == ("image@sha256:digest", {"WorkingDir": "/app"})
    agent_env = snapshot_environment(tmp_path, context, session_id="task-r1b1")
    assert agent_env.restores_snapshot is True
    assert asyncio.run(agent_env._prepare_image(False)) == (
        SNAPSHOT, {"User": "nobody", "WorkingDir": "/work", "Env": []})
    with pytest.raises(ValueError, match="restore its snapshot"):
        asyncio.run(agent_env._prepare_image(True))


@pytest.mark.parametrize("guidance", ["user-hint", "none"])
def test_hint_and_point_only_branches_reach_mini_as_declared(tmp_path, monkeypatch, guidance):
    path = context_file(tmp_path)
    data = json.loads(path.read_text())
    data.pop("assistant_turn")
    data.update(branch_guidance=guidance, commit_required=True)
    if guidance == "user-hint":
        data["hint"] = "Check the boundary condition."
    path.write_text(json.dumps(data))
    environment = snapshot_environment(tmp_path, path)
    environment.session = SimpleNamespace()
    environment.backend = {"backend": "microvm"}
    environment.image = SNAPSHOT
    agent = branching.BranchMini(logs_dir=tmp_path / "agent", model_name="fixture",
                                 inference_endpoint="http://127.0.0.1:18252", api_key_env="EVAL_KEY",
                                 branch_context=str(path))
    point = SimpleNamespace(snapshot_id=SNAPSHOT, session_ckpt="session")
    monkeypatch.setattr(branching, "available_branch_points", lambda _: {2: point})
    reference = {"slot": "mini-swe-agent", "session_id": "session", "cut": "turn-id",
                 "path": "native.jsonl", "byte_length": 10, "sha256": "digest"}
    monkeypatch.setattr(branching, "reference_at", lambda *_: reference)
    monkeypatch.setattr(branching, "load_prefix", lambda _: [{"type": "mini.session"}])
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = agent._make_spec("ignored", workspace, tmp_path / "child.jsonl", environment)
    assert "assistant_turn" not in spec.extra
    if guidance == "user-hint":
        assert "Check the boundary condition." in spec.prompt and "Commit your changes" in spec.prompt
        assert "resume_without_hint" not in spec.extra
        assert spec.origin["actor_hint"] == "Check the boundary condition."
    else:
        assert spec.prompt == "" and spec.extra["resume_without_hint"] is True
    data.pop("hint", None)
    path.write_text(json.dumps(data))
    if guidance == "user-hint":
        with pytest.raises(ValueError, match="non-empty hint"):
            branching.read_branch_context(path)
