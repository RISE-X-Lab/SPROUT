# ruff: noqa: E402
"""The shared worker, branch restoration, reviewer and training-export boundaries."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

pytest.importorskip("opencollab")

from sprout.contract.run_spec import RunSpec
from sprout.worker.agents.history import available_branch_points, reference_at
from sprout.worker.jobs.export import complete_message_result
from sprout.worker.jobs.review.structured_review import extract_assistant_turn_plan
from sprout.worker.jobs.rollout import _from_point
from sprout.worker.run.runner import _agent_slot

from .test_opencollab import Model, call, response, run
from .test_review50_protocol import repository


def test_runtime_submission_receipt_is_context_without_becoming_policy_output(tmp_path):
    root = repository(tmp_path / "repo")
    result, _, journal = run(
        root, tmp_path / "run", Model([response(call("submit", "submit_patch"))]), protocol="review50-v1"
    )
    assert result.status == "completed", result.error
    exported = complete_message_result(asdict(result), journal.parent, "opencollab-base")
    assert "training_export_error" not in exported
    messages = exported["training_messages"]
    receipt = len(messages) - 1
    action = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
    assert receipt in exported["training_provenance"]["inserted_messages"]
    assert action not in exported["training_provenance"]["inserted_messages"]
    assert exported["training_provenance"]["prefix_messages"] == 0
    assert "patch_sha256" in messages[receipt]["content"]


@pytest.mark.parametrize("rendering", ["content", "reasoning"])
def test_queued_branch_materializes_exact_sdk_state_and_masks_reviewer(tmp_path, rendering):
    root = repository(tmp_path / "repo")
    parent, sandbox, journal = run(
        root,
        tmp_path / "parent",
        Model([response(call("read", "read_file", filepath="source.txt")), response(content="Parent done.")]),
        protocol="review50-v1",
    )
    assert parent.status == "completed", parent.error
    step, checkpoint = min(available_branch_points(journal).items())
    native = reference_at(journal, step, parent.native_session_id)
    point = {
        "id": "point",
        "snapshot_id": checkpoint.snapshot_id,
        "tool_depth": step,
        "message_step": 1,
        "native": native,
    }
    turn = {
        "role": "assistant",
        "content": "Check the retained source before editing.",
        "tool_calls": [call("review", "run_command", command="git diff --stat")],
    }
    destination = tmp_path / "queued"
    fields = _from_point(
        {"prompt": "Fix the fixture", "extra": {}},
        point,
        continuation={
            "guidance": "assistant-turn",
            "assistant_turn": turn,
            "assistant_turn_rendering": rendering,
        },
        directory=destination,
        slot="opencollab-base",
    )
    assert fields["sandbox_image"] == checkpoint.snapshot_id
    assert Path(fields["extra"]["native_prefix"]["path"]).is_file()
    result, _, child_journal = run(
        root,
        tmp_path / "child",
        Model([response(content="Child done.")]),
        prefix=fields["extra"]["native_prefix"],
        reviewer=fields["extra"]["assistant_turn"],
        sandbox=sandbox,
        protocol="review50-v1",
    )
    assert result.status == "completed", result.error
    exported = complete_message_result(
        asdict(result), child_journal.parent, "opencollab-base", recovery=point
    )
    assert "training_export_error" not in exported
    provenance = exported["training_provenance"]
    prefix = provenance["prefix_messages"]
    assert prefix > 0 and prefix in provenance["inserted_messages"]
    assert len(exported["training_messages"]) - 1 not in provenance["inserted_messages"]


def test_flat_review_commands_use_the_selected_parents_actual_tool():
    from sprout.worker.agents.opencollab.review50 import definition

    branch = {
        "name": "repair",
        "base": "parent",
        "branch_step": 1,
        "why": "inspect",
        "content": "Inspect the source first.",
        "commands": ["git diff"],
    }
    plan = extract_assistant_turn_plan(
        json.dumps({"synthesis": "Inspect first", "branches": [branch]}),
        json_only=True,
        tools_by_base={"parent": definition()["tools"]},
    )
    tool = plan["branches"][0]["assistant_turn"]["tool_calls"][0]
    assert tool["function"]["name"] == "run_command"
    assert json.loads(tool["function"]["arguments"]) == {"command": "git diff"}
    mini = extract_assistant_turn_plan(
        json.dumps({"synthesis": "Inspect first", "branches": [branch]}), json_only=True
    )
    assert mini["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["name"] == "bash"


def test_slot_is_selected_by_the_existing_worker_runner():
    assert _agent_slot(RunSpec(prompt="fix", slot="opencollab-base")).name == "opencollab-base"
    assert _agent_slot(RunSpec(prompt="fix")).name == "mini-swe-agent"


@pytest.mark.parametrize(
    "protocol,success", [("single2", True), ("review50-v1", True), ("review50-v1", False)]
)
def test_historical_workspace_alias_is_prepared_without_overwriting_another_checkout(
    tmp_path, monkeypatch, protocol, success
):
    from types import SimpleNamespace

    from sprout.worker.agents.opencollab import workspace
    from sprout.worker.run.journal import JournalWriter
    from sprout.worker.tools.result import ToolResult

    resolved = {
        "mini_workspace": {"cwd": "/testbed", "repository_dir": "/testbed"},
        "opencollab_tool_protocol": protocol,
    }
    monkeypatch.setattr(workspace, "resolve_native_workspace", lambda *_: resolved)
    calls = []

    def execute(name, arguments, **kwargs):
        calls.append((name, arguments))
        return ToolResult(success, "")

    session = SimpleNamespace(execute=execute)
    with JournalWriter(tmp_path / "workspace.jsonl") as journal:
        if not success:
            with pytest.raises(ValueError, match="alias"):
                workspace.resolve_workspace(session, resolved, journal)
        else:
            assert workspace.resolve_workspace(session, resolved, journal) == resolved
    if protocol == "single2":
        assert not calls
    else:
        assert len(calls) == 1 and calls[0][0] == "shell"
        assert 'test "$(cd /workspace && pwd -P)" = /testbed' in calls[0][1]["command"]
        assert "ln -s -- /testbed /workspace" in calls[0][1]["command"]


def test_native_sdk_uses_nonstreaming_requests_through_the_real_gateway(tmp_path, monkeypatch):
    from sprout.worker.gateway import GatewayServer, RoutingTable
    from sprout.worker.gateway.routing import ModelRoute
    from sprout.worker.run.journal import JournalWriter
    from tests.support.http import chat_server

    completion = {
        "id": "response",
        "object": "chat.completion",
        "created": 0,
        "model": "fixture-model",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Done."}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    with chat_server([completion]) as provider, JournalWriter(tmp_path / "gateway.jsonl") as journal:
        table = RoutingTable({"default": ModelRoute(base_url=provider.url)})
        gateway = GatewayServer(table, journal=journal).start()
        token = table.mint("agent")
        try:
            for key, value in gateway.env_for(token).items():
                monkeypatch.setenv(key, value)
            checkout = tmp_path / "repo"
            checkout.mkdir()
            result, _, _ = run(checkout, tmp_path / "run", None)
        finally:
            gateway.stop()
    assert result.status == "completed", result.error
    assert result.final_text == "Done."
    assert len(provider.requests) == 1 and not provider.requests[0].get("stream", False)
