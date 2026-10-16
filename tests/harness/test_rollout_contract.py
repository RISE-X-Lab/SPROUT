import time
from types import SimpleNamespace

import httpx
import pytest

from sprout.harness_core.control import RunControl
from sprout.harness_core.journal import read_journal
from sprout.harness_core.result import ToolResult
from sprout.harness_core.slot import SlotResult
from sprout.harness_tools.pipeline import CallContext
from sprout.harness_orchestrator.run import Orchestrator
from sprout.contract.run_spec import RunSpec
from sprout.harness_orchestrator.rollout import RolloutControls
from sprout.contract.messages import export_tools
from tests.harness.test_gateway import upstream as upstream

TOOL = {"type": "function", "function": {
    "name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}


def contract(url, **changes):
    return {"model_endpoint": url, "model": "checkpoint-model", "deadline_at": time.time() + 30,
            "message_export": True, "max_turns": 2,
            "sampling_params": {"temperature": 0.4, "max_new_tokens": 7, "top_k": 11, "stop": ["stop-here"]},
            **changes}


def test_orchestrator_mounts_real_endpoint_sampling_and_turn_limit(tmp_path, upstream, monkeypatch):
    destroyed, executed = [], []
    session = SimpleNamespace(sandbox_id="test-sandbox", supports_snapshot=lambda: False)
    owned = SimpleNamespace(session=session, sandbox_id="test-sandbox", keep=False,
                            destroy=lambda: destroyed.append(True))
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned)
    # The stand-in sandbox has no shell for mini's workspace probe.
    monkeypatch.setattr("sprout.harness_agents.mini_swe.sandbox.resolve_workspace",
                        lambda session, extra, journal: extra)

    class Slot:
        def run(self, task, journal, tools):
            url = task.env["OPENAI_BASE_URL"] + "/chat/completions"
            headers = {"Authorization": "Bearer " + task.env["OPENAI_API_KEY"]}
            for _ in range(2):
                response = httpx.post(url, headers=headers, timeout=5,
                                      json={"model": "agent-requested", "temperature": 1, "tools": [TOOL]})
                assert response.status_code == 200
            ctx = CallContext("agent", "sandbox", "shell", {"command": "true"})

            def execute(name, args):
                executed.append(name)
                return ToolResult(True, "ok")

            # Tool calls are counted, not limited.
            for _ in range(10):
                assert tools.pipeline.execute(ctx, execute).success
            # The third model call is past max_turns.
            assert httpx.post(url, headers=headers, json={"model": "agent-requested"}, timeout=5).status_code == 400
            return SlotResult(status="completed")

    monkeypatch.setattr("sprout.harness_orchestrator.run.MiniSweSlot", Slot)
    path = tmp_path / "journal.jsonl"
    result = Orchestrator().run(RunSpec(prompt="task", slot="mini-swe-agent", journal_path=path,
                                       extra={"rollout_contract": contract(upstream.base_url + "/v1")}))
    assert result.status == "error" and "budget" in result.error
    assert result.stop_reason == "max_turns_reached"
    assert len(upstream.requests) == 2 and executed == ["shell"] * 10 and destroyed
    for item in upstream.requests:
        assert item["body"]["temperature"] == 0.4
        assert item["body"]["max_tokens"] == 7
        assert item["body"]["top_k"] == 11
        assert item["body"]["stop"] == ["stop-here"]
        assert item["body"]["model"] == "checkpoint-model"
        assert item["body"]["tools"] == [TOOL]
        # The contract names no credential, so none is forged and the slot token is stripped.
        assert "authorization" not in {name.lower() for name in item["headers"]}
    usage = [e for e in read_journal(path) if e["type"] == "rollout.usage"][-1]
    assert usage["model_calls"] == 2 and usage["tool_calls"] == 10
    # Training export reads the schema the model was actually offered from the journal.
    assert export_tools(read_journal(path)) == [TOOL]


@pytest.mark.parametrize("changes, match", [
    # True pins `type(max_turns) is int`: a bool passes isinstance and would cap every rollout at one turn.
    *(pytest.param({"max_turns": value}, "max_turns", id=f"max_turns={value}")
      for value in (None, 0, -1, True, 1.5)),
    # The contract has no version tag, so the pre-max_turns keys are refused rather than silently ignored.
    pytest.param({"max_model_calls": 2}, "model/tool call budgets", id="legacy-max_model_calls"),
    pytest.param({"max_tool_calls": 2}, "model/tool call budgets", id="legacy-max_tool_calls"),
])
def test_controls_reject_malformed_contract(changes, match):
    with pytest.raises(ValueError, match=match):
        RolloutControls(contract("http://model", **changes), None, RunControl())


def test_expired_group_never_allocates_a_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr("sprout.harness_orchestrator.run.MiniSweSlot", lambda: SimpleNamespace())
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: pytest.fail("Expired group allocated a VM"))
    result = Orchestrator().run(RunSpec(prompt="task", journal_path=tmp_path / "journal",
                                       extra={"rollout_contract": contract("http://model", deadline_at=time.time() - 1)}))
    assert "expired before execution" in result.error
