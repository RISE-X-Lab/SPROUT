import json

import pytest

from sprout.harness.core.journal import read_journal
from sprout.harness_orchestrator.run import Orchestrator
from sprout.harness.agents.mini_swe.history import read_entries
from tests.harness.test_assistant_turn import assistant_turn
from tests.harness.test_mini_swe import model_server, owned_filesystem, pytestmark, reply, spec
from sprout.harness.agents.mini_swe.history import index_native, materialize, read_prefix
from tests.runstore.test_assistant_turn_branch import restore_files
from tests.runstore.test_mini_native import parent_run


def native_messages(reference):
    return [{k: v for k, v in e["message"].items() if k != "extra"}
            for e in read_entries(read_prefix(reference)) if e["type"] == "mini.message"]


def test_point_only_resume_never_adds_even_an_empty_message_and_remains_branchable(tmp_path, monkeypatch):
    memory, _, points = parent_run(tmp_path / "parent", monkeypatch)
    cut = points[0]
    original = read_prefix(cut.native)
    for generation in range(2):
        directory = tmp_path / f"child-{generation}"
        child = restore_files(memory, cut.snapshot_id, directory / "sandbox")
        monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
        with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
            run_spec = spec(directory, url, prompt="NO_NEW_USER_MESSAGE")
            run_spec.extra.update(materialize(cut.native, directory / "cwd", directory / "restoration",
                                  directory / "native-home"), resume_without_hint=True)
            outcome = Orchestrator().run(run_spec)
        assert outcome.status == "completed", outcome.error
        assert requests[0]["messages"] == native_messages(cut.native)
        assert len(requests) == 2 and outcome.usage["input_tokens"] == 22
        assert "NO_NEW_USER_MESSAGE" not in json.dumps(requests)
        assert "echo late" not in json.dumps(requests)
        assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "early\n"
        assert not any(e["type"] == "branch.assistant_turn" for e in read_journal(outcome.journal_path))
        native = directory / "native-home" / f"{outcome.native_session_id}.jsonl"
        new_points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id,
                                  inherited_native=cut.native)
        assert [p.tool_depth for p in new_points] == [1, 2]
        if generation == 0:
            assert read_prefix(cut.native) == original
        cut, memory = new_points[0], child


@pytest.mark.parametrize("extra,match", [
    ({"resume_without_hint": True}, "exact mini history prefix"),
    ({"resume_without_hint": "true"}, "must be a boolean"),
    ({"resume_without_hint": True, "assistant_turn": assistant_turn()}, "cannot include an assistant_turn"),
])
def test_bad_point_only_configuration_stops_without_model_or_tool_calls(tmp_path, monkeypatch, extra, match):
    from tests.harness.test_mini_swe import FilesystemSession

    child = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra.update(extra)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "error" and match in outcome.error
    assert requests == []
    assert not any(e["type"] == "tool.started" for e in read_journal(outcome.journal_path))
