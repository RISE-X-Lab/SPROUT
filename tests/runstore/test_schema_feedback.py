from copy import deepcopy
import json

import pytest

from sprout.harness_core.journal import read_journal
from sprout.harness_orchestrator.run import Orchestrator
from sprout.harness_checkpoint.rollback import branch_checkpoints
from sprout.contract.mini_history import read_entries
from tests.harness.test_mini_swe import FilesystemSession, model_server, owned_filesystem, pytestmark, reply, spec
from sprout.runstore.messages import export_messages
from sprout.harness_agents.mini_swe.history import index_native
from sprout.contract.mini_history import materialize, read_prefix
from sprout.harness_agents.mini_swe.tools import malformed_tool_calls
from sprout.harness.atif import journal_to_atif
from tests.runstore.test_assistant_turn_branch import restore_files


def malformed_reply(kind="extra", commands=None):
    response = reply(*(commands or ["touch must-not-execute"]), ids=["bad-" + str(i) for i in range(len(commands or [1]))])
    function = response["choices"][0]["message"]["tool_calls"][-1]["function"]
    if kind == "extra":
        arguments = json.loads(function["arguments"])
        arguments["timeout_ms"] = 10000
        function["arguments"] = json.dumps(arguments)
    elif kind == "json":
        function["arguments"] = "{not-json"
    elif kind == "type":
        function["arguments"] = '{"command": 123}'
    elif kind == "unknown":
        function["name"] = "shell"
    return response


@pytest.mark.parametrize("kind", ["extra", "json", "type", "unknown"])
def test_schema_error_reaches_model_then_corrected_call_runs_and_remains_indexable(tmp_path, monkeypatch, kind):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    bad = malformed_reply(kind)
    original = deepcopy(bad)
    with model_server([bad, reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert bad == original
    assert len(requests) == 3 and outcome.usage["input_tokens"] == 33
    assert not (memory.root / "must-not-execute").exists()
    assert (memory.root / "answer").read_text() == "fixed"
    # The original bad call is preserved, not cleaned or silently executed.
    previous = requests[1]["messages"][-2:]
    assert previous[0]["role"] == "assistant"
    assert previous[0]["tool_calls"][0]["function"] == original["choices"][0]["message"]["tool_calls"][0]["function"]
    assert previous[1]["role"] == "tool" and previous[1]["tool_call_id"] == "bad-0"
    feedback = json.loads(previous[1]["content"])
    assert feedback["executed"] is False and feedback["error"]["type"] == "tool_schema_error"
    assert feedback["available_tools"] == requests[1]["tools"]
    events = read_journal(outcome.journal_path)
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["corrected", "submit"]
    assert len([e for e in events if e["type"] == "tool.rejected"]) == 1
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2}
    native = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)
    assert [p.tool_depth for p in points] == [1, 2]
    assert points[0].message_step == 2  # rejected response + corrected response
    messages = export_messages(tmp_path, outcome.native_session_id, "mini-swe-agent")
    assert any(m.get("tool_call_id") == "bad-0" and json.loads(m["content"]).get("executed") is False
               for m in messages if m["role"] == "tool")


def test_entire_mixed_batch_is_rejected_before_any_command_executes(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([
        malformed_reply(commands=["touch valid-sibling-must-not-execute", "touch must-not-execute"]),
        reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
    ]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert not (memory.root / "valid-sibling-must-not-execute").exists()
    assert not (memory.root / "must-not-execute").exists()
    feedback = requests[1]["messages"][-2:]
    assert [m["tool_call_id"] for m in feedback] == ["bad-0", "bad-1"]
    assert [json.loads(m["content"])["error"]["type"] for m in feedback] == [
        "tool_batch_rejected", "tool_schema_error"]
    assert all(json.loads(m["content"])["executed"] is False for m in feedback)


def test_feedback_retries_use_existing_model_budget(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    first, second = malformed_reply(), malformed_reply()
    second["choices"][0]["message"]["tool_calls"][0]["id"] = "bad-second"
    with model_server([first, second]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["rollout_contract"]["max_turns"] = 2
        run_spec.extra["mini"] = {"agent": {"max_consecutive_format_errors": 0}}
        outcome = Orchestrator().run(run_spec)
    assert outcome.stop_reason == "max_turns_reached"
    assert len(requests) == 2 and outcome.usage["input_tokens"] == 22
    events = read_journal(outcome.journal_path)
    assert not any(e["type"] == "tool.started" for e in events)
    # Only the rollout's final snapshot; with no turn behind it, it is no branch point.
    captures = [e for e in events if e["type"] == "checkpoint.captured"]
    assert len(captures) == 1 and captures[0]["final"] and captures[0]["turn_id"] is None
    assert len([e for e in events if e["type"] == "mini.turn.rejected"]) == 2
    assert branch_checkpoints(outcome.journal_path) == {}


def test_provider_protocol_errors_are_not_mislabeled_as_tool_schema_errors(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([{"error": {"message": "provider failed"}}]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "error"
    assert len(requests) == 1
    assert not any(e["type"] == "tool.rejected" for e in read_journal(outcome.journal_path))


def test_correction_checkpoint_can_be_branched_with_rejected_history_intact(tmp_path, monkeypatch):
    parent_dir = tmp_path / "parent"
    memory = FilesystemSession(parent_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([malformed_reply(), reply("printf fixed > answer"),
                       reply("printf future > answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, _):
        parent = Orchestrator().run(spec(parent_dir, url))
    assert parent.status == "completed", parent.error
    native = parent_dir / "native-home" / f"{parent.native_session_id}.jsonl"
    points = index_native(parent.journal_path, native, "mini-swe-agent", parent.native_session_id)
    cut = points[0]
    preserved = read_prefix(cut.native)
    child_dir = tmp_path / "child"
    child = restore_files(memory, cut.snapshot_id, child_dir / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(child))
    with model_server([reply("cat answer"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        run_spec = spec(child_dir, url)
        run_spec.extra.update(materialize(cut.native, child_dir / "cwd",
                              child_dir / "restoration", child_dir / "native-home"), resume_without_hint=True)
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "completed", outcome.error
    assert requests[0]["messages"] == [
        {k: v for k, v in e["message"].items() if k != "extra"}
        for e in read_entries(preserved) if e["type"] == "mini.message"]
    assert "printf future" not in json.dumps(requests)
    assert json.loads(requests[1]["messages"][-1]["content"])["output"] == "fixed"
    child_native = child_dir / "native-home" / f"{outcome.native_session_id}.jsonl"
    assert len(index_native(outcome.journal_path, child_native, "mini-swe-agent",
                            outcome.native_session_id, inherited_native=cut.native)) == 2
    assert read_prefix(cut.native) == preserved
    # Reject a fabricated rejection marker with no corresponding journal proof.
    altered = [e for e in read_journal(parent.journal_path) if e["type"] != "mini.turn.rejected"]
    parent.journal_path.write_text("".join(json.dumps(e) + "\n" for e in altered))
    assert index_native(parent.journal_path, native, "mini-swe-agent", parent.native_session_id) == []


# --- a malformed tool-call envelope ------------------------------------------------
#  INCIDENT: a response whose one call had function.name "" ended a 234-turn run
#  with "ValueError: mini tool calls require a function name and JSON-string
#  arguments". It is the same class of fault as a schema error and gets the same
#  treatment -- feedback and another turn -- but as a user message: no tool result
#  can truthfully answer an id that is missing, blank, repeated or reused, and a
#  strict provider refuses a malformed envelope echoed back in the next request,
#  so the malformed response stays out of the conversation. The journal keeps it.

def malformed_envelope(kind, identifier="bad-0", command="touch must-not-execute"):
    """A response whose tool-call envelope is malformed in the way ``kind`` names."""
    response = reply(command, ids=[identifier], content="Trying a malformed call.", reasoning="Guessing the shape.")
    call = response["choices"][0]["message"]["tool_calls"][0]
    if kind == "missing_name":
        del call["function"]["name"]
    elif kind == "blank_name":
        call["function"]["name"] = ""
    elif kind == "list_arguments":
        call["function"]["arguments"] = [command]
    elif kind == "blank_id":
        call["id"] = ""
    elif kind == "missing_id":
        del call["id"]
    elif kind == "duplicate_id":
        response["choices"][0]["message"]["tool_calls"].append({
            **deepcopy(call), "function": {"name": "bash", "arguments": json.dumps({"command": "touch nor-this"})}})
    return response


ENVELOPE_FAULTS = {
    "missing_name": "function.name is missing or blank", "blank_name": "function.name is missing or blank",
    "list_arguments": "function.arguments must be a JSON-encoded string",
    "blank_id": "id is missing or blank", "missing_id": "id is missing or blank",
    "duplicate_id": "id repeats another call's id in this response",
}


@pytest.mark.parametrize("kind", sorted(ENVELOPE_FAULTS))
def test_malformed_envelope_is_fed_back_as_a_user_message_and_the_run_continues(tmp_path, monkeypatch, kind):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    bad = malformed_envelope(kind)
    with model_server([bad, reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert len(requests) == 3, "the next model turn happens"
    assert not (memory.root / "must-not-execute").exists() and not (memory.root / "nor-this").exists()
    assert (memory.root / "answer").read_text() == "fixed"
    # The model sees a user-role rejection after the task prompt, quoting the
    # rejected calls as data; the malformed assistant message is not re-sent.
    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "user"]
    rejection = messages[-1]["content"]
    assert "rejected before execution" in rejection and ENVELOPE_FAULTS[kind] in rejection
    assert "must-not-execute" in rejection and "<rejected_response>" in rejection
    # The journal records what was rejected and why, and keeps the raw response.
    events = read_journal(outcome.journal_path)
    rejected = [e for e in events if e["type"] == "mini.turn.rejected"]
    assert len(rejected) == 1 and rejected[0]["source"] == "malformed_tool_calls"
    assert rejected[0]["executed"] is False and ENVELOPE_FAULTS[kind] in rejected[0]["problems"][0]["error"]
    assert next(e for e in events if e["type"] == "raw.mini-swe-agent")["response"] == bad
    assert not any(e["type"] == "tool.rejected" for e in events), "no per-call event names an ambiguous id"
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["corrected", "submit"]
    assert [e.get("rejected", False) for e in events if e["type"] == "turn.completed"] == [True, False, False]
    # A rejected turn advances history but is no branch point; the later ones are.
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2}
    native = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)
    assert [p.tool_depth for p in points] == [1, 2]
    exported = export_messages(tmp_path, outcome.native_session_id, "mini-swe-agent")
    assert [m["role"] for m in exported[:4]] == ["system", "user", "user", "assistant"]
    assert "rejected before execution" in exported[2]["content"]
    # What the model said is journaled (agent.message / agent.thinking), so the
    # rejected turn is not a blank step in the ATIF export.
    assert [e["text"] for e in events if e["type"] == "agent.message"] == [
        "Trying a malformed call.", "Thinking.", "Thinking."]
    assert [e["text"] for e in events if e["type"] == "agent.thinking"][0] == "Guessing the shape."
    steps = journal_to_atif(events)["steps"]
    assert len(steps) == 3 and steps[0]["message"] == "Trying a malformed call."
    assert steps[0]["reasoning_content"] == "Guessing the shape."
    assert steps[0]["tool_calls"] == [] and steps[0]["extra"]["rejected"] is True
    assert all("rejected" not in (step.get("extra") or {}) for step in steps[1:])


def test_an_id_reused_from_history_is_rejected_and_the_run_continues(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("printf first > answer", ids=["first"]),
                       reply("touch must-not-execute", ids=["first"]),
                       reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert len(requests) == 4 and not (memory.root / "must-not-execute").exists()
    messages = requests[2]["messages"]
    assert [m["role"] for m in messages[-3:]] == ["assistant", "tool", "user"]
    assert messages[-2]["tool_call_id"] == "first"
    assert "id was already used earlier in this conversation" in messages[-1]["content"]
    events = read_journal(outcome.journal_path)
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["first", "corrected", "submit"]
    rejected = next(e for e in events if e["type"] == "mini.turn.rejected")
    assert rejected["problems"] == [{"index": 0, "id": "first",
                                     "error": "id was already used earlier in this conversation"}]
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2, 3}
    native = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)
    assert [p.tool_depth for p in points] == [1, 2, 3]


@pytest.mark.parametrize("fault,limit,overrides", [
    ("schema", 2, {"agent": {"max_consecutive_format_errors": 2}}),  # the override reaches mini
    ("envelope", 3, {}),  # mini's own default; mini.yaml sets no limit
])
def test_consecutive_format_error_limit_bounds_schema_and_envelope_rejections(
        tmp_path, monkeypatch, fault, limit, overrides):
    """Neither rejection has a bound of its own: both are FormatErrors to mini,
    so its consecutive-format-error limit -- the default of three, or the
    extra.mini.agent.max_consecutive_format_errors override -- ends the run as
    RepeatedFormatError, inside the rollout's max_turns and deadline, which
    count every request (test_feedback_retries_use_existing_model_budget)."""
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    if fault == "schema":
        garbage = [malformed_reply() for _ in range(limit)]
        # Distinct ids: a schema-rejected response stays in the conversation, so a
        # repeated id would be an envelope fault (reused from history), not a schema one.
        for i, response in enumerate(garbage):
            response["choices"][0]["message"]["tool_calls"][0]["id"] = "bad-%d" % i
    else:
        garbage = [malformed_envelope("blank_name", identifier="bad-%d" % i) for i in range(limit)]
    with model_server(garbage + [reply("echo never")]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["mini"] = overrides
        outcome = Orchestrator().run(run_spec)
    assert outcome.status == "error" and outcome.error == "RepeatedFormatError"
    assert len(requests) == limit, "the run stopped at the bound"
    events = read_journal(outcome.journal_path)
    assert not any(e["type"] == "tool.started" for e in events)
    assert len([e for e in events if e["type"] == "mini.turn.rejected"]) == limit


@pytest.mark.parametrize("calls,history,expected", [
    (["not an object"], set(), [(0, "the tool call is not an object")]),
    ([{"id": "a", "type": "tool", "function": {"name": "bash", "arguments": "{}"}}], set(),
     [(0, 'type must be "function"')]),
    ([{"id": "a", "type": "function", "function": None}], set(), [(0, "function must be an object")]),
    ([{"id": "a", "type": "function", "function": {"name": None, "arguments": "{}"}}], set(),
     [(0, "function.name is missing or blank")]),
    ([{"id": "a", "type": "function", "function": {"name": "bash", "arguments": 5}}], set(),
     [(0, "function.arguments must be a JSON-encoded string")]),
    ([{"id": 7, "type": "function", "function": {"name": "bash", "arguments": "{}"}}], set(),
     [(0, "id is missing or blank")]),
    ([{"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}], {"a"},
     [(0, "id was already used earlier in this conversation")]),
    # Both members of a repeated pair are named, not only the later one.
    ([{"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}}] * 2, set(),
     [(0, "id repeats another call's id in this response"), (1, "id repeats another call's id in this response")]),
    # Not faults: object arguments (litellm serialises them, and the schema check
    # then reads them), a missing or null type (read as "function", the wire
    # default) and ids that collide with nothing in the conversation.
    ([{"id": "a", "type": "function", "function": {"name": "bash", "arguments": {"command": "ls"}}},
      {"id": "b", "type": "function", "function": {"name": "bash", "arguments": "{}"}},
      {"id": "d", "function": {"name": "bash", "arguments": "{}"}},
      {"id": "e", "type": None, "function": {"name": "bash", "arguments": "{}"}}], {"c"}, []),
])
def test_every_envelope_fault_is_named_for_the_model(calls, history, expected):
    problems = malformed_tool_calls(calls, history)
    assert [p["index"] for p in problems] == [index for index, _ in expected]
    assert all(error in p["error"] for p, (_, error) in zip(problems, expected))


def test_a_well_formed_sibling_of_a_malformed_call_does_not_execute(tmp_path, monkeypatch):
    """The shape a real model is most likely to send: one good call beside one
    bad one. No call of a response with a malformed envelope runs, the rejection
    names only the faulty call, and the next model turn happens."""
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    mixed = reply("touch valid-sibling-must-not-execute", "touch must-not-execute", ids=["ok-0", "bad-1"])
    mixed["choices"][0]["message"]["tool_calls"][1]["function"]["name"] = ""
    with model_server([mixed, reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert len(requests) == 3, "the next model turn happens"
    assert not (memory.root / "valid-sibling-must-not-execute").exists()
    assert not (memory.root / "must-not-execute").exists()
    assert (memory.root / "answer").read_text() == "fixed"
    rejection = requests[1]["messages"][-1]
    assert rejection["role"] == "user" and "1 of its 2 tool call(s) were malformed" in rejection["content"]
    quoted = json.loads(rejection["content"].split("<rejected_response>\n", 1)[1].split("\n</rejected_response>", 1)[0])
    assert [(c["index"], c["id"]) for c in quoted["rejected_tool_calls"]] == [(1, "bad-1")]
    events = read_journal(outcome.journal_path)
    assert next(e for e in events if e["type"] == "mini.turn.rejected")["problems"] == [
        {"index": 1, "id": "bad-1", "error": "function.name is missing or blank; the only tool is bash"}]
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["corrected", "submit"]
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2}


@pytest.mark.parametrize("shape", ["object_arguments", "no_type", "null_type"])
def test_object_arguments_and_an_absent_type_execute(tmp_path, monkeypatch, shape):
    """litellm serialises object arguments and reads a missing or null type as
    "function" (the wire default), so such calls executed before the envelope
    check existed and still do; the conversation carries the normalised call."""
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    loose = reply("printf fixed > answer", ids=["loose"])
    call = loose["choices"][0]["message"]["tool_calls"][0]
    if shape == "object_arguments":
        call["function"]["arguments"] = {"command": "printf fixed > answer"}
    elif shape == "no_type":
        del call["type"]
    else:
        call["type"] = None
    with model_server([loose, reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert (memory.root / "answer").read_text() == "fixed"
    events = read_journal(outcome.journal_path)
    assert [e["call_id"] for e in events if e["type"] == "tool.started"] == ["loose", "submit"]
    assert not any(e["type"] in {"mini.turn.rejected", "tool.rejected"} for e in events)
    echoed = requests[1]["messages"][-2]
    assert echoed["role"] == "assistant" and echoed["tool_calls"][0]["type"] == "function"
    assert json.loads(echoed["tool_calls"][0]["function"]["arguments"]) == {"command": "printf fixed > answer"}
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2}


def test_a_response_without_tool_calls_is_journaled_as_a_rejected_turn(tmp_path, monkeypatch):
    """mini's own format error -- a response with no tool call -- continues the
    run through upstream's retry; the journal shows the turn as rejected and
    keeps what the model said, as it does for the other two rejections."""
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    silent = reply(content="I believe the task is already done.", reasoning="Nothing left.")
    silent["choices"][0]["finish_reason"] = "stop"
    with model_server([silent, reply("printf fixed > answer", ids=["corrected"]),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", ids=["submit"])]) as (url, requests):
        outcome = Orchestrator().run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert len(requests) == 3 and (memory.root / "answer").read_text() == "fixed"
    # Upstream's treatment is unchanged: the response stays in the conversation,
    # followed by mini's own format-error message.
    messages = requests[1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "I believe the task is already done."
    assert "No tool calls found" in messages[3]["content"]
    events = read_journal(outcome.journal_path)
    rejected = [e for e in events if e["type"] == "mini.turn.rejected"]
    assert len(rejected) == 1 and rejected[0]["source"] == "no_tool_calls" and rejected[0]["executed"] is False
    assert "No tool calls found" in rejected[0]["feedback"]
    assert [e.get("rejected", False) for e in events if e["type"] == "turn.completed"] == [True, False, False]
    assert [e["text"] for e in events if e["type"] == "agent.message"][0] == "I believe the task is already done."
    assert [e["text"] for e in events if e["type"] == "agent.thinking"][0] == "Nothing left."
    assert not any(e["type"] == "tool.rejected" for e in events)
    assert set(branch_checkpoints(outcome.journal_path)) == {1, 2}
    native = tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl"
    points = index_native(outcome.journal_path, native, "mini-swe-agent", outcome.native_session_id)
    assert [p.tool_depth for p in points] == [1, 2]
    steps = journal_to_atif(events)["steps"]
    assert len(steps) == 3 and steps[0]["message"] == "I believe the task is already done."
    assert steps[0]["tool_calls"] == [] and steps[0]["extra"]["rejected"] is True
