"""A model response that a run rejects: no call of it runs, the model learns why, and the rollout goes on.

A response is rejected for a call that breaks the bash schema, for a malformed tool-call envelope, or
for having no tool call. Each case runs mini on a ``DirectorySession`` against a provider stub that
sends its replies unchanged, also the ids that a reply leaves out.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.agents.mini_swe.tools import malformed_tool_calls
from sprout.worker.checkpoint.rollback import branch_checkpoints
from sprout.worker.jobs.export import export_messages
from sprout.worker.run.atif import journal_to_atif
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import Runner, RunOutcome
from tests.support.http import chat_completion
from tests.worker.run.directory_session import DirectorySession
from tests.worker.run.rollouts import SUBMIT, events, needs_mini, rollout_contract, rollout_spec, transcript
from tests.worker.run.upstream import provider

pytestmark = needs_mini

#: What the model sees in the tool result of each call of a response that failed the schema.
SCHEMA_INSTRUCTION = "No calls in this response were executed. Correct and resend the tool calls."


def run(
    tmp_path: Path, replies: list[dict[str, Any]], **changes: Any
) -> tuple[RunOutcome, list[dict[str, Any]]]:
    """The outcome of a mini rollout on ``replies``, and the body of each model request, in order."""
    with provider(replies) as upstream:
        outcome = Runner().run(rollout_spec(tmp_path, upstream.url, **changes))
    return outcome, [request.body for request in upstream.requests]


def corrected_and_submitted() -> list[dict[str, Any]]:
    """The replies after a rejection: a call that writes ``fixed`` to ``answer``, then the submission."""
    return [
        chat_completion("printf fixed > answer", ids=("corrected",)),
        chat_completion(SUBMIT, ids=("submit",)),
    ]


def schema_fault(kind: str) -> dict[str, Any]:
    """A reply whose one call, ``bad-0``, breaks the bash schema in the way that ``kind`` names."""
    reply = chat_completion("touch must-not-execute", ids=("bad-0",))
    function = reply["choices"][0]["message"]["tool_calls"][0]["function"]
    if kind == "extra-argument":
        function["arguments"] = json.dumps({"command": "touch must-not-execute", "timeout_ms": 10000})
    elif kind == "invalid-json":
        function["arguments"] = "{not-json"
    elif kind == "wrong-type":
        function["arguments"] = '{"command": 123}'
    elif kind == "unknown-tool":
        function["name"] = "shell"
    return reply


def envelope_fault(kind: str) -> dict[str, Any]:
    """A reply whose tool-call envelope is malformed in the way that ``kind`` names."""
    reply = chat_completion(
        "touch must-not-execute", ids=("bad-0",), content="A malformed call.", reasoning="Guess."
    )
    call = reply["choices"][0]["message"]["tool_calls"][0]
    if kind == "missing-name":
        del call["function"]["name"]
    elif kind == "blank-name":
        call["function"]["name"] = ""
    elif kind == "list-arguments":
        call["function"]["arguments"] = ["touch must-not-execute"]
    elif kind == "blank-id":
        call["id"] = ""
    elif kind == "missing-id":
        del call["id"]
    elif kind == "duplicate-id":
        twin = {
            **deepcopy(call),
            "function": {"name": "bash", "arguments": json.dumps({"command": "touch nor-this"})},
        }
        reply["choices"][0]["message"]["tool_calls"].append(twin)
    return reply


def rejected_response(content: str) -> dict[str, Any]:
    """The rejected response that a user-role rejection quotes between its ``<rejected_response>`` tags."""
    quoted = content.split("<rejected_response>\n", 1)[1].split("\n</rejected_response>", 1)[0]
    return json.loads(quoted)


def drop_events(journal: Path, event_type: str) -> None:
    """Rewrite ``journal`` without its events of ``event_type``, as a journal that lacks them."""
    kept = [record for record in read_journal(journal) if record["type"] != event_type]
    journal.write_text("".join(json.dumps(record) + "\n" for record in kept))


def recovery_depths(tmp_path: Path, outcome: RunOutcome) -> list[int]:
    """The tool depth of each recovery point of the run's native history."""
    return [depth for _, depth in recovery_points(tmp_path, outcome)]


def recovery_points(tmp_path: Path, outcome: RunOutcome) -> list[tuple[int, int]]:
    """``(message_step, tool_depth)`` of each recovery point of the run's native history."""
    points = index_native(
        outcome.journal_path, transcript(tmp_path, outcome), "mini-swe-agent", outcome.native_session_id
    )
    return [(point.message_step, point.tool_depth) for point in points]


@pytest.mark.parametrize(
    "kind,message",
    [
        ("extra-argument", "Additional properties are not allowed ('timeout_ms' was unexpected)"),
        (
            "invalid-json",
            "Arguments are not valid JSON: "
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
        ),
        ("wrong-type", "123 is not of type 'string'"),
        ("unknown-tool", "Unknown tool. The available function is bash(command)."),
    ],
)
def test_a_call_that_breaks_the_bash_schema_does_not_run_and_its_tool_result_is_the_schema_error(
    tmp_path, sandbox: DirectorySession, kind, message
):
    bad = schema_fault(kind)

    outcome, requests = run(tmp_path, [bad, *corrected_and_submitted()])

    echoed, result = requests[1]["messages"][-2:]
    assert (sandbox.root / "must-not-execute").exists() is False
    assert echoed["tool_calls"][0]["function"] == bad["choices"][0]["message"]["tool_calls"][0]["function"]
    assert (result["role"], result["tool_call_id"], json.loads(result["content"])) == (
        "tool",
        "bad-0",
        {
            "error": {"type": "tool_schema_error", "message": message},
            "executed": False,
            "instruction": SCHEMA_INSTRUCTION,
            "available_tools": requests[1]["tools"],
        },
    )


def test_after_a_schema_error_only_the_corrected_calls_are_tool_steps_and_branch_points(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [schema_fault("extra-argument"), *corrected_and_submitted()])

    assert (sandbox.root / "answer").read_text() == "fixed"
    assert [e["call_id"] for e in events(outcome, "tool.started")] == ["corrected", "submit"]
    assert [e["call_id"] for e in events(outcome, "tool.rejected")] == ["bad-0"]
    assert sorted(branch_checkpoints(outcome.journal_path)) == [1, 2]


def test_the_message_step_of_a_recovery_point_counts_the_rejected_response(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [schema_fault("extra-argument"), *corrected_and_submitted()])

    assert recovery_points(tmp_path, outcome) == [(2, 1), (3, 2)]


@pytest.mark.parametrize(
    "rejected",
    [
        pytest.param(lambda: envelope_fault("blank-name"), id="malformed-envelope"),
        pytest.param(lambda: no_tool_call(), id="no-tool-call"),
    ],
)
def test_the_message_step_of_a_recovery_point_counts_each_model_turn_before_it(
    tmp_path, sandbox: DirectorySession, rejected
):
    """A branch gets ``max_turns - message_step`` turns: the turns that its parent had left at the point."""
    outcome, _ = run(tmp_path, [rejected(), *corrected_and_submitted()])

    assert recovery_points(tmp_path, outcome) == [(2, 1), (3, 2)]


def test_the_training_export_keeps_the_rejected_call_and_its_schema_error(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [schema_fault("extra-argument"), *corrected_and_submitted()])

    messages = export_messages(tmp_path, outcome.native_session_id, "mini-swe-agent")

    results = [message for message in messages if message["role"] == "tool"]
    assert [(m["tool_call_id"], json.loads(m["content"]).get("executed")) for m in results] == [
        ("bad-0", False),
        ("corrected", None),
        ("submit", None),
    ]


def test_no_call_of_a_response_runs_when_one_of_its_calls_breaks_the_schema(
    tmp_path, sandbox: DirectorySession
):
    mixed = chat_completion(
        "touch valid-sibling-must-not-execute", "touch must-not-execute", ids=("ok-0", "bad-1")
    )
    mixed["choices"][0]["message"]["tool_calls"][1]["function"]["arguments"] = '{"command": 123}'

    outcome, requests = run(tmp_path, [mixed, *corrected_and_submitted()])

    results = requests[1]["messages"][-2:]
    assert sorted(path.name for path in sandbox.root.iterdir()) == ["answer"]
    assert [(m["tool_call_id"], json.loads(m["content"])["error"]["type"]) for m in results] == [
        ("ok-0", "tool_batch_rejected"),
        ("bad-1", "tool_schema_error"),
    ]


def test_rejected_responses_spend_the_turn_budget_of_the_rollout(tmp_path, sandbox: DirectorySession):
    second = schema_fault("extra-argument")
    second["choices"][0]["message"]["tool_calls"][0]["id"] = "bad-1"

    with provider([schema_fault("extra-argument"), second]) as upstream:
        spec = rollout_spec(tmp_path, upstream.url, contract=rollout_contract(upstream.url, max_turns=2))
        spec.extra["mini"] = {"agent": {"max_consecutive_format_errors": 0}}
        outcome = Runner().run(spec)

    assert (outcome.stop_reason, len(upstream.requests), len(events(outcome, "mini.turn.rejected"))) == (
        "max_turns_reached",
        2,
        2,
    )
    assert (events(outcome, "tool.started"), branch_checkpoints(outcome.journal_path)) == ([], {})


def test_a_provider_error_is_not_taken_for_a_schema_error(tmp_path, sandbox: DirectorySession):
    outcome, requests = run(tmp_path, [{"error": {"message": "provider failed"}}])

    assert (outcome.status, outcome.error, len(requests)) == (
        "error",
        "ValueError: mini model response is missing a valid terminal choice",
        1,
    )
    assert events(outcome, "tool.rejected", "mini.turn.rejected") == []


ENVELOPE_FAULTS = [
    pytest.param(
        "missing-name",
        [{"index": 0, "id": "bad-0", "error": "function.name is missing or blank; the only tool is bash",
          "name": None, "arguments": '{"command": "touch must-not-execute"}'}],
        id="missing-name",
    ),
    pytest.param(
        "blank-name",
        [{"index": 0, "id": "bad-0", "error": "function.name is missing or blank; the only tool is bash",
          "name": "", "arguments": '{"command": "touch must-not-execute"}'}],
        id="blank-name",
    ),
    pytest.param(
        "list-arguments",
        [{"index": 0, "id": "bad-0", "error": "function.arguments must be a JSON-encoded string",
          "name": "bash", "arguments": '["touch must-not-execute"]'}],
        id="list-arguments",
    ),
    pytest.param(
        "blank-id",
        [{"index": 0, "id": "", "error": "id is missing or blank",
          "name": "bash", "arguments": '{"command": "touch must-not-execute"}'}],
        id="blank-id",
    ),
    pytest.param(
        "missing-id",
        [{"index": 0, "id": None, "error": "id is missing or blank",
          "name": "bash", "arguments": '{"command": "touch must-not-execute"}'}],
        id="missing-id",
    ),
    # Both calls of a repeated pair are named, not only the later one.
    pytest.param(
        "duplicate-id",
        [{"index": 0, "id": "bad-0", "error": "id repeats another call's id in this response",
          "name": "bash", "arguments": '{"command": "touch must-not-execute"}'},
         {"index": 1, "id": "bad-0", "error": "id repeats another call's id in this response",
          "name": "bash", "arguments": '{"command": "touch nor-this"}'}],
        id="duplicate-id",
    ),
]  # fmt: skip


@pytest.mark.parametrize("kind,rejected_calls", ENVELOPE_FAULTS)
def test_a_malformed_envelope_is_rejected_with_a_user_message_that_quotes_it_and_the_run_goes_on(
    tmp_path, sandbox: DirectorySession, kind, rejected_calls
):
    """A strict provider refuses a malformed envelope sent back, so the conversation gets a user message."""
    outcome, requests = run(tmp_path, [envelope_fault(kind), *corrected_and_submitted()])

    messages = requests[1]["messages"]
    assert ([message["role"] for message in messages], rejected_response(messages[-1]["content"])) == (
        ["system", "user", "user"],
        {"content": "A malformed call.", "rejected_tool_calls": rejected_calls},
    )
    assert (outcome.status, sorted(path.name for path in sandbox.root.iterdir())) == ("completed", ["answer"])


def test_a_malformed_envelope_is_journaled_as_a_rejected_turn_with_the_raw_response(
    tmp_path, sandbox: DirectorySession
):
    bad = envelope_fault("blank-name")

    outcome, _ = run(tmp_path, [bad, *corrected_and_submitted()])

    [rejected] = events(outcome, "mini.turn.rejected")
    assert {key: rejected[key] for key in ("source", "executed", "problems")} == {
        "source": "malformed_tool_calls",
        "executed": False,
        "problems": [
            {"index": 0, "id": "bad-0", "error": "function.name is missing or blank; the only tool is bash"}
        ],
    }
    assert events(outcome, "raw.mini-swe-agent")[0]["response"] == bad
    assert [e.get("rejected", False) for e in events(outcome, "turn.completed")] == [True, False, False]
    assert events(outcome, "tool.rejected") == []


def test_the_turn_of_a_malformed_envelope_is_no_recovery_point_and_the_later_turns_are(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [envelope_fault("blank-name"), *corrected_and_submitted()])

    assert (sorted(branch_checkpoints(outcome.journal_path)), recovery_depths(tmp_path, outcome)) == (
        [1, 2],
        [1, 2],
    )


def test_the_atif_step_of_a_malformed_envelope_keeps_what_the_model_said_and_is_marked_rejected(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [envelope_fault("blank-name"), *corrected_and_submitted()])

    steps = journal_to_atif(read_journal(outcome.journal_path))["steps"]
    first = steps[0]
    assert (first["message"], first["reasoning_content"], first["tool_calls"], first.get("extra")) == (
        "A malformed call.",
        "Guess.",
        [],
        {"rejected": True},
    )
    assert [step.get("extra") for step in steps[1:]] == [None, None]


def test_the_training_export_of_a_malformed_envelope_holds_the_rejection_and_not_the_malformed_message(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [envelope_fault("blank-name"), *corrected_and_submitted()])

    messages = export_messages(tmp_path, outcome.native_session_id, "mini-swe-agent")

    assert [message["role"] for message in messages[:4]] == ["system", "user", "user", "assistant"]
    assert messages[2]["content"].startswith(
        "Tool call error: the previous response was rejected before execution because 1 of its 1 tool call(s)"
    )


def test_a_call_id_used_earlier_in_the_conversation_is_rejected_and_the_run_goes_on(
    tmp_path, sandbox: DirectorySession
):
    replies = [
        chat_completion("printf first > answer", ids=("first",)),
        chat_completion("touch must-not-execute", ids=("first",)),
        *corrected_and_submitted(),
    ]

    outcome, requests = run(tmp_path, replies)

    [rejected] = events(outcome, "mini.turn.rejected")
    assert rejected["problems"] == [
        {"index": 0, "id": "first", "error": "id was already used earlier in this conversation"}
    ]
    assert (
        [m["role"] for m in requests[2]["messages"][-3:]],
        (sandbox.root / "must-not-execute").exists(),
    ) == (
        ["assistant", "tool", "user"],
        False,
    )
    assert recovery_depths(tmp_path, outcome) == [1, 2, 3]


def repeated_faults(fault: str, count: int) -> list[dict[str, Any]]:
    """``count`` replies with a ``schema`` or an ``envelope`` fault, whose calls have distinct ids.

    A repeated id would be an envelope fault of its own, not the fault under test."""
    make = schema_fault("extra-argument") if fault == "schema" else envelope_fault("blank-name")
    replies = [deepcopy(make) for _ in range(count)]
    for index, reply in enumerate(replies):
        reply["choices"][0]["message"]["tool_calls"][0]["id"] = "bad-%d" % index
    return replies


@pytest.mark.parametrize(
    "fault,limit,overrides",
    [
        pytest.param("schema", 2, {"agent": {"max_consecutive_format_errors": 2}}, id="schema-with-override"),
        # mini's own default. Its mini.yaml sets no limit.
        pytest.param("envelope", 3, {}, id="envelope-with-default"),
    ],
)
def test_consecutive_rejections_end_the_run_at_the_format_error_limit_of_mini(
    tmp_path, sandbox: DirectorySession, fault, limit, overrides
):
    replies = [*repeated_faults(fault, limit), chat_completion("echo never", ids=("never",))]

    with provider(replies) as upstream:
        spec = rollout_spec(tmp_path, upstream.url)
        spec.extra["mini"] = overrides
        outcome = Runner().run(spec)

    assert (outcome.status, outcome.error, len(upstream.requests)) == ("error", "RepeatedFormatError", limit)
    assert (len(events(outcome, "mini.turn.rejected")), events(outcome, "tool.started")) == (limit, [])


CALL = {"id": "a", "type": "function", "function": {"name": "bash", "arguments": "{}"}}


@pytest.mark.parametrize(
    "calls,history,problems",
    [
        pytest.param(
            ["not an object"], set(), [(0, None, "the tool call is not an object")], id="not-an-object"
        ),
        pytest.param([{**CALL, "type": "tool"}], set(), [(0, "a", 'type must be "function"')], id="type"),
        pytest.param(
            [{**CALL, "function": None}],
            set(),
            [(0, "a", "function must be an object with name and arguments")],
            id="function",
        ),
        pytest.param(
            [{**CALL, "function": {"name": None, "arguments": "{}"}}],
            set(),
            [(0, "a", "function.name is missing or blank; the only tool is bash")],
            id="name",
        ),
        pytest.param(
            [{**CALL, "function": {"name": "bash", "arguments": 5}}],
            set(),
            [(0, "a", "function.arguments must be a JSON-encoded string")],
            id="arguments",
        ),
        pytest.param([{**CALL, "id": 7}], set(), [(0, None, "id is missing or blank")], id="id-not-text"),
        pytest.param(
            [CALL], {"a"}, [(0, "a", "id was already used earlier in this conversation")], id="id-in-history"
        ),
        pytest.param(
            [{**CALL, "type": "tool", "id": " "}],
            set(),
            [(0, " ", 'type must be "function"; id is missing or blank')],
            id="two-faults",
        ),
        # Object arguments (litellm serialises them), a missing or null type (the wire default is
        # "function") and ids that collide with nothing are not faults.
        pytest.param(
            [
                {**CALL, "function": {"name": "bash", "arguments": {"command": "ls"}}},
                {**CALL, "id": "b"},
                {"id": "d", "function": {"name": "bash", "arguments": "{}"}},
                {**CALL, "id": "e", "type": None},
            ],
            {"c"},
            [],
            id="no-fault",
        ),
    ],
)
def test_each_fault_of_a_tool_call_envelope_is_named(calls, history, problems):
    found = malformed_tool_calls(calls, history)

    assert [(problem["index"], problem["id"], problem["error"]) for problem in found] == problems


def test_a_well_formed_call_beside_a_malformed_one_does_not_run_and_only_the_malformed_one_is_quoted(
    tmp_path, sandbox: DirectorySession
):
    """One good call beside one bad one is the malformed response that a real model sends most."""
    mixed = chat_completion(
        "touch valid-sibling-must-not-execute", "touch must-not-execute", ids=("ok-0", "bad-1")
    )
    mixed["choices"][0]["message"]["tool_calls"][1]["function"]["name"] = ""

    outcome, requests = run(tmp_path, [mixed, *corrected_and_submitted()])

    rejection = requests[1]["messages"][-1]["content"]
    assert rejection.startswith(
        "Tool call error: the previous response was rejected before execution because 1 of its 2 tool call(s)"
    )
    assert [(call["index"], call["id"]) for call in rejected_response(rejection)["rejected_tool_calls"]] == [
        (1, "bad-1")
    ]
    assert sorted(path.name for path in sandbox.root.iterdir()) == ["answer"]


def loose_call(shape: str) -> dict[str, Any]:
    """A reply whose call writes ``fixed`` to ``answer`` in a loose but valid shape that ``shape`` names."""
    reply = chat_completion("printf fixed > answer", ids=("loose",))
    call = reply["choices"][0]["message"]["tool_calls"][0]
    if shape == "object-arguments":
        call["function"]["arguments"] = {"command": "printf fixed > answer"}
    elif shape == "no-type":
        del call["type"]
    elif shape == "null-type":
        call["type"] = None
    return reply


@pytest.mark.parametrize("shape", ["object-arguments", "no-type", "null-type"])
def test_object_arguments_and_a_missing_or_null_type_run_as_a_well_formed_call(
    tmp_path, sandbox: DirectorySession, shape
):
    """litellm reads them so, and such calls ran before the envelope was checked."""
    outcome, requests = run(tmp_path, [loose_call(shape), chat_completion(SUBMIT, ids=("submit",))])

    echoed = requests[1]["messages"][-2]["tool_calls"][0]
    assert (
        (sandbox.root / "answer").read_text(),
        echoed["type"],
        json.loads(echoed["function"]["arguments"]),
    ) == (
        "fixed",
        "function",
        {"command": "printf fixed > answer"},
    )
    assert events(outcome, "mini.turn.rejected", "tool.rejected") == []


def no_tool_call() -> dict[str, Any]:
    """A reply that says something and calls no tool, as a model that thinks it is done."""
    reply = chat_completion(content="I believe the task is already done.", reasoning="Nothing left.")
    reply["choices"][0]["finish_reason"] = "stop"
    return reply


def test_a_response_without_a_tool_call_stays_in_the_conversation_with_the_format_error_of_mini(
    tmp_path, sandbox: DirectorySession
):
    outcome, requests = run(tmp_path, [no_tool_call(), *corrected_and_submitted()])

    messages = requests[1]["messages"]
    assert ([m["role"] for m in messages], messages[2]["content"]) == (
        ["system", "user", "assistant", "user"],
        "I believe the task is already done.",
    )
    assert messages[3]["content"].startswith(
        "Tool call error:\n\n<error>\nNo tool calls found in the response."
    )
    assert (outcome.status, (sandbox.root / "answer").read_text()) == ("completed", "fixed")


def test_a_response_without_a_tool_call_is_journaled_as_a_rejected_turn_and_is_no_recovery_point(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [no_tool_call(), *corrected_and_submitted()])

    [rejected] = events(outcome, "mini.turn.rejected")
    assert (rejected["source"], rejected["executed"], events(outcome, "agent.message")[0]["text"]) == (
        "no_tool_calls",
        False,
        "I believe the task is already done.",
    )
    assert [e.get("rejected", False) for e in events(outcome, "turn.completed")] == [True, False, False]
    assert recovery_depths(tmp_path, outcome) == [1, 2]


def test_a_history_whose_rejection_the_journal_does_not_confirm_has_no_recovery_points(
    tmp_path, sandbox: DirectorySession
):
    outcome, _ = run(tmp_path, [schema_fault("extra-argument"), *corrected_and_submitted()])
    drop_events(outcome.journal_path, "mini.turn.rejected")

    assert recovery_points(tmp_path, outcome) == []
