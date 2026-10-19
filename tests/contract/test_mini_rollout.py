"""A mini rollout from a ``RunSpec``: what the worker sends to the model, and what the run records.

Each test runs the real mini loop in the worker's ``Runner``, against a scripted chat endpoint and the
fake sandbox backend of ``fake_sandbox.py``. The native history and the snapshots are the contract formats
that a branch and a training export read.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from typing import Any

import pytest

from sprout.contract.messages import clean_messages
from sprout.contract.mini_history import training_messages
from sprout.worker.agents.mini_swe.history import actor_tools_at, available_branch_points
from sprout.worker.checkpoint.rollback import branch_checkpoints
from sprout.worker.jobs.review.branching import reviewer_context
from sprout.worker.run.journal import JournalWriter
from sprout.worker.run.runner import Runner
from sprout.worker.tools.result import CommandOutcome
from tests.contract.builders import bash_tools
from tests.contract.mini_runs import NEEDS_MINI, SUBMIT, events, mini_spec, native_entries, rollout_contract
from tests.support.http import chat_completion, chat_server

pytestmark = NEEDS_MINI

#: sglang's refusal of a prompt and a completion budget that exceed the context window.
CONTEXT_REFUSAL = (
    400,
    {
        "object": "error",
        "type": "BadRequestError",
        "code": 400,
        "message": "Requested token count exceeds the model's maximum context length of 262144 tokens. "
        "You requested a total of 262203 tokens.",
    },
)
#: ``BASH_TOOL`` of mini-swe-agent 2.4.6, which leaves ``additionalProperties`` out.
MINI_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}
#: A ``runuser`` that runs the command after ``--`` with ``RUNUSER`` set to the user of ``-u``.
FAKE_RUNUSER = """#!/bin/sh
while [ "$1" != "--" ]; do
    [ "$1" = "-u" ] && user="$2"
    shift
done
shift
RUNUSER="$user" exec "$@"
"""
#: A ``getent passwd`` that knows every user.
FAKE_GETENT = """#!/bin/sh
echo "$2:x:1000:1000::/home/$2:/bin/sh"
"""


def three_turn_rollout() -> list[dict[str, Any]]:
    """Replies that write ``first``, then ``second``, to ``answer``, and then submit."""
    return [
        chat_completion("printf first > answer", "cat answer"),
        chat_completion("printf second > answer"),
        chat_completion(SUBMIT),
    ]


def observation(request: dict[str, Any]) -> dict[str, Any]:
    """What the last tool message of a model request tells the model."""
    return json.loads(request["messages"][-1]["content"])


def distinct(values: Iterable[Any]) -> list[Any]:
    """Each of the JSON ``values`` once, in the order of its first occurrence."""
    texts = dict.fromkeys(json.dumps(value, sort_keys=True) for value in values)
    return [json.loads(text) for text in texts]


@pytest.fixture
def image_users(tmp_path, monkeypatch):
    """The fake ``runuser`` and ``getent``, first on PATH."""
    tools = tmp_path / "bin"
    tools.mkdir()
    for name, script in (("runuser", FAKE_RUNUSER), ("getent", FAKE_GETENT)):
        (tools / name).write_text(script)
        (tools / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}{os.pathsep}{os.environ['PATH']}")


def test_each_closed_turn_has_a_snapshot_of_its_files_and_the_last_one_is_final(tmp_path, sandboxes):
    """A turn's snapshot is taken when the next turn starts, so it holds the files at the end of the turn."""
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    points = branch_checkpoints(outcome.journal_path)

    assert {
        step: (sandboxes.snapshots[point.snapshot_id], point.final) for step, point in points.items()
    } == {
        2: ({"answer": b"first"}, False),
        3: ({"answer": b"second"}, False),
        4: ({"answer": b"second"}, True),
    }
    assert outcome.final_snapshot_id == points[4].snapshot_id


def test_a_run_destroys_its_sandbox_when_it_ends(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    assert [session.destroyed for session in sandboxes.sessions] == [True]


def test_each_model_request_carries_the_sampling_controls_as_chat_completions_fields(tmp_path, sandboxes):
    sampling = {"temperature": 0.8, "top_k": 20, "max_new_tokens": 300}

    with chat_server(three_turn_rollout()) as server:
        contract = rollout_contract(server.url, sampling_params=sampling)
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url, contract=contract))

    fields = [
        {key: request.get(key) for key in ("temperature", "top_k", "max_tokens", "max_new_tokens")}
        for request in server.requests
    ]

    assert fields == [{"temperature": 0.8, "top_k": 20, "max_tokens": 300, "max_new_tokens": None}] * 3


def test_each_model_request_and_the_journal_offer_the_pinned_bash_tool(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    offered = [request["tools"] for request in server.requests]
    recorded = distinct(event["tools"] for event in events(outcome.journal_path, "rollout.model_tools"))

    assert (offered, recorded) == ([bash_tools()] * 3, [bash_tools()])


def test_each_model_request_extends_the_messages_of_the_one_before(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    first, second, third = (request["messages"] for request in server.requests)

    assert (second[: len(first)], third[: len(second)]) == (first, second)


def test_the_system_message_names_the_default_working_directory(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    system = server.requests[0]["messages"][0]

    assert "\n<workspace>\nDefault working directory for every bash call: /testbed\n" in system["content"]


def test_the_native_history_records_the_workspace_of_the_session(tmp_path, sandboxes):
    """A branch restores this workspace and refuses to run in any other."""
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    session = native_entries(tmp_path / "native-home", outcome.native_session_id)[0]

    assert (session["type"], session["workspace"]) == (
        "mini.session",
        {"cwd": "/testbed", "repository_dir": None},
    )


def test_the_native_history_is_a_training_conversation_with_the_reasoning_of_each_turn(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    messages = clean_messages(
        training_messages(native_entries(tmp_path / "native-home", outcome.native_session_id))
    )

    assert [(message["role"], message.get("reasoning_content")) for message in messages] == [
        ("system", None),
        ("user", None),
        ("assistant", "Inspect the actual file."),
        ("tool", None),
        ("tool", None),
        ("assistant", "Inspect the actual file."),
        ("tool", None),
        ("assistant", "Inspect the actual file."),
        ("tool", None),
    ]


def test_the_usage_of_a_run_is_the_sum_over_its_model_requests(tmp_path, sandboxes):
    """Each reply reports 11 prompt tokens, 3 of them cached, and 7 completion tokens, 5 of them reasoning."""
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    assert outcome.usage == {
        "input_tokens": 33,
        "output_tokens": 21,
        "cached_input_tokens": 9,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 15,
        "cost_usd": 0.0,
    }


@pytest.mark.parametrize(
    "max_turns, replies, stop_reason",
    [
        pytest.param(1, [chat_completion("echo fix > answer")], "max_turns_reached", id="turn-budget"),
        pytest.param(
            5, [chat_completion("echo fix > answer"), CONTEXT_REFUSAL], "context_exhausted", id="context"
        ),
    ],
)
def test_a_rollout_out_of_turns_or_context_ends_as_gradable_without_another_request(
    tmp_path, sandboxes, max_turns, replies, stop_reason
):
    """The gateway refuses the request past the turn budget. The model refuses a request past its window."""
    with chat_server(replies) as server:
        spec = mini_spec(tmp_path, server.url, contract=rollout_contract(server.url, max_turns=max_turns))
        outcome = Runner(out_dir=tmp_path).run(spec)

    assert (outcome.status, outcome.stop_reason, len(server.requests)) == (
        "timeout",
        stop_reason,
        len(replies),
    )


def test_the_final_snapshot_of_a_rollout_out_of_context_is_its_last_closed_turn(tmp_path, sandboxes):
    with chat_server([chat_completion("echo fix > answer"), CONTEXT_REFUSAL]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    points = branch_checkpoints(outcome.journal_path)

    assert (list(points), outcome.final_snapshot_id, sandboxes.snapshots[outcome.final_snapshot_id]) == (
        [1],
        points[1].snapshot_id,
        {"answer": b"fix\n"},
    )


def test_any_other_refused_model_request_ends_the_run_as_an_error(tmp_path, sandboxes):
    refusal = (400, {"error": {"message": "budget exhausted for this run", "type": "invalid_request_error"}})

    with chat_server([chat_completion("echo fix > answer"), refusal]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    assert (outcome.status, outcome.stop_reason) == ("error", None)


def test_a_working_directory_that_the_sandbox_lacks_fails_the_run_before_any_model_request(
    tmp_path, sandboxes
):
    with chat_server([]) as server:
        spec = mini_spec(tmp_path, server.url)
        spec.extra["mini"] = {"environment": {"cwd": "/missing-workspace"}}
        outcome = Runner(out_dir=tmp_path).run(spec)

    assert (outcome.error, server.requests, events(outcome.journal_path, "tool.started")) == (
        "ValueError: mini working-directory probe failed: "
        "failed to start shell: chdir /missing-workspace: no such file or directory",
        [],
        [],
    )


def test_without_turn_checkpoints_the_run_still_takes_its_final_disk_snapshot(tmp_path, sandboxes):
    replies = [chat_completion("echo fixed > answer"), chat_completion("echo more >> answer")]

    with chat_server(replies) as server:
        spec = mini_spec(
            tmp_path, server.url, contract=rollout_contract(server.url, max_turns=2), checkpoint_enabled=False
        )
        outcome = Runner(out_dir=tmp_path).run(spec)

    captured = [
        (event["final"], event["disk_only"]) for event in events(outcome.journal_path, "checkpoint.captured")
    ]
    assert (captured, sandboxes.snapshots[outcome.final_snapshot_id]) == (
        [(True, True)],
        {"answer": b"fixed\nmore\n"},
    )


def test_with_the_image_user_each_action_runs_as_that_user_with_the_image_env(
    tmp_path, sandboxes, image_users
):
    """mini's own environment, such as its ``PAGER``, still applies inside the image's."""
    sandboxes.image_config = {"User": "app", "Env": ["FOO=from the image"], "WorkingDir": "/testbed"}
    backend = {"backend": "podman", "podman": {"image_user": True, "image_env": True}}

    with chat_server([chat_completion('echo "$RUNUSER|$FOO|$PAGER"'), chat_completion(SUBMIT)]) as server:
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url, backend=backend))

    assert observation(server.requests[1]) == {"returncode": 0, "output": "app|from the image|cat\n"}


@pytest.mark.parametrize(
    "answer, kind",
    [
        pytest.param(TimeoutError("the runtime did not answer"), "transport_exception", id="transport"),
        pytest.param(CommandOutcome(timed_out=True), "runtime_timeout", id="timeout-without-exit-code"),
        pytest.param(
            CommandOutcome(timed_out=True, running=True, exit_code=137), "runtime_timeout", id="still-killing"
        ),
        pytest.param(CommandOutcome(running=True), "runtime_still_running", id="still-running"),
    ],
)
def test_a_call_whose_outcome_is_unknown_stops_the_run_before_another_request_or_call(
    tmp_path, sandboxes, answer, kind
):
    sandboxes.unsettle(answer)

    with chat_server([chat_completion("echo unsettled"), chat_completion(SUBMIT)]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    uncertain = [
        (event["kind"], event["step"]) for event in events(outcome.journal_path, "execution.uncertain")
    ]
    assert (outcome.status, len(server.requests), sandboxes.sessions[0].calls, uncertain) == (
        "error",
        1,
        1,
        [(kind, 1)],
    )


def test_a_state_reached_through_an_unknown_outcome_is_no_branch_point(tmp_path, sandboxes):
    """The run still takes its final snapshot, and marks it ``uncertain``."""
    sandboxes.unsettle(CommandOutcome(running=True))

    with chat_server([chat_completion("echo unsettled"), chat_completion(SUBMIT)]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    final = [
        (event["final"], event["uncertain"]) for event in events(outcome.journal_path, "checkpoint.captured")
    ]
    assert (final, branch_checkpoints(outcome.journal_path)) == ([(True, True)], {})


@pytest.mark.parametrize(
    "command, seen",
    [
        pytest.param(
            "printf diagnostic; exit 42", {"returncode": 42, "output": "diagnostic"}, id="exit-code"
        ),
        pytest.param(
            "printf partial > answer; printf 'partial output'; printf 'partial stderr' >&2; : partial-write",
            {
                "returncode": 137,
                "output": "partial outputpartial stderr",
                "exception_info": "Command timed out",
            },
            id="settled-timeout",
        ),
    ],
)
def test_a_command_that_settled_without_success_is_an_observation_for_the_next_request(
    tmp_path, sandboxes, command, seen
):
    sandboxes.time_out("partial-write")

    with chat_server([chat_completion(command), chat_completion(SUBMIT)]) as server:
        Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    assert observation(server.requests[1]) == seen


@pytest.mark.parametrize(
    "command, files",
    [
        pytest.param("printf diagnostic; exit 42", {}, id="exit-code"),
        pytest.param(
            "printf partial > answer; : partial-write", {"answer": b"partial"}, id="settled-timeout"
        ),
    ],
)
def test_the_turn_of_a_command_that_settled_without_success_is_a_branch_point(
    tmp_path, sandboxes, command, files
):
    sandboxes.time_out("partial-write")

    with chat_server([chat_completion(command), chat_completion(SUBMIT)]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    assert sandboxes.snapshots[branch_checkpoints(outcome.journal_path)[1].snapshot_id] == files


def test_the_journal_records_the_prompt_of_the_run_as_its_task(tmp_path, sandboxes):
    with chat_server([chat_completion(SUBMIT)]) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    [started] = events(outcome.journal_path, "run.started")

    assert started["task_prompt"] == "Write the answer."


def test_a_reviewer_reads_the_tools_and_the_workspace_of_the_actor_at_a_point(tmp_path, sandboxes):
    with chat_server(three_turn_rollout()) as server:
        outcome = Runner(out_dir=tmp_path).run(mini_spec(tmp_path, server.url))

    context = reviewer_context(outcome.journal_path, {2: available_branch_points(outcome.journal_path)[2]})

    assert (context["tools"], context["workspace"]) == (
        bash_tools(),
        {"cwd": "/testbed", "repository_dir": None},
    )


def test_a_tool_record_after_a_turn_leaves_the_tools_of_that_turn(tmp_path):
    """A turn's tools are the ones that its model request offered, whatever the actor got later."""
    with JournalWriter(tmp_path / "trajectory.jsonl") as journal:
        journal.emit("rollout.model_tools", shape="chat/completions", tools=bash_tools())
        journal.emit("model.turn.completed", turn_id="turn-1", call_ids=["call-1"], step=2, source="actor")
        journal.emit("rollout.model_tools", shape="chat/completions", tools=[])

    assert actor_tools_at(tmp_path / "trajectory.jsonl", 2) == bash_tools()


def test_without_a_tool_record_before_a_turn_the_reviewer_gets_no_tools(tmp_path):
    """The reviewer must not guess the actor's tools from the defaults of this host."""
    with JournalWriter(tmp_path / "trajectory.jsonl") as journal:
        journal.emit("model.turn.completed", turn_id="turn-1", call_ids=["call-1"], step=2, source="actor")
        journal.emit("rollout.model_tools", shape="chat/completions", tools=bash_tools())

    with pytest.raises(ValueError, match="^No recorded actor tool schema at the selected checkpoint$"):
        actor_tools_at(tmp_path / "trajectory.jsonl", 2)


def test_the_pinned_tool_is_the_bash_tool_of_the_installed_mini(tmp_path, monkeypatch):
    """Up to ``additionalProperties``. The pinned tool sets it: the executor refuses extra arguments."""
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    from sprout.worker.agents.mini_swe.tools import mini_tool_schema

    assert mini_tool_schema(BASH_TOOL) == bash_tools()


def test_comparing_the_installed_tool_with_the_pinned_one_leaves_the_installed_tool_as_it_was(
    tmp_path, monkeypatch
):
    """mini's ``BASH_TOOL`` is a global of the process. Each run of mini compares it, earlier ones too."""
    monkeypatch.setenv("MSWEA_GLOBAL_CONFIG_DIR", str(tmp_path / "mini-config"))
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "1")
    from minisweagent.models.utils.actions_toolcall import BASH_TOOL

    from sprout.worker.agents.mini_swe.tools import mini_tool_schema

    mini_tool_schema(BASH_TOOL)

    assert BASH_TOOL == MINI_BASH_TOOL


def test_reading_the_pinned_tool_needs_neither_mini_nor_its_host_config(tmp_path):
    """An evaluation client reads the pinned tool on a host without mini, and without its global ``.env``."""
    config = tmp_path / "not-created"
    code = (
        "import json, sys\n"
        "from sprout.worker.agents.mini_swe.tools import mini_tool_schema\n"
        "print(json.dumps([mini_tool_schema(), 'minisweagent' in sys.modules]))\n"
    )

    done = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "MSWEA_GLOBAL_CONFIG_DIR": str(config)},
    )

    assert (json.loads(done.stdout), config.exists()) == ([bash_tools(), False], False)
