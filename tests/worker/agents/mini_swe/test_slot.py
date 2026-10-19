"""``MiniSweSlot`` runs mini-swe-agent on a task through the run's ``ToolAccess``, and says how it ended.

Each test runs the slot as the runner does (``rollouts.run_mini``), against a scripted model. The
agent's commands run with bash in a directory of the test.
"""

from __future__ import annotations

import json
import os
import re
import threading
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import read_entries
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import read_journal
from sprout.worker.sandbox.image_shell import ImageShell
from sprout.worker.tools.result import CommandOutcome, ToolResult
from tests.support.http import chat_completion
from tests.worker.agents.mini_swe.rollouts import (
    BASH_TOOL,
    SUBMIT,
    ShellSandbox,
    needs_mini,
    outline,
    run_mini,
    text_reply,
    with_tool_name,
)

pytestmark = needs_mini

#: mini's environment around each agent command (mini 2.4.6, ``config/mini.yaml``).
MINI_ENV = "env PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1"
#: A reviewer's assistant turn in the form that its author writes.
TURN = {
    "role": "assistant",
    "content": "Check the boundary first.",
    "tool_calls": [
        {
            "id": "reviewer-1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "true"}'},
        }
    ],
}


def first_lines(messages: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """The role and the first line of the content of each message."""
    return [(message["role"], message["content"].splitlines()[0]) for message in messages]


def not_installed(name: str) -> str:
    raise PackageNotFoundError(name)


def last_entries(home: Path) -> list[dict[str, Any]]:
    """The entries of the one native history under ``home``."""
    (transcript,) = home.glob("*.jsonl")
    return read_entries(transcript.read_bytes())


def observations(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool messages at the end of a model request, each with its content decoded from JSON."""
    tail = []
    for message in reversed(request["messages"]):
        if message["role"] != "tool":
            break
        tail.insert(0, {**message, "content": json.loads(message["content"])})
    return tail


# --- how a run ends ---------------------------------------------------------------------------


def test_a_run_that_submits_completes_with_the_output_after_the_submit_line(tmp_path):
    rollout = run_mini(
        tmp_path, [chat_completion("printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\nthe patch\\n'")]
    )

    assert (rollout.result.status, rollout.result.final_text, rollout.result.error) == (
        "completed",
        "the patch\n",
        None,
    )


def test_a_run_that_mini_ends_without_a_submission_is_an_error(tmp_path):
    """mini gives up after three responses in a row without a tool call."""
    replies = [text_reply("I am thinking."), text_reply("Still thinking."), text_reply("Done thinking.")]

    rollout = run_mini(tmp_path, replies)

    assert (rollout.result.status, rollout.result.error) == ("error", "RepeatedFormatError")


@pytest.mark.parametrize(
    "stop_reason, status",
    [
        ("timeout", "timeout"),
        ("max_turns_reached", "timeout"),
        ("context_exhausted", "timeout"),
        (None, "killed"),
    ],
)
def test_a_stop_for_a_spent_budget_ends_gradable_and_any_other_stop_kills_the_run(
    tmp_path, stop_reason, status
):
    control = RunControl()
    sandbox = ShellSandbox(
        tmp_path / "sandbox", on_call=lambda args: control.request_stop("the reason", stop_reason=stop_reason)
    )

    rollout = run_mini(tmp_path, [chat_completion("printf work > answer")], sandbox=sandbox, control=control)

    assert (rollout.result.status, rollout.result.error) == (status, "the reason")


def test_a_run_that_outlives_its_time_stops_as_a_gradable_timeout(tmp_path):
    control = RunControl()
    stopped = threading.Event()
    control.subscribe(stopped.set)
    sandbox = ShellSandbox(tmp_path / "sandbox", on_call=lambda args: stopped.wait(10))

    rollout = run_mini(tmp_path, [chat_completion("true")], sandbox=sandbox, control=control, timeout_s=0.2)

    assert (rollout.result.status, rollout.result.error, control.stop_reason) == (
        "timeout",
        "mini wall-time budget exhausted",
        "timeout",
    )


def test_a_command_whose_outcome_is_unknown_ends_the_run_before_another_model_request(tmp_path):
    """Nothing after such a command pairs with a known filesystem."""
    still_running = ToolResult(False, "", outcome=CommandOutcome(running=True))
    sandbox = ShellSandbox(tmp_path / "sandbox", scripted={"make build": still_running})

    rollout = run_mini(tmp_path, [chat_completion("make build"), chat_completion(SUBMIT)], sandbox=sandbox)

    assert (rollout.result.status, rollout.result.error, len(rollout.requests)) == (
        "error",
        "ToolInfraError: tool execution did not settle: "
        "{'kind': 'runtime_still_running', 'running': True, 'timed_out': False, 'exit_code': None}",
        1,
    )


def test_the_usage_of_a_run_is_the_sum_of_the_usage_of_its_responses(tmp_path):
    rollout = run_mini(tmp_path, [chat_completion("true"), chat_completion(SUBMIT)])

    assert rollout.result.usage == {
        "input_tokens": 22,
        "output_tokens": 14,
        "cached_input_tokens": 6,
        "cache_creation_tokens": 0,
        "reasoning_output_tokens": 10,
        "cost_usd": 0.0,
    }


# --- what runs in the sandbox -----------------------------------------------------------------


@pytest.mark.parametrize(
    "environment, timeout", [({}, 600), ({"timeout": 45}, 45)], ids=["default-timeout", "configured-timeout"]
)
def test_each_bash_action_is_one_shell_call_in_the_workspace_with_mini_s_environment(
    tmp_path, environment, timeout
):
    rollout = run_mini(
        tmp_path,
        [chat_completion("printf first > answer"), chat_completion(SUBMIT)],
        extra={"mini": {"environment": environment}},
    )

    assert rollout.sandbox.calls[0] == {
        "command": MINI_ENV + " bash -c 'printf first > answer'",
        "working_dir": "/testbed",
        "timeout": timeout,
    }


def test_with_the_image_shell_each_action_runs_as_the_images_user(tmp_path):
    shell = ImageShell(user="app", env={"LANG": "C.UTF-8"}, workdir="/")
    line = shell.command(MINI_ENV + " bash -c '%s'" % SUBMIT, working_dir="/testbed")
    sandbox = ShellSandbox(
        tmp_path / "sandbox", scripted={line: ToolResult(True, "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n")}
    )

    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)], sandbox=sandbox, image_shell=shell)

    assert (rollout.sandbox.calls, rollout.result.status) == (
        [{"command": line, "working_dir": "/testbed", "timeout": 600}],
        "completed",
    )


def test_the_calls_after_the_submitting_call_do_not_run_and_get_a_not_executed_result(tmp_path):
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT, "touch late")])

    assert rollout.sandbox.commands == [SUBMIT]
    assert rollout.entries()[-3]["message"] == {
        "role": "tool",
        "tool_call_id": "call-2",
        "content": "Not executed: the task was already submitted by an earlier call in this response.",
    }


def test_a_response_that_submits_before_its_last_call_still_closes_its_turn(tmp_path):
    """The final snapshot is the last turn's branch point (``ToolAccess.finalize``)."""
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT, "touch late")])

    assert outline(rollout.entries())[-4:] == [
        ("message", "tool", "call-1"),
        ("message", "tool", "call-2"),
        ("turn", ["call-1", "call-2"]),
        ("message", "exit"),
    ]


# --- what the model sees ----------------------------------------------------------------------


def test_a_failed_command_reaches_the_model_with_its_exit_code_and_both_streams(tmp_path):
    replies = [chat_completion("printf diagnostic; printf ' and error' >&2; exit 7"), chat_completion(SUBMIT)]

    rollout = run_mini(tmp_path, replies)

    assert observations(rollout.requests[1]) == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": {"returncode": 7, "output": "diagnostic and error"},
        }
    ]


def test_a_failed_call_does_not_stop_the_later_calls_of_its_response(tmp_path):
    rollout = run_mini(tmp_path, [chat_completion("exit 3", "printf after"), chat_completion(SUBMIT)])

    assert rollout.sandbox.commands == ["exit 3", "printf after", SUBMIT]
    assert observations(rollout.requests[1]) == [
        {"role": "tool", "tool_call_id": "call-1", "content": {"returncode": 3, "output": ""}},
        {"role": "tool", "tool_call_id": "call-2", "content": {"returncode": 0, "output": "after"}},
    ]


def test_a_command_that_the_runtime_timed_out_reaches_the_model_with_its_partial_output(tmp_path):
    timed_out = CommandOutcome(exit_code=137, stdout="partial", stderr=" output", timed_out=True)
    sandbox = ShellSandbox(
        tmp_path / "sandbox", scripted={"make test": ToolResult(False, "", outcome=timed_out)}
    )

    rollout = run_mini(tmp_path, [chat_completion("make test"), chat_completion(SUBMIT)], sandbox=sandbox)

    assert observations(rollout.requests[1]) == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": {"returncode": 137, "output": "partial output", "exception_info": "Command timed out"},
        }
    ]


def test_a_response_that_breaks_the_bash_schema_runs_no_call_and_gets_a_tool_error_for_each_call(tmp_path):
    invalid = with_tool_name(chat_completion("ls", "pwd"), index=0, name="shell")

    rollout = run_mini(tmp_path, [invalid, chat_completion(SUBMIT)])

    instruction = "No calls in this response were executed. Correct and resend the tool calls."
    assert rollout.sandbox.commands == [SUBMIT]
    assert observations(rollout.requests[1]) == [
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": {
                "error": {
                    "type": "tool_schema_error",
                    "message": "Unknown tool. The available function is bash(command).",
                },
                "executed": False,
                "instruction": instruction,
                "available_tools": [BASH_TOOL],
            },
        },
        {
            "role": "tool",
            "tool_call_id": "call-2",
            "content": {
                "error": {
                    "type": "tool_batch_rejected",
                    "message": "Another call in this response failed schema validation.",
                },
                "executed": False,
                "instruction": instruction,
                "available_tools": [BASH_TOOL],
            },
        },
    ]


def test_a_response_with_a_malformed_tool_call_gets_feedback_and_leaves_the_conversation(tmp_path):
    """A strict provider refuses a call without an id when it is sent back, so the response is dropped."""
    rollout = run_mini(tmp_path, [chat_completion("ls", ids=("",)), chat_completion(SUBMIT)])

    first, second = rollout.requests
    assert rollout.sandbox.commands == [SUBMIT]
    assert second["messages"][:-1] == first["messages"]
    assert first_lines(second["messages"][-1:]) == [
        (
            "user",
            "Tool call error: the previous response was rejected before execution because 1 of its 1 "
            "tool call(s) were malformed. None of them ran, and the response is not part of the "
            "conversation.",
        )
    ]


def test_a_response_without_a_tool_call_stays_in_the_conversation_before_the_format_feedback(tmp_path):
    rollout = run_mini(tmp_path, [text_reply("I will look first."), chat_completion(SUBMIT)])

    first, second = rollout.requests
    assert first_lines(second["messages"][len(first["messages"]) :]) == [
        ("assistant", "I will look first."),
        ("user", "Tool call error:"),
    ]


@pytest.mark.parametrize(
    "repository, workspace_lines",
    [
        (None, "Default working directory for every bash call: /testbed\n"),
        ("/testbed", "Default working directory for every bash call: /testbed\nRepository root: /testbed\n"),
    ],
    ids=["outside-a-repository", "repository-root"],
)
def test_the_system_prompt_names_the_workspace_and_its_repository(tmp_path, repository, workspace_lines):
    sandbox = ShellSandbox(tmp_path / "sandbox", repository=repository)

    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)], sandbox=sandbox)

    assert rollout.requests[0]["messages"][0] == {
        "role": "system",
        "content": "You are a helpful assistant that can interact with a computer.\n\n\n<workspace>\n"
        + workspace_lines
        + "Each call runs in a new shell. Changes made by cd or export do not persist across calls.\n"
        + "</workspace>",
    }


def test_the_prompt_names_no_kernel_or_machine_of_the_host(tmp_path):
    """The prompt is the same on each host."""
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)])

    assert (
        "<system_information>\nLinux   \n</system_information>"
        in rollout.requests[0]["messages"][1]["content"]
    )


# --- the native history -----------------------------------------------------------------------


def test_each_message_is_in_the_native_history_before_the_next_command_runs(tmp_path):
    seen: list[tuple[Any, ...]] = []
    home = tmp_path / "native-home"
    sandbox = ShellSandbox(
        tmp_path / "sandbox", on_call=lambda args: seen.append(outline(last_entries(home))[-1])
    )

    run_mini(tmp_path, [chat_completion("true", "true"), chat_completion(SUBMIT)], sandbox=sandbox)

    assert seen == [
        ("message", "assistant", ["call-1", "call-2"]),
        ("message", "tool", "call-1"),
        ("message", "assistant", ["call-3"]),
    ]


def test_a_turn_closes_in_the_native_history_after_every_observation_of_its_response(tmp_path):
    rollout = run_mini(tmp_path, [chat_completion("true", "true"), chat_completion(SUBMIT)])

    assert outline(rollout.entries()) == [
        ("mini.session",),
        ("message", "system"),
        ("message", "user"),
        ("message", "assistant", ["call-1", "call-2"]),
        ("message", "tool", "call-1"),
        ("message", "tool", "call-2"),
        ("turn", ["call-1", "call-2"]),
        ("message", "assistant", ["call-3"]),
        ("message", "tool", "call-3"),
        ("turn", ["call-3"]),
        ("message", "exit"),
    ]


def test_the_journal_names_the_native_history_of_the_session_under_the_native_home(tmp_path):
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)])

    session_id = rollout.result.native_session_id
    assert rollout.transcript == tmp_path / "native-home" / ("%s.jsonl" % session_id)
    assert rollout.entries()[0] == {
        "type": "mini.session",
        "format": "sprout-mini-v1",
        "mini_version": "2.4.6",
        "session_id": session_id,
        "workspace": {"cwd": "/testbed", "repository_dir": None},
    }


# --- refusals ---------------------------------------------------------------------------------

#: The message of the refused environment values.
ENVIRONMENT = "mini requires a discovered or explicit absolute sandbox cwd, positive timeout and text env"
PREFIX = "This branch mode requires an exact mini history prefix"


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"mini": {"tools": {}}}, "mini config accepts agent/environment/model overrides"),
        (
            {"mini": {"agent": {"step_limit": 5}}},
            "Use Sprout rollout controls instead of mini agent.step_limit",
        ),
        (
            {"mini": {"agent": {"cost_limit": 1.0}}},
            "Use Sprout rollout controls instead of mini agent.cost_limit",
        ),
        (
            {"mini": {"agent": {"wall_time_limit_seconds": 60}}},
            "Use Sprout rollout controls instead of mini agent.wall_time_limit_seconds",
        ),
        (
            {"mini": {"agent": {"output_path": "/path/to/trajectory.json"}}},
            "Use Sprout rollout controls instead of mini agent.output_path",
        ),
        ({"mini": {"environment": {"shell": "zsh"}}}, "mini environment only accepts cwd, timeout and env"),
        ({"mini": {"environment": {"timeout": 0}}}, ENVIRONMENT),
        ({"mini": {"environment": {"timeout": True}}}, ENVIRONMENT),
        ({"mini": {"environment": {"timeout": 1.5}}}, ENVIRONMENT),
        ({"mini": {"environment": {"env": {"DEBUG": 1}}}}, ENVIRONMENT),
        (
            {"mini": {"model": {"model_kwargs": {"seed": 1}}}},
            "Unsupported mini model_kwargs; parameters must reach the model unchanged",
        ),
        ({"resume_without_hint": "yes"}, "resume_without_hint must be a boolean"),
        ({"resume_without_hint": True}, PREFIX),
        ({"assistant_turn": TURN}, PREFIX),
        (
            {"assistant_turn": TURN, "resume_without_hint": True},
            "Point-only continuation cannot include an assistant_turn",
        ),
    ],
    ids=[
        "unknown-section",
        "step-limit",
        "cost-limit",
        "wall-time-limit",
        "output-path",
        "unknown-environment-key",
        "zero-timeout",
        "boolean-timeout",
        "fractional-timeout",
        "non-text-env",
        "unknown-model-kwarg",
        "non-boolean-resume",
        "resume-without-prefix",
        "turn-without-prefix",
        "turn-and-resume",
    ],
)
def test_the_slot_refuses_an_invalid_configuration_before_the_run_starts(tmp_path, extra, message):
    with pytest.raises(ValueError, match="^%s$" % re.escape(message)):
        run_mini(tmp_path, [], extra=extra)

    assert [record["type"] for record in read_journal(tmp_path / "journal.jsonl")] == ["mini.workspace"]


@pytest.mark.parametrize(
    "version, installed",
    [(lambda name: "2.4.7", "2.4.7"), (not_installed, None)],
    ids=["other-version", "not-installed"],
)
def test_the_slot_refuses_any_mini_but_the_pinned_version(tmp_path, monkeypatch, version, installed):
    monkeypatch.setattr("sprout.worker.agents.mini_swe.slot.version", version)

    with pytest.raises(
        ValueError, match=re.escape("mini-swe-agent 2.4.6 required (installed: %s)" % installed)
    ):
        run_mini(tmp_path, [chat_completion(SUBMIT)])


def test_a_run_leaves_the_process_environment_as_it_found_it(tmp_path, monkeypatch):
    """The slot sets mini's settings for mini's first import only."""
    monkeypatch.delenv("MSWEA_GLOBAL_CONFIG_DIR", raising=False)
    monkeypatch.setenv("MSWEA_SILENT_STARTUP", "0")

    run_mini(tmp_path, [chat_completion(SUBMIT)])

    assert (os.environ.get("MSWEA_GLOBAL_CONFIG_DIR"), os.environ.get("MSWEA_SILENT_STARTUP")) == (None, "0")
