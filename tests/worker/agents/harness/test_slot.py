"""The harness slot end to end, against a scripted model and a sandbox of real files: what each turn runs,
what the journal and the native history keep, and how each stop condition ends the rollout."""

from __future__ import annotations

import json
from pathlib import Path

from sprout.contract.harness_history import training_messages
from sprout.contract.messages import clean_messages
from sprout.worker.agents.harness.history import index_harness
from sprout.worker.run.journal import journal_events
from tests.worker.agents.harness.rollouts import reply, run_harness, turn

BUGGY = "def add(a, b):\n    return a - b\n"
FIX = (
    reply(turn("execute_bash", ("command", "cat calc.py"))),
    reply(turn("str_replace", ("file_path", "calc.py"), ("old_string", "a - b"), ("new_string", "a + b"))),
    reply(turn("execute_bash", ("command", "python -c 'import calc; print(calc.add(2, 3))'"))),
    reply(turn("submit")),
)


def fixture(tmp_path: Path) -> Path:
    (tmp_path / "testbed").mkdir()
    (tmp_path / "testbed" / "calc.py").write_text(BUGGY)
    return tmp_path


def test_a_rollout_reads_edits_runs_and_submits_one_call_a_turn(tmp_path: Path) -> None:
    rollout = run_harness(fixture(tmp_path), list(FIX))

    assert rollout.result.status == "completed"
    assert (tmp_path / "testbed" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    assert [name for name, _ in rollout.sandbox.calls] == ["shell", "text_editor", "shell"]
    assert [call["command"] for name, call in rollout.sandbox.calls if name == "text_editor"] == [
        "str_replace"
    ]
    started = rollout.events("run.started")[0]
    assert (started["slot"], started["slot_version"], started["model"]) == ("sprout-harness", "1", "qwen")
    completed = rollout.events("model.turn.completed")
    assert [event["step"] for event in completed] == [1, 2, 3]
    skipped = rollout.events("tool.skipped")
    assert [event["reason"] for event in skipped] == ["submit"]


def test_each_request_asks_for_log_probs_and_offers_the_papers_three_tools(tmp_path: Path) -> None:
    rollout = run_harness(fixture(tmp_path), list(FIX))

    assert all(request["logprobs"] is True and request["stream"] is False for request in rollout.requests)
    assert [tool["function"]["name"] for tool in rollout.requests[0]["tools"]] == [
        "execute_bash",
        "str_replace",
        "submit",
    ]
    system, user = rollout.requests[0]["messages"]
    assert "Working directory: /testbed" in system["content"] and "Fix the bug in calc.py." in user["content"]
    assert "sed -n" in system["content"] and "write_file" not in user["content"]


def test_the_full_tool_set_reads_files_with_numbered_lines(tmp_path: Path) -> None:
    read = reply(turn("read_file", ("file_path", "calc.py")))
    rollout = run_harness(fixture(tmp_path), [read, FIX[3]], harness={"toolset": "full"})

    names = [tool["function"]["name"] for tool in rollout.requests[0]["tools"]]
    assert names == ["execute_bash", "read_file", "write_file", "str_replace", "submit"]
    result = [m["content"] for m in rollout.requests[-1]["messages"] if m["role"] == "tool"][0]
    assert result.startswith("/testbed/calc.py, lines 1-2 (end of file):\n       1 | def add(a, b):")


def test_the_model_reads_friendly_results(tmp_path: Path) -> None:
    rollout = run_harness(fixture(tmp_path), list(FIX))

    results = [m["content"] for m in rollout.requests[-1]["messages"] if m["role"] == "tool"]
    assert results[0].startswith("Exit code: 0 (") and results[0].endswith("Output:\n" + BUGGY)
    assert results[1].startswith("Edited /testbed/calc.py")
    assert results[2].startswith("Exit code: 0 (") and results[2].endswith("Output:\n5\n")


def test_the_history_holds_the_conversation_that_training_reads(tmp_path: Path) -> None:
    rollout = run_harness(fixture(tmp_path), list(FIX))

    messages = clean_messages(training_messages(rollout.entries()))
    assert [m["role"] for m in messages] == ["system", "user"] + ["assistant", "tool"] * 4
    assert messages[2]["reasoning_content"] == "I know what to do next."
    assert json.loads(messages[2]["tool_calls"][0]["function"]["arguments"]) == {"command": "cat calc.py"}


def test_each_turn_that_ran_a_call_is_a_branch_point_and_submit_is_not(tmp_path: Path) -> None:
    rollout = run_harness(fixture(tmp_path), list(FIX))

    session = rollout.result.native_session_id
    assert session is not None
    events = journal_events(rollout.journal_path)
    points = index_harness(rollout.journal_path, rollout.transcript, session, events=events)
    # The final snapshot pairs with the last turn that ran a call, the command.
    assert [(p.message_step, p.tool_depth) for p in points] == [(1, 1), (2, 2), (3, 3)]
    assert all(p.native["slot"] == "sprout-harness" for p in points)


def test_feedback_mode_tells_the_model_what_was_wrong_and_lets_it_answer_again(tmp_path: Path) -> None:
    malformed = reply("I will read it.\n\n<tool_call>\n<function=read_file>\n</tool_call>")
    rollout = run_harness(fixture(tmp_path), [malformed, *FIX])

    assert rollout.result.status == "completed"
    assert [e["code"] for e in rollout.events("harness.turn.rejected")] == ["unclosed_reasoning"]
    second = rollout.requests[1]["messages"]
    assert second[-1]["role"] == "user" and "was rejected and nothing ran" in second[-1]["content"]
    assert [e["format"]["valid"] for e in rollout.events("raw.sprout-harness")] == [
        False,
        True,
        True,
        True,
        True,
    ]


def test_terminate_mode_ends_the_rollout_at_the_first_invalid_turn(tmp_path: Path) -> None:
    malformed = reply(turn("execute_bash", ("command", "cat calc.py")) + "\nDone.")
    rollout = run_harness(
        fixture(tmp_path), [FIX[0], malformed, *FIX[1:]], harness={"on_invalid": "terminate"}
    )

    assert rollout.result.status == "timeout"
    assert rollout.control.stop_reason == "format_invalid"
    assert len(rollout.requests) == 2
    messages = clean_messages(training_messages(rollout.entries()))
    assert messages[-1]["role"] == "assistant" and messages[-1]["content"].endswith("</tool_call>\nDone.")
    assert rollout.entries()[-1] == {
        "type": "harness.turn",
        "turn_id": rollout.events("harness.turn.rejected")[0]["turn_id"],
        "call_ids": [],
        "source": "actor",
        "rejected": True,
        "code": "text_after_call",
    }


def test_a_response_cut_off_by_the_token_limit_stops_a_training_rollout_as_truncated(tmp_path: Path) -> None:
    cut = reply("I will read it and then", finish_reason="length")
    rollout = run_harness(fixture(tmp_path), [cut], harness={"on_invalid": "terminate"})

    assert rollout.control.stop_reason == "response_truncated"
    assert rollout.result.status == "timeout"


def test_the_token_budget_stops_the_rollout_as_a_gradable_one(tmp_path: Path) -> None:
    big = reply(turn("execute_bash", ("command", "cat calc.py")), tokens=(900, 200))
    rollout = run_harness(fixture(tmp_path), [big, *FIX[1:]], harness={"max_total_tokens": 1000})

    assert rollout.control.stop_reason == "max_total_tokens"
    assert rollout.result.status == "timeout"
    assert len(rollout.requests) == 1


def test_a_call_repeated_twice_with_its_result_unchanged_ends_a_training_rollout(tmp_path: Path) -> None:
    ls = reply(turn("execute_bash", ("command", "ls")))
    rollout = run_harness(
        fixture(tmp_path), [ls, ls, ls, *FIX], harness={"on_invalid": "terminate", "repetition_limit": 2}
    )

    assert rollout.control.stop_reason == "repetition"
    assert [name for name, _ in rollout.sandbox.calls] == ["shell", "shell", "shell"]
    assert rollout.events("tool.skipped") == []


def test_a_repeated_call_whose_result_changed_is_no_repetition(tmp_path: Path) -> None:
    test = reply(turn("execute_bash", ("command", "python -B -c 'import calc; print(calc.add(2, 3))'")))
    fix = reply(turn("execute_bash", ("command", "sed -i 's/a - b/a + b/' calc.py")))
    replies = [test, test, fix, test, FIX[3]]
    rollout = run_harness(
        fixture(tmp_path), replies, harness={"on_invalid": "terminate", "repetition_limit": 2}
    )

    assert rollout.result.status == "completed" and rollout.control.stop_reason is None


def test_a_turn_cut_off_in_its_reasoning_is_kept_as_reasoning(tmp_path: Path) -> None:
    cut = reply("I will read it and then", finish_reason="length")
    rollout = run_harness(fixture(tmp_path), [cut], harness={"on_invalid": "terminate"})

    messages = clean_messages(training_messages(rollout.entries()))
    assert messages[-1] == {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I will read it and then",
    }


def test_a_parsed_response_with_broken_arguments_is_a_format_violation(tmp_path: Path) -> None:
    broken = reply(turn("execute_bash", ("command", "cat calc.py")), logprobs=False)
    broken["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    rollout = run_harness(fixture(tmp_path), [broken, FIX[3]])

    assert rollout.result.status == "completed"
    assert [e["code"] for e in rollout.events("harness.turn.rejected")] == ["malformed_call"]


def test_a_repeated_call_in_feedback_mode_runs_with_a_note(tmp_path: Path) -> None:
    ls = reply(turn("execute_bash", ("command", "ls")))
    rollout = run_harness(fixture(tmp_path), [ls, ls, FIX[3]])

    results = [m["content"] for m in rollout.requests[-1]["messages"] if m["role"] == "tool"]
    assert (
        "you made this exact call before" not in results[0]
        and "you made this exact call before" in results[1]
    )


def test_without_log_probs_the_check_reads_the_text_rebuilt_from_the_parsed_fields(tmp_path: Path) -> None:
    replies = [
        reply(turn("execute_bash", ("command", "cat calc.py")), logprobs=False),
        reply(turn("submit"), logprobs=False),
    ]
    rollout = run_harness(fixture(tmp_path), replies)

    assert rollout.result.status == "completed"
    assert [e["raw_source"] for e in rollout.events("raw.sprout-harness")] == ["structured", "structured"]
