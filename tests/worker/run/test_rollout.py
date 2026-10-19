"""The limits of an externally scheduled rollout: its deadline, its turn limit and its sampling parameters."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.run.rollout import RolloutControls, remaining_timeout
from sprout.worker.tools.pipeline import CallContext, Continue, Reject
from tests.support.waiting import wait_until

#: The time that the fixed clock reads, in Unix seconds.
NOW = 1_000_000.0
SHELL_CALL = CallContext(agent_id="agent", sandbox_id="sandbox-1", tool_name="shell", args={"command": "ls"})


class Clock:
    """The Unix time that the rollout limits read. A test moves it by setting ``now``."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """The clock of the rollout limits, at ``NOW``."""
    fixed = Clock(NOW)
    monkeypatch.setattr("sprout.worker.run.rollout.time.time", fixed)
    return fixed


def contract(**changes: Any) -> dict[str, Any]:
    """A rollout contract with a turn limit of 2 and a deadline an hour after ``NOW``."""
    return {
        "model_endpoint": "http://127.0.0.1:8000/v1",
        "model": "contract-model",
        "deadline_at": NOW + 3600,
        "max_turns": 2,
        "sampling_params": {"temperature": 0.4, "max_new_tokens": 7, "top_k": 11, "stop": ["stop-here"]},
        **changes,
    }


def controls(tmp_path: Path, **changes: Any) -> RolloutControls:
    journal = JournalWriter(tmp_path / "trajectory.jsonl")
    return RolloutControls(contract(**changes), journal, RunControl())


def journaled(tmp_path: Path, event_type: str) -> list[dict[str, Any]]:
    """The events of ``event_type`` in the journal of ``controls``."""
    return [event for event in read_journal(tmp_path / "trajectory.jsonl") if event["type"] == event_type]


@pytest.mark.parametrize(
    "deadline_at,timeout_s,expected",
    [
        pytest.param(NOW + 7200, 3600.0, 3600.0, id="deadline-after-the-timeout"),
        pytest.param(NOW + 30, 3600.0, 30.0, id="deadline-before-the-timeout"),
        pytest.param(NOW + 0.5, 3600.0, 0.5, id="deadline-just-ahead"),
    ],
)
def test_the_timeout_is_cut_to_the_seconds_left_before_the_deadline(clock, deadline_at, timeout_s, expected):
    assert remaining_timeout({"deadline_at": deadline_at}, timeout_s) == expected


@pytest.mark.parametrize(
    "rollout",
    [
        pytest.param({}, id="missing"),
        pytest.param({"deadline_at": True}, id="bool"),
        pytest.param({"deadline_at": "2030-01-01"}, id="text"),
        pytest.param({"deadline_at": float("inf")}, id="infinite"),
        pytest.param({"deadline_at": float("nan")}, id="nan"),
    ],
)
def test_a_deadline_that_is_not_a_finite_number_is_refused(clock, rollout):
    with pytest.raises(ValueError, match="^Rollout deadline must be finite$"):
        remaining_timeout(rollout, 3600.0)


@pytest.mark.parametrize("deadline_at", [NOW, NOW - 1], ids=["now", "past"])
def test_a_deadline_that_has_come_is_refused(clock, deadline_at):
    with pytest.raises(ValueError, match="^Rollout group wall-time budget expired before execution$"):
        remaining_timeout({"deadline_at": deadline_at}, 3600.0)


@pytest.mark.parametrize(
    "changes,message",
    [
        pytest.param(
            {"deadline_at": float("inf")}, "Rollout deadline must be finite", id="infinite-deadline"
        ),
        pytest.param({"max_turns": None}, "Invalid max_turns", id="max_turns-missing"),
        pytest.param({"max_turns": 0}, "Invalid max_turns", id="max_turns-zero"),
        pytest.param({"max_turns": -1}, "Invalid max_turns", id="max_turns-negative"),
        # A bool is an int to isinstance, and True would cap every rollout at one turn.
        pytest.param({"max_turns": True}, "Invalid max_turns", id="max_turns-bool"),
        pytest.param({"max_turns": 1.5}, "Invalid max_turns", id="max_turns-float"),
        # The contract has no version, so a key of the older budgets is refused, not ignored.
        pytest.param({"max_model_calls": 2}, "model/tool call budgets", id="legacy-max_model_calls"),
        pytest.param({"max_tool_calls": 2}, "model/tool call budgets", id="legacy-max_tool_calls"),
        pytest.param({"sampling_params": {"seed": 1}}, "Unknown message sampling controls", id="sampling"),
        pytest.param({"model_endpoint": "ftp://model"}, "Rollout endpoint must be HTTP", id="endpoint"),
    ],
)
def test_a_contract_that_cannot_be_applied_is_refused_when_the_controls_are_made(
    tmp_path, clock, changes, message
):
    with pytest.raises(ValueError, match=message):
        controls(tmp_path, **changes)


def test_a_contract_without_a_deadline_is_refused_with_value_error(tmp_path, clock):
    rollout = contract()
    del rollout["deadline_at"]

    with pytest.raises(ValueError):
        RolloutControls(rollout, JournalWriter(tmp_path / "trajectory.jsonl"), RunControl())


def test_a_model_request_gets_the_sampling_parameters_and_the_model_of_the_contract(tmp_path, clock):
    request = {"model": "agent-requested", "temperature": 1, "messages": [{"role": "user", "content": "hi"}]}

    prepared = controls(tmp_path).prepare_model_request(request)

    assert prepared == {
        "model": "contract-model",
        "temperature": 0.4,
        "max_tokens": 7,
        "top_k": 11,
        "stop": ["stop-here"],
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_a_contract_without_a_model_keeps_the_model_of_the_request(tmp_path, clock):
    prepared = controls(tmp_path, model=None, sampling_params={}).prepare_model_request(
        {"model": "agent-requested"}
    )

    assert prepared == {"model": "agent-requested"}


def test_each_model_request_journals_the_tools_that_the_model_is_offered(tmp_path, clock):
    """The training export reads the tools that the model saw from these events."""
    limits = controls(tmp_path)
    bash = {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}}

    limits.prepare_model_request({"model": "m", "tools": [bash]})
    limits.prepare_model_request({"model": "m"})

    offered = [(event["shape"], event["tools"]) for event in journaled(tmp_path, "rollout.model_tools")]
    assert offered == [("chat/completions", [bash]), ("chat/completions", [])]


def test_the_request_past_the_turn_limit_is_refused_and_stops_the_run(tmp_path, clock):
    limits = controls(tmp_path, max_turns=2)
    limits.prepare_model_request({"model": "m"})
    limits.prepare_model_request({"model": "m"})

    with pytest.raises(ValueError, match="^rollout turn budget exhausted$"):
        limits.prepare_model_request({"model": "m"})

    assert (limits.control.reason, limits.control.stop_reason) == (
        "rollout turn budget exhausted",
        "max_turns_reached",
    )


def test_a_model_request_after_the_deadline_is_refused_and_stops_the_run_as_a_timeout(tmp_path, clock):
    limits = controls(tmp_path, deadline_at=NOW + 10)
    clock.now = NOW + 10

    with pytest.raises(ValueError, match="^rollout wall-time budget exhausted or execution stopped$"):
        limits.prepare_model_request({"model": "m"})

    assert limits.control.stop_reason == "timeout"


def test_a_model_request_after_a_stop_is_refused_and_the_stop_keeps_its_reason(tmp_path, clock):
    limits = controls(tmp_path)
    limits.control.request_stop("execution uncertain at step 3")

    with pytest.raises(ValueError, match="^rollout wall-time budget exhausted or execution stopped$"):
        limits.prepare_model_request({"model": "m"})

    assert (limits.control.reason, limits.control.stop_reason) == ("execution uncertain at step 3", None)


def test_tool_calls_are_counted_and_not_limited(tmp_path, clock):
    limits = controls(tmp_path, max_turns=1)
    limits.prepare_model_request({"model": "m"})

    verdicts = (limits.before(SHELL_CALL), limits.before(SHELL_CALL), limits.before(SHELL_CALL))
    limits.finish()

    [usage] = journaled(tmp_path, "rollout.usage")
    assert (verdicts, usage["model_calls"], usage["tool_calls"]) == (
        (Continue(), Continue(), Continue()),
        1,
        3,
    )


def test_a_tool_call_after_a_stop_is_rejected(tmp_path, clock):
    limits = controls(tmp_path)
    limits.control.request_stop("operator stop")

    verdict = limits.before(SHELL_CALL)

    assert verdict == Reject("rollout wall-time budget exhausted or execution stopped")


def test_the_armed_deadline_stops_the_run_as_a_timeout(tmp_path):
    limits = RolloutControls(
        contract(deadline_at=time.time() + 0.2), JournalWriter(tmp_path / "trajectory.jsonl"), RunControl()
    )

    limits.start()

    wait_until(lambda: limits.control.reason is not None, timeout_s=10, what="the deadline stop")
    assert (limits.control.reason, limits.control.stop_reason) == (
        "rollout wall-time budget exhausted",
        "timeout",
    )


def test_controls_whose_deadline_has_passed_refuse_to_start(tmp_path, clock):
    limits = controls(tmp_path, deadline_at=NOW + 10)
    clock.now = NOW + 11

    with pytest.raises(ValueError, match="^Rollout group wall-time budget expired before execution$"):
        limits.start()
