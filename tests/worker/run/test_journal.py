"""The journal of a run: its envelope, its append-only file, its readers and its subscribers."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.run.events import Usage
from sprout.worker.run.journal import JournalWriter, journal_events, read_journal

#: RFC 3339 UTC with milliseconds. Journals on disk have this precision, so it cannot change.
MILLISECOND_UTC = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z")


class Unencodable:
    """A payload value that ``json`` cannot encode."""

    def __str__(self) -> str:
        return "<unencodable>"


def without_timestamp(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "ts"}


def test_each_event_is_written_in_the_version_2_envelope_and_emit_returns_the_written_record(tmp_path):
    path = tmp_path / "trajectory.jsonl"

    with JournalWriter(path, run_id="run-1", agent_id="agent-1", sandbox_id="sandbox-1") as journal:
        returned = [
            journal.emit("agent.message", text="first"),
            journal.emit("agent.thinking", text="second"),
        ]

    records = read_journal(path)
    assert records == returned
    assert [without_timestamp(record) for record in records] == [
        {
            "v": 2,
            "type": "agent.message",
            "seq": 1,
            "run_id": "run-1",
            "agent_id": "agent-1",
            "sandbox_id": "sandbox-1",
            "text": "first",
        },
        {
            "v": 2,
            "type": "agent.thinking",
            "seq": 2,
            "run_id": "run-1",
            "agent_id": "agent-1",
            "sandbox_id": "sandbox-1",
            "text": "second",
        },
    ]
    assert [bool(MILLISECOND_UTC.fullmatch(record["ts"])) for record in records] == [True, True]


def test_an_event_is_on_disk_before_the_journal_is_closed(tmp_path):
    """A killed run is never closed, and its journal must still hold each event that it emitted."""
    path = tmp_path / "trajectory.jsonl"
    journal = JournalWriter(path, run_id="run-1")

    journal.emit("agent.message", text="before the kill")

    assert [record["text"] for record in journal_events(path)] == ["before the kill"]
    journal.close()


def test_a_writer_on_an_existing_journal_appends_and_keeps_the_earlier_records(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path, run_id="run-1") as journal:
        journal.emit("agent.message", text="first run")

    with JournalWriter(path, run_id="run-2") as journal:
        journal.emit("agent.message", text="second run")

    assert [(record["run_id"], record["text"]) for record in read_journal(path)] == [
        ("run-1", "first run"),
        ("run-2", "second run"),
    ]


def test_a_value_that_json_cannot_encode_is_written_as_its_string(tmp_path):
    path = tmp_path / "trajectory.jsonl"

    with JournalWriter(path) as journal:
        journal.emit("agent.message", text="kept", value=Unencodable())

    [record] = read_journal(path)
    assert (record["text"], record["value"]) == ("kept", "<unencodable>")


def test_each_started_tool_call_gets_its_ordinal_in_the_run_as_its_step(tmp_path):
    with JournalWriter(tmp_path / "trajectory.jsonl") as journal:
        journal.emit("tool.started", call_id="call-a", name="shell", args={})
        journal.emit("agent.message", text="between the calls")
        journal.emit("tool.started", call_id="call-b", name="shell", args={})

        calls = journal.tool_calls()

    assert [(call["call_id"], call["step"]) for call in calls] == [("call-a", 1), ("call-b", 2)]


def test_a_second_start_of_the_same_tool_call_is_refused_and_not_written(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path) as journal:
        journal.emit("tool.started", call_id="call-a", name="shell", args={})

        with pytest.raises(ValueError, match="tool call 'call-a' was already started"):
            journal.emit("tool.started", call_id="call-a", name="shell", args={})

    assert [(record["type"], record["step"]) for record in read_journal(path)] == [("tool.started", 1)]


class ChainingSubscriber:
    """A subscriber that answers each event below ``depth`` with a ``derived`` event one level deeper."""

    def __init__(self, journal: JournalWriter, *, depth: int) -> None:
        self.journal, self.depth = journal, depth
        self.seen: list[tuple[str, int]] = []

    def __call__(self, record: dict[str, Any]) -> None:
        self.seen.append((record["type"], record["seq"]))
        level = record.get("depth", 0)
        if level < self.depth:
            self.journal.emit("derived", depth=level + 1)


def test_an_event_that_a_subscriber_emits_reaches_every_subscriber_once_in_emit_order(tmp_path):
    """A subscriber that emits, such as the checkpoint bridge, must not hide its event from the others."""
    journal = JournalWriter(tmp_path / "trajectory.jsonl")
    emitter = journal.subscribe(ChainingSubscriber(journal, depth=3))
    seen_by_observer: list[tuple[str, int]] = []
    journal.subscribe(lambda record: seen_by_observer.append((record["type"], record["seq"])))

    journal.emit("trigger")
    journal.close()

    chain = [("trigger", 1), ("derived", 2), ("derived", 3), ("derived", 4)]
    assert (emitter.seen, seen_by_observer) == (chain, chain)


def test_a_failing_subscriber_stops_neither_the_event_nor_the_other_subscribers(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    journal = JournalWriter(path)
    observed: list[str] = []

    def broken(record: dict[str, Any]) -> None:
        raise RuntimeError("observer is broken")

    journal.subscribe(broken)
    journal.subscribe(lambda record: observed.append(record["text"]))

    journal.emit("agent.message", text="still recorded")
    journal.close()

    assert ([record["text"] for record in read_journal(path)], observed) == (
        ["still recorded"],
        ["still recorded"],
    )


def test_a_subscriber_is_notified_after_its_event_is_on_disk(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    journal = JournalWriter(path)
    on_disk_when_notified: list[list[str]] = []
    journal.subscribe(lambda record: on_disk_when_notified.append([r["type"] for r in journal_events(path)]))

    journal.emit("agent.message", text="durable first")
    journal.close()

    assert on_disk_when_notified == [["agent.message"]]


#: U+0085 (NEL) and U+2028 (LINE SEPARATOR). A journal keeps them as a tool printed them, and
#: str.splitlines() breaks a line at either.
UNICODE_LINE_BREAKS = "before\x85between\u2028after"


@pytest.mark.parametrize("reader", [read_journal, journal_events], ids=["read_journal", "journal_events"])
def test_a_record_that_holds_unicode_line_breaks_is_read_whole(
    tmp_path, reader: Callable[[Path], list[dict[str, Any]]]
):
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path) as journal:
        journal.emit("tool.finished", call_id="call-a", output=UNICODE_LINE_BREAKS)
        journal.emit("agent.message", text="after the output")

    records = reader(path)

    assert [(record["type"], record.get("output")) for record in records] == [
        ("tool.finished", UNICODE_LINE_BREAKS),
        ("agent.message", None),
    ]


def test_journal_events_skips_a_torn_last_line(tmp_path):
    """Another process can still be writing the last line of the journal."""
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path) as journal:
        journal.emit("agent.message", text="whole")
    with path.open("a") as stream:
        stream.write('{"v": 2, "type": "tool.finished", "output": "cut off')

    records = journal_events(path)

    assert [record["text"] for record in records] == ["whole"]


def test_journal_events_of_a_missing_journal_is_empty(tmp_path):
    assert journal_events(tmp_path / "missing.jsonl") == []


def test_usage_adds_each_dimension_on_its_own():
    total = Usage(input_tokens=10, cached_input_tokens=5, cost_usd=0.5)

    total.add(
        Usage(
            input_tokens=3, output_tokens=2, cache_creation_tokens=4, reasoning_output_tokens=7, cost_usd=0.25
        )
    )

    assert total.as_dict() == {
        "input_tokens": 13,
        "output_tokens": 2,
        "cached_input_tokens": 5,
        "cache_creation_tokens": 4,
        "reasoning_output_tokens": 7,
        "cost_usd": 0.75,
    }
