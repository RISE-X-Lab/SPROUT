"""Journal invariants: monotonic seq, append-only, crash-survivable, thread-safe."""

from __future__ import annotations

import json
import threading

from sprout.harness_core.events import AGENT_MESSAGE, JOURNAL_SCHEMA_VERSION, Usage
from sprout.harness_core.journal import JournalWriter, read_journal


def test_envelope_and_monotonic_seq(tmp_path):
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, run_id="r1", agent_id="a1", sandbox_id="s1") as journal:
        returned = [journal.emit(AGENT_MESSAGE, text=text) for text in ("one", "two")]

    records = read_journal(path)
    assert records == returned  # emit returns the full record it wrote, seq and payload included
    assert [r["seq"] for r in records] == [1, 2]
    assert all(r["v"] == JOURNAL_SCHEMA_VERSION for r in records)
    first = records[0]
    assert first["run_id"] == "r1" and first["agent_id"] == "a1" and first["sandbox_id"] == "s1"
    assert first["ts"].endswith("Z")
    assert first["text"] == "one"


def test_flushed_per_line_so_killed_runs_keep_trajectory(tmp_path):
    """No context manager, no close: the file must already be readable."""
    path = tmp_path / "killed.jsonl"
    journal = JournalWriter(path, run_id="r2")
    journal.emit(AGENT_MESSAGE, text="before the kill")
    assert len(read_journal(path)) == 1


def test_appends_across_writers(tmp_path):
    path = tmp_path / "resumed.jsonl"
    with JournalWriter(path, run_id="r3") as journal:
        journal.emit(AGENT_MESSAGE, text="first session")
    with JournalWriter(path, run_id="r3") as journal:
        journal.emit(AGENT_MESSAGE, text="second session")
    assert [r["text"] for r in read_journal(path)] == ["first session", "second session"]


def test_concurrent_emitters_produce_unique_seqs(tmp_path):
    """CLI slots emit from a reader thread plus a stderr drain thread."""
    path = tmp_path / "threads.jsonl"
    journal = JournalWriter(path, run_id="r4")

    def emit_many(tag):
        for i in range(50):
            journal.emit(AGENT_MESSAGE, text="%s-%d" % (tag, i))

    threads = [threading.Thread(target=emit_many, args=(t,)) for t in ("a", "b", "c")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    journal.close()

    records = read_journal(path)
    seqs = [r["seq"] for r in records]
    assert len(records) == 150
    assert sorted(seqs) == list(range(1, 151))  # no duplicates, no gaps


def test_non_json_values_do_not_break_the_line(tmp_path):
    class Weird:
        def __repr__(self):
            return "<weird>"

    path = tmp_path / "weird.jsonl"
    with JournalWriter(path) as journal:
        journal.emit(AGENT_MESSAGE, text="x", obj=Weird())
    line = path.read_text().strip()
    assert json.loads(line)["obj"] == "<weird>"


def test_usage_accumulates_dimensions_separately():
    total = Usage()
    total.add(Usage(input_tokens=10, cached_input_tokens=5, cost_usd=0.5))
    total.add(Usage(input_tokens=3, reasoning_output_tokens=7, cost_usd=0.25))
    assert total.input_tokens == 13
    assert total.cached_input_tokens == 5
    assert total.reasoning_output_tokens == 7
    assert total.cost_usd == 0.75
    # a report that leaves dimensions out changes none of the others
    total.add(Usage(output_tokens=2, cache_creation_tokens=4))
    assert total.as_dict() == {"input_tokens": 13, "output_tokens": 2, "cached_input_tokens": 5,
                               "cache_creation_tokens": 4, "reasoning_output_tokens": 7,
                               "cost_usd": 0.75}


def test_events_emitted_from_inside_a_subscriber_reach_every_subscriber(tmp_path):
    """Regression: nested emissions used to be suppressed entirely.

    A subscriber that emits is normal -- the checkpoint bridge turns a
    turn.completed into a checkpoint.captured. Suppressing that nested event made
    it invisible to *other* subscribers, so the resource ledger never recorded the
    snapshots the bridge had just claimed and a killed process left them
    unreclaimable. Recursing instead would let a subscriber re-enter the fan-out
    it is already in, so nested events are queued and drained after it -- to the
    end, not one generation: the emitter here reacts to its own derived events
    too, so the chain proves every level is delivered, to it and to the observer,
    once and parent-first. (An emitter with no stop condition is still the
    caller's bug; the drain only keeps the depth flat.)
    """
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r")
    seen_by_observer = []
    seen_by_emitter = []

    def emitter(record):
        seen_by_emitter.append(record["seq"])
        depth = record.get("depth", 0)
        if depth < 4:                      # a stop condition, as any emitter needs
            journal.emit("derived", depth=depth + 1, note="from inside a subscriber")

    journal.subscribe(emitter)
    journal.subscribe(lambda r: seen_by_observer.append((r["type"], r.get("depth", 0))))

    journal.emit("trigger")
    journal.close()

    chain = [("trigger", 0), ("derived", 1), ("derived", 2), ("derived", 3), ("derived", 4)]
    assert seen_by_observer == chain, seen_by_observer
    assert seen_by_emitter == [1, 2, 3, 4, 5]  # its own nested events included, each once
    assert [(r["type"], r.get("depth", 0)) for r in read_journal(tmp_path / "j.jsonl")] == chain


def test_observer_failure_cannot_break_the_run(tmp_path):
    """Journal subscribers are isolated: a broken observer must not kill a run."""
    journal = JournalWriter(tmp_path / "j.jsonl", run_id="r1")

    def explode(_record):
        raise RuntimeError("observer is broken")

    journal.subscribe(explode)
    record = journal.emit("agent.message", text="still recorded")
    assert record["seq"] == 1
    assert read_journal(tmp_path / "j.jsonl")[0]["text"] == "still recorded"
    journal.close()
