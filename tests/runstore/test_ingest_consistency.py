"""Concurrent journal/native growth must not combine different observation frames."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sprout.harness.agents.mini_swe.history import index_native
from sprout.runstore.worker import Worker

SESSION = "session"


def tool_events(depth: int) -> list[dict]:
    return [
        {"seq": depth * 10, "type": "tool.started", "call_id": f"call-{depth}", "step": depth,
         "turn_id": f"turn-{depth}", "name": "shell", "args": {}},
        {"seq": depth * 10 + 1, "type": "tool.finished", "call_id": f"call-{depth}", "output": "ok"},
    ]


def turn_events(turn: str, calls: list[str], step: int) -> list[dict]:
    return [{"seq": step * 10 + 5, "type": "model.turn.completed", "turn_id": turn, "call_ids": calls,
             "step": step},
            {"seq": step * 10 + 6, "type": "checkpoint.captured", "step": step, "turn_id": turn,
             "snapshot_id": f"snapshot-{step}", "session_ckpt": SESSION, "reason": "captured"}]


def native_turn(turn: str, calls: list[str], results: list[str] | None = None) -> list[dict]:
    """One mini response: the assistant message, its tool results, the turn marker."""
    results = calls if results is None else results
    return [{"type": "mini.message", "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function", "function": {"name": "bash", "arguments": "{}"}}
                for call in calls]}},
            *[{"type": "mini.message", "message": {"role": "tool", "tool_call_id": call, "content": "ok"}}
              for call in results],
            {"type": "mini.turn", "turn_id": turn, "call_ids": calls}]


def append(path: Path, entries: list[dict]) -> None:
    with path.open("a") as stream:
        for entry in entries:
            stream.write(json.dumps(entry) + "\n")


def fixture_files(directory: Path) -> tuple[Path, Path]:
    journal = directory / "trajectory.jsonl"
    native = directory / f"native-home/{SESSION}.jsonl"
    native.parent.mkdir(parents=True)
    append(journal, [{"seq": 2, "type": "session.ref", "native_session_id": SESSION}])
    append(native, [{"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"}])
    return journal, native


def test_ingestion_uses_one_journal_frame_when_writer_advances(tmp_path):
    journal, native = fixture_files(tmp_path)
    append(journal, [*tool_events(1), *turn_events("turn-1", ["call-1"], 1)])
    append(native, native_turn("turn-1", ["call-1"]))
    observations = []
    advanced = False

    def persist(job_id, token, events):
        nonlocal advanced
        if not advanced:
            append(journal, [*tool_events(2), *turn_events("turn-2", ["call-2"], 2)])
            append(native, native_turn("turn-2", ["call-2"]))
            advanced = True

    def project(job_id, token, scope, events, points):
        tool_count = sum(event.get("type") == "tool.started" for event in events)
        assert all(point.tool_depth <= tool_count for point in points)
        observations.append([point.tool_depth for point in points])

    worker = Worker(SimpleNamespace(append_events=persist), {
        "artifact_root": str(tmp_path), "profiles": {}}, snapshot_valid=lambda point: True)
    worker.index = SimpleNamespace(project=project)
    job = {"id": "job", "lease_token": "lease", "request": {
        "spec": {"slot": "mini-swe-agent"}, "context": {}, "profile_hash": None}}
    envelope = {"effective_spec": {"slot": "mini-swe-agent"}, "recovery": None}
    (tmp_path / "request.json").write_text("invalid obsolete file")
    worker._ingest(job, tmp_path, envelope)
    worker._ingest(job, tmp_path, envelope)
    assert observations == [[1], [1, 2]]


@pytest.mark.parametrize("unjournaled_turn, calls, pinned_depths", [
    # A whole leading native turn the journal never saw: only a pinned prefix admits it.
    (native_turn("turn-0", ["call-0"]), ["call-1"], [1]),
    # The journal has seen only call-1 and a turn claiming only call-1, but the
    # native response holds call-1 and call-2: that prefix is not complete, and
    # a pin of what precedes the response holds no call-2 to admit.
    ([], ["call-1", "call-2"], []),
])
def test_inherited_tools_require_an_explicit_pinned_native_prefix(tmp_path, unjournaled_turn, calls,
                                                                  pinned_depths):
    journal, native = fixture_files(tmp_path)
    append(native, unjournaled_turn)
    data = native.read_bytes()
    inherited = {"slot": "mini-swe-agent", "path": str(native), "byte_length": len(data),
                 "sha256": hashlib.sha256(data).hexdigest()}
    append(journal, [*tool_events(1), *turn_events("turn-1", ["call-1"], 1)])
    append(native, native_turn("turn-1", calls))
    assert index_native(journal, native, "mini-swe-agent", SESSION) == []
    points = index_native(journal, native, "mini-swe-agent", SESSION, inherited_native=inherited)
    assert [point.tool_depth for point in points] == pinned_depths


def test_frozen_frame_does_not_reopen_the_growing_journal(tmp_path, monkeypatch):
    from sprout.harness_checkpoint import rollback

    journal, native = fixture_files(tmp_path)
    append(journal, [*tool_events(1), *turn_events("turn-1", ["call-1"], 1)])
    append(native, native_turn("turn-1", ["call-1"]))
    events = [json.loads(line) for line in journal.read_text().splitlines()]

    def refuse(*args, **kwargs):
        raise AssertionError("Reopened journal after selecting the observation frame")

    monkeypatch.setattr(Path, "read_text", refuse)
    monkeypatch.setattr(rollback, "read_journal", refuse)
    assert len(index_native(journal, native, "mini-swe-agent", SESSION, events=events)) == 1
