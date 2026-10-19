"""A fake Run Store worker for the tests of a gate or an evaluation: it settles each queued job.

``Worker`` is the ``pause`` of the gate or the evaluation under test. Each call claims each job that
the test's Run Store holds and finishes it in the attempt directory ``<artifacts>/<job id>/<attempt
id>``, as a worker does. It starts no sandbox and calls no model:

- A rollout writes a mini journal and native history of three turns, one ``ls`` call each. It
  publishes a recovery point after each turn, ends with a final snapshot and keeps its sandbox when
  its spec asks. The rollout of a sample in ``failing`` fails instead.
- A grade gives the verdict that ``grade`` makes of its submitter record, a reward of 0 by default.
  It leaves the verifier's ``test-stdout.txt``.
- A review answers with the next entry of ``reviews``: a list of points, or a pair of the points and
  the plan's ``validation_error``. Each point gets the job and the attempt of the parent that its
  ``base`` names. The review writes ``plan-points.json``, as the review job does.
- A job whose submitter record names a sample in ``losing`` fails as the Run Store fails a job of a
  lost host, with ``DATA_LOST``. It takes one entry of the name out of ``losing``. A branch keeps the
  record of the job that it continues, so only the grade of a branch names the branch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sprout.runstore.index import Index
from sprout.runstore.store import Store
from sprout.worker.agents.mini_swe.history import History, index_native
from sprout.worker.agents.mini_swe.tools import mini_tool_schema
from sprout.worker.run.journal import JournalWriter, journal_events
from tests.support.runstore import registered

#: The worker id that claims every job, so that a grade in a kept sandbox is this worker's too.
WORKER_ID = "fake-worker"
#: The tool steps of each rollout. A recovery point follows each one.
STEPS = (1, 2, 3)
#: The completion tokens of each model response in a rollout's journal.
COMPLETION_TOKENS = 100
#: More calls than this mean that the run under test waits for a job that no call settles.
MOST_PAUSES = 200
#: The result of a job whose affinity key is bound to the lost host-1 (``Store.settle_dead_workers``).
DATA_LOST = {
    "status": "error",
    "failure_kind": "infrastructure",
    "reason": "data_lost",
    "host": "host-1",
    "error": "Host host-1 is lost: none of its workers reported a heartbeat in 600 s",
}


def snapshot_id(job_id: str, step: int) -> str:
    """The snapshot that the rollout ``job_id`` takes after tool step ``step``."""
    return f"{job_id}-snap-{step}"


def final_snapshot_id(job_id: str) -> str:
    """The final snapshot of the rollout ``job_id``, which a grade from the snapshot names."""
    return f"{job_id}-final"


class Worker:
    """Settles the queued jobs of ``store``. Each call is one pause of the coordinator."""

    def __init__(
        self,
        store: Store,
        *,
        artifacts: Path,
        context_key: str,
        grade: Callable[[dict[str, Any]], dict[str, Any]] = lambda record: {"reward": 0.0, "reason": None},
        reviews: Sequence[list[dict[str, Any]] | tuple[list[dict[str, Any]], str]] = (),
        failing: frozenset[str] = frozenset(),
        losing: Sequence[str] = (),
    ) -> None:
        registered(store, WORKER_ID)
        self._store = store
        self._index = Index(store, snapshot_valid=lambda point: True)
        self._artifacts = artifacts
        self._context_key = context_key
        self._reviews = list(reviews)
        self._failing = failing
        self._losing = list(losing)
        self._grade = grade
        self.pauses = 0

    def __call__(self) -> None:
        self.pauses += 1
        assert self.pauses <= MOST_PAUSES, "the run waits for a job that the worker never settles"
        while (job := self._store.claim(WORKER_ID)) is not None:
            directory = self._artifacts / job["id"] / job["active_attempt"]
            directory.mkdir(parents=True)
            record = job["request"]["context"].get(self._context_key) or {}
            if record.get("sample_slot_id") in self._losing:
                self._losing.remove(record["sample_slot_id"])
                self._store.finish(job["id"], job["lease_token"], DATA_LOST, state="failed")
                continue
            settle = {"rollout": self._rollout, "grade": self._grade_job, "review": self._review}[job["kind"]]
            settle(job, record=record, directory=directory)

    def _rollout(self, job: dict[str, Any], *, record: dict[str, Any], directory: Path) -> None:
        sample = record.get("sample_slot_id")
        if sample in self._failing:
            self._store.finish(
                job["id"], job["lease_token"], {"error": "mini did not submit"}, state="failed"
            )
            return
        journal = write_rollout(directory, job_id=job["id"])
        transcript = directory / "mini-native.jsonl"
        points = index_native(journal, transcript, "mini-swe-agent", "ses-1")
        self._index.project(job["id"], job["lease_token"], {}, journal_events(journal), points)
        result: dict[str, Any] = {
            "status": "completed",
            "journal_path": str(journal),
            "final_snapshot_id": final_snapshot_id(job["id"]),
        }
        if job["request"]["spec"].get("keep_sandbox"):
            keep_until = datetime.now(UTC) + timedelta(minutes=20)
            result["kept_sandbox"] = {
                "sandbox_id": "sandbox-" + job["id"],
                "keep_until": keep_until.isoformat(),
            }
        self._store.finish(job["id"], job["lease_token"], result)

    def _grade_job(self, job: dict[str, Any], *, record: dict[str, Any], directory: Path) -> None:
        logs = directory / "verifier"
        logs.mkdir()
        (logs / "test-stdout.txt").write_text("FAILED test_answer\n")
        verdict = self._grade(record)
        self._store.finish(job["id"], job["lease_token"], {**verdict, "verifier": {"logs": str(logs)}})

    def _review(self, job: dict[str, Any], *, record: dict[str, Any], directory: Path) -> None:
        spec = job["request"]["spec"]
        answer = self._reviews.pop(0)
        chosen, error = answer if isinstance(answer, tuple) else (answer, None)
        # As the review job does, each point names the job and the attempt of the parent it is on.
        parents = {parent["name"]: parent for parent in spec["parents"]}
        points = [
            {
                **point,
                "job_id": parents[point["base"]]["job_id"],
                "attempt_id": parents[point["base"]]["attempt_id"],
            }
            for point in chosen
        ]
        selected = [
            {"run_name": point["run_name"], "base": point["base"], "branch_step": point["step"]}
            for point in points
        ]
        plan = {"branch_limit": spec["points"], "selected_branches": selected, "validation_error": error}
        (directory / "plan-points.json").write_text(json.dumps(plan))
        result = {"status": "completed", "guidance": spec["guidance"], "points": points}
        self._store.finish(job["id"], job["lease_token"], result)


def write_rollout(directory: Path, *, job_id: str) -> Path:
    """Write the journal and the native history of a mini run of ``STEPS`` into ``directory``.

    Each turn calls ``ls`` once and is captured in ``snapshot_id(job_id, step)``. Returns the journal."""
    journal = directory / "trajectory.jsonl"
    history = History(directory / "mini-native.jsonl", "ses-1")
    with JournalWriter(journal, run_id=job_id) as writer:
        writer.emit("run.started", slot="mini-swe-agent")
        writer.emit(
            "sandbox.created", sandbox_id="sandbox-" + job_id, backend="podman", store="/path/to/store"
        )
        writer.emit("session.ref", native_session_id="ses-1", transcript_path=str(history.path))
        writer.emit("rollout.model_tools", shape="chat/completions", tools=mini_tool_schema())
        for step in STEPS:
            _turn(writer, history, job_id=job_id, step=step)
        writer.emit("run.finished", status="completed")
    history.close()
    return journal


def turn_messages(step: int) -> list[dict[str, Any]]:
    """The chat messages of turn ``step`` in the native history that ``write_rollout`` writes."""
    call = {
        "id": f"call-{step}",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
    return [
        {"role": "assistant", "content": "Inspect.", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": f"call-{step}", "content": "listing"},
    ]


def _turn(writer: JournalWriter, history: History, *, job_id: str, step: int) -> None:
    call, turn = f"call-{step}", f"turn-{step}"
    response = {
        "choices": [{"message": {"content": "Inspect."}}],
        "usage": {"completion_tokens": COMPLETION_TOKENS},
    }
    writer.emit("raw.mini-swe-agent", response=response)
    assistant, result = turn_messages(step)
    history.append({"type": "mini.message", "message": assistant})
    writer.emit("tool.started", call_id=call, turn_id=turn, name="shell", args={"command": "ls"})
    writer.emit("tool.finished", call_id=call, turn_id=turn, status="ok", output="listing")
    history.append({"type": "mini.message", "message": result})
    history.append({"type": "mini.turn", "turn_id": turn, "call_ids": [call]})
    writer.emit("model.turn.completed", turn_id=turn, call_ids=[call], step=step)
    writer.emit(
        "checkpoint.captured",
        step=step,
        turn_id=turn,
        snapshot_id=snapshot_id(job_id, step),
        session_ckpt="ses-1",
        reason="captured",
    )
