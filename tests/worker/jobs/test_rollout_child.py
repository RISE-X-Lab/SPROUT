"""The attempt child, ``sprout.worker.jobs.rollout``: it runs one attempt and leaves what the worker reads.

The worker reads the attempt's ``outcome.json``, its resource ledger ``resources.jsonl`` and its
``progress.json``. The in-process tests run ``execute`` with the directory sandboxes of
``sandboxes.py`` in the place of ``SandboxSession``. The tests of ``main`` start the child as the
worker starts it, with the payload on stdin. Some of them run it on those directory sandboxes, from
this process or from a stand-in worker process that the test kills or that exits at once.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

import pytest

from sprout.runstore.payload import encode_payload
from sprout.runstore.specs import digest
from sprout.worker.attempts import process_identity, stop_process
from sprout.worker.jobs.rollout import execute
from sprout.worker.run.resources import ResourceLedger
from tests.support.http import chat_completion, chat_server, silent_endpoint
from tests.support.paths import HARBOR_SHELL_TASK, REPO
from tests.support.waiting import wait_until
from tests.worker.jobs.sandboxed_child import DESTROY_HELD, DESTROY_RELEASED


@pytest.fixture
def attempt(tmp_path):
    """The attempt directory ``job-1/attempt-1``, which the worker makes before the child starts."""
    (tmp_path / "artifacts" / "job-1" / "attempt-1").mkdir(parents=True)
    return tmp_path / "artifacts" / "job-1" / "attempt-1"


def rollout_request(**spec: Any) -> dict[str, Any]:
    """A rollout request of the nop agent from the image ``base``, with ``spec`` over its effective spec."""
    return {
        "kind": "rollout",
        "attempt_id": "attempt-1",
        "profile_config": {},
        "effective_spec": {"prompt": "Do nothing.", "agent": "nop", "sandbox_image": "base", **spec},
    }


def ledger(directory: Path) -> list[dict[str, Any]]:
    """The records of the attempt's resource ledger, without their time and process id."""
    entries = ResourceLedger(directory / "resources.jsonl").entries()
    return [{key: value for key, value in entry.items() if key not in ("ts", "pid")} for entry in entries]


def picked(result: dict[str, Any], *keys: str) -> dict[str, Any]:
    """The items of ``result`` under ``keys``."""
    return {key: result[key] for key in keys}


def roles(messages: list[dict[str, Any]]) -> list[str]:
    return [message["role"] for message in messages]


def modes(directory: Path, *names: str) -> dict[str, int]:
    """The permission bits of each of ``names`` under ``directory``."""
    return {name: (directory / name).stat().st_mode & 0o777 for name in names}


def test_a_rollout_gives_the_outcome_of_its_run_and_where_its_records_are(run_sandboxes, attempt):
    result = execute(rollout_request(), attempt)

    assert result == {
        "run_id": "attempt-1",
        "journal_path": str(attempt / "trajectory.jsonl"),
        "status": "completed",
        "final_text": "",
        "usage": {},
        "native_session_id": None,
        "sandbox_id": "sandbox-1",
        "checkpoints": 0,
        "final_snapshot_id": "snapshot-1",
        "final_memory_snapshot_id": None,
        "error": None,
        "stop_reason": None,
        "native_home": str(attempt / "native-home"),
        "failure_kind": None,
    }


def test_a_rollout_records_the_start_the_claim_and_the_release_of_its_sandbox(run_sandboxes, attempt):
    """The worker's cleanup reclaims each claimed sandbox, and warns of a start that no claim answers."""
    execute(rollout_request(), attempt)

    assert ledger(attempt) == [
        {"event": "allocation_started", "run_id": "attempt-1", "image": "base"},
        {
            "event": "claim",
            "run_id": "attempt-1",
            "kind": "sandbox",
            "id": "sandbox-1",
            "keep": False,
            "backend": "podman",
        },
        {"event": "release", "run_id": "attempt-1", "kind": "sandbox", "id": "sandbox-1"},
        {"event": "run_done", "run_id": "attempt-1"},
    ]


def test_a_rollout_whose_sandbox_does_not_start_leaves_a_start_that_no_claim_answers(run_sandboxes, attempt):
    result = execute(rollout_request(sandbox_image="absent"), attempt)

    assert (result["status"], result["error"]) == (
        "error",
        "RuntimeError: could not create a sandbox from absent: PodmanError: image absent not known",
    )
    assert ledger(attempt) == [
        {"event": "allocation_started", "run_id": "attempt-1", "image": "absent"},
        {"event": "run_done", "run_id": "attempt-1"},
    ]


def test_the_attempt_reports_the_sandbox_that_it_started_in_its_progress(run_sandboxes, attempt):
    execute(rollout_request(), attempt)

    assert json.loads((attempt / "progress.json").read_text()) == {
        "phase": "sandbox",
        "sandbox_id": "sandbox-1",
    }


def test_the_files_of_an_attempt_are_private_to_the_user_of_the_worker(run_sandboxes, attempt):
    """The attempt directory holds the agent's history and the prompts, which other users must not read."""
    execute(rollout_request(), attempt)

    assert modes(attempt, "cwd", "native-home", "trajectory.jsonl") == {
        "cwd": 0o700,
        "native-home": 0o700,
        "trajectory.jsonl": 0o600,
    }


def test_a_rollout_cut_off_at_its_turn_limit_is_exported_as_a_truncated_episode(run_sandboxes, attempt):
    """The RL driver grades and trains on the episode: its messages, its tools and its final snapshot."""
    with chat_server([chat_completion("echo partial > answer")]) as server:
        contract = {
            "message_export": True,
            "model_endpoint": server.url,
            "model": "fixture",
            "deadline_at": time.time() + 60,
            "max_turns": 1,
            "sampling_params": {},
        }
        request = rollout_request(
            agent="mini", model="fixture", timeout_s=60, extra={"rollout_contract": contract}
        )

        result = execute(request, attempt)

    assert picked(result, "status", "agent_status", "stop_reason", "failure_kind", "final_snapshot_id") == {
        "status": "truncated",
        "agent_status": "timeout",
        "stop_reason": "max_turns_reached",
        "failure_kind": None,
        "final_snapshot_id": "snapshot-1",
    }
    assert picked(result, "rollout_usage", "training_provenance") == {
        "rollout_usage": {"model_calls": 1, "tool_calls": 1},
        "training_provenance": {"prefix_messages": 0, "inserted_messages": []},
    }
    messages = result["training_messages"]
    assert roles(messages) == ["system", "user", "assistant", "tool"]
    assert messages[2] == {
        "role": "assistant",
        "content": "Thinking.",
        "reasoning_content": "Inspect the actual file.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "echo partial > answer"}'},
            }
        ],
    }


def test_a_grade_whose_sandbox_does_not_start_leaves_an_allocation_that_the_worker_must_reconcile(attempt):
    """A failed start may still have made a sandbox, which only a reconciliation finds."""
    request = {
        "kind": "grade",
        "attempt_id": "attempt-1",
        "profile_config": {},
        "effective_spec": {
            "benchmark": "harbor",
            "task_dir": str(HARBOR_SHELL_TASK),
            "snapshot_id": "snapshot-of-the-rollout",
            "backend": {"backend": "podman", "podman": {}},
        },
    }

    with pytest.raises(RuntimeError, match="^Could not restore the agent's snapshot: BackendError: "):
        execute(request, attempt)

    assert ledger(attempt) == [
        {"event": "allocation_started", "run_id": "grader", "image": "snapshot-of-the-rollout"},
        {"event": "allocation_unknown", "run_id": "grader", "image": "snapshot-of-the-rollout"},
    ]


def payload(spec: dict[str, Any], *, kind: str = "grade", **fields: Any) -> dict[str, Any]:
    """The frozen payload of the attempt ``attempt-1`` of the job ``job-1``."""
    return {
        "version": 1,
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "kind": kind,
        "effective_spec": spec,
        "profile_config": {},
        "recovery": None,
        **fields,
    }


def child(directory: Path, envelope: dict[str, Any], *argv: str) -> subprocess.CompletedProcess:
    """The attempt child as the worker starts it, with the payload on stdin.

    Its arguments are the attempt directory, ``argv`` and the pid of this process, as its worker's."""
    return subprocess.run(
        [sys.executable, "-m", "sprout.worker.jobs.rollout", str(directory), *argv, str(os.getpid())],
        input=encode_payload(envelope, "job-1", "attempt-1"),
        cwd=REPO,
        capture_output=True,
        timeout=120,
    )


def test_the_child_writes_the_result_of_its_attempt_to_the_outcome(attempt):
    envelope = payload(
        {"benchmark": "harbor", "task_dir": str(HARBOR_SHELL_TASK), "in_place_job": "rollout-job"},
        kept_sandbox=None,
    )

    finished = child(attempt, envelope, digest(envelope))

    outcome = json.loads((attempt / "outcome.json").read_text())
    elapsed = outcome.pop("elapsed_s")
    assert (finished.returncode, outcome) == (
        0,
        {
            "status": "completed",
            "reward": 0.0,
            "reason": "sandbox_lost",
            "verifier": {
                "in_place": True,
                "error": "this worker keeps no such sandbox, or its keep_until has passed",
            },
            "attempt_id": "attempt-1",
        },
    )
    assert isinstance(elapsed, float) and elapsed >= 0


def test_an_attempt_that_raises_still_writes_its_outcome_with_the_error_and_logs_the_traceback(attempt):
    """``failure_kind`` reads the error's type and message. The traceback goes to ``worker.log``."""
    envelope = payload({"benchmark": "harbor"})

    finished = child(attempt, envelope, digest(envelope))

    outcome = json.loads((attempt / "outcome.json").read_text())
    outcome.pop("elapsed_s")
    assert (finished.returncode, outcome) == (
        0,
        {
            "status": "error",
            "failure_kind": "infrastructure",
            "error": "ValueError: A harbor grade needs the task_dir of its Harbor task",
            "attempt_id": "attempt-1",
        },
    )
    assert b"Traceback (most recent call last)" in finished.stderr


@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(("sha256:" + "0" * 64,), id="payload-of-another-digest"),
        pytest.param((), id="no-digest"),
    ],
)
def test_a_child_without_its_payload_writes_no_outcome_and_exits_2(attempt, argv):
    envelope = payload({"benchmark": "harbor"})

    finished = child(attempt, envelope, *argv)

    assert (finished.returncode, (attempt / "outcome.json").exists()) == (2, False)


#: A stand-in worker process. It starts its argv as the worker starts a child: in a new session, with
#: its own stdin, its stderr as the child's output and its pid as the last argument. It prints the pid.
STARTS_A_CHILD = (
    "import os, signal, subprocess, sys\n"
    "argv = sys.argv[1:] + [str(os.getpid())]\n"
    "child = subprocess.Popen(argv, stdout=sys.stderr, start_new_session=True)\n"
    "print(child.pid, flush=True)\n"
)
#: A stand-in worker process that waits after it started its child, until a signal ends it.
STAND_IN_WORKER = STARTS_A_CHILD + "signal.pause()\n"
#: A stand-in worker process that exits at once after it started its child.
EXITING_WORKER = STARTS_A_CHILD
#: What the outcome of an attempt says when its worker process exited during it.
WORKER_DIED = {
    "status": "error",
    "failure_kind": "infrastructure",
    "stop_reason": "worker_died",
    "error": "Worker process exited",
}


@contextmanager
def stand_in_worker(
    argv: list[str], data: bytes, log: IO[str], *, script: str = STAND_IN_WORKER
) -> Iterator[tuple[subprocess.Popen, int]]:
    """A stand-in worker ``script`` that started the child ``argv`` with ``data`` on stdin, and its pid.

    At the exit of the block, it kills the stand-in and the child's process group, if they still run."""
    worker = subprocess.Popen(
        [sys.executable, "-c", script, *argv],
        cwd=REPO,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
    )
    worker.stdin.write(data)
    worker.stdin.close()
    child = int(worker.stdout.readline())
    try:
        yield worker, child
    finally:
        worker.kill()
        worker.wait()
        if process_identity(child) is not None:
            os.killpg(child, signal.SIGKILL)


@contextmanager
def started_child(argv: list[str], data: bytes, log: IO[str]) -> Iterator[subprocess.Popen]:
    """The child ``argv``, started by this process in a new session, with ``data`` on stdin.

    At the exit of the block, it kills the child's process group, if the child still runs."""
    process = subprocess.Popen(
        argv, cwd=REPO, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    process.stdin.write(data)
    process.stdin.close()
    try:
        yield process
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def silent_rollout(url: str) -> dict[str, Any]:
    """The payload of a mini rollout from the image ``base``, whose model at ``url`` never answers."""
    contract = {
        "model_endpoint": url,
        "model": "fixture",
        "deadline_at": time.time() + 120,
        "max_turns": 1,
        "sampling_params": {},
    }
    request = rollout_request(
        agent="mini", model="fixture", timeout_s=120, extra={"rollout_contract": contract}
    )
    return payload(request["effective_spec"], kind="rollout")


def sandboxed_child(root: Path, directory: Path, envelope: dict[str, Any], *options: str) -> list[str]:
    """The command of the attempt child on the sandboxes under ``root``, with the launcher's ``options``.

    The worker that starts the child adds its own pid."""
    return [
        sys.executable,
        "-m",
        "tests.worker.jobs.sandboxed_child",
        *options,
        str(root),
        str(directory),
        digest(envelope),
    ]


#: The resource ledger of a rollout that started its sandbox and then removed it.
REMOVED_ITS_SANDBOX = [
    {"event": "allocation_started", "run_id": "attempt-1", "image": "base"},
    {
        "event": "claim",
        "run_id": "attempt-1",
        "kind": "sandbox",
        "id": "sandbox-1",
        "keep": False,
        "backend": "podman",
    },
    {"event": "release", "run_id": "attempt-1", "kind": "sandbox", "id": "sandbox-1"},
    {"event": "run_done", "run_id": "attempt-1"},
]


def test_a_child_whose_worker_process_dies_stops_and_removes_its_sandbox(tmp_path, attempt):
    """A child that outlived its worker would hold its sandbox, and run beside its job's next attempt.

    Its outcome is an infrastructure failure, so the worker that settles the job runs it again."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    with silent_endpoint(connections=1) as (url, request_out), (attempt / "worker.log").open("w") as log:
        envelope = silent_rollout(url)
        argv = sandboxed_child(tmp_path, attempt, envelope)
        with stand_in_worker(argv, encode_payload(envelope, "job-1", "attempt-1"), log) as (worker, child):
            assert request_out.wait(60), (attempt / "worker.log").read_text()

            worker.kill()
            wait_until(lambda: process_identity(child) is None, timeout_s=30, what="the exit of the child")

    outcome = json.loads((attempt / "outcome.json").read_text())
    assert picked(outcome, *WORKER_DIED) == WORKER_DIED, "the outcome names why the run stopped"
    assert ledger(attempt) == REMOVED_ITS_SANDBOX
    assert list((tmp_path / "sandboxes" / "sandboxes").iterdir()) == []


def test_a_child_whose_worker_process_exited_before_the_child_looked_stops_as_its_worker_died(
    tmp_path, attempt
):
    """The worker process can exit between the start of the child and the child's first check of it."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    envelope = payload(rollout_request()["effective_spec"], kind="rollout")
    argv = sandboxed_child(tmp_path, attempt, envelope)
    with (attempt / "worker.log").open("w") as log:
        data = encode_payload(envelope, "job-1", "attempt-1")
        with stand_in_worker(argv, data, log, script=EXITING_WORKER) as (worker, child):
            worker.wait(timeout=30)

            wait_until(lambda: process_identity(child) is None, timeout_s=30, what="the exit of the child")

    outcome = json.loads((attempt / "outcome.json").read_text())
    assert picked(outcome, *WORKER_DIED) == WORKER_DIED


def test_a_child_that_a_sigterm_stops_after_its_worker_process_exited_stops_as_its_worker_died(
    tmp_path, attempt
):
    """A worker that starts again under the same id can stop the child before the child checks again."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    with silent_endpoint(connections=1) as (url, request_out), (attempt / "worker.log").open("w") as log:
        envelope = silent_rollout(url)
        # The child checks its worker process when it starts, and not again during the test.
        argv = sandboxed_child(tmp_path, attempt, envelope, "--worker-check-s", "600")
        with stand_in_worker(argv, encode_payload(envelope, "job-1", "attempt-1"), log) as (worker, child):
            assert request_out.wait(60), (attempt / "worker.log").read_text()
            worker.kill()
            worker.wait(timeout=30)

            stopped = stop_process(process_identity(child))

    outcome = json.loads((attempt / "outcome.json").read_text())
    assert (stopped, picked(outcome, *WORKER_DIED)) == (True, WORKER_DIED)


def test_a_second_sigterm_does_not_cut_short_the_teardown_that_the_first_started(tmp_path, attempt):
    """A live worker's watchdog kills the child 10 s after its SIGTERM. Until then, the child cleans up."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    with silent_endpoint(connections=1) as (url, request_out), (attempt / "worker.log").open("w") as log:
        envelope = silent_rollout(url)
        argv = [*sandboxed_child(tmp_path, attempt, envelope, "--hold-destroys"), str(os.getpid())]
        with started_child(argv, encode_payload(envelope, "job-1", "attempt-1"), log) as process:
            assert request_out.wait(60), (attempt / "worker.log").read_text()
            process.send_signal(signal.SIGTERM)
            wait_until(lambda: (tmp_path / DESTROY_HELD).exists(), timeout_s=30, what="the sandbox removal")

            process.send_signal(signal.SIGTERM)
            (tmp_path / DESTROY_RELEASED).touch()
            returncode = process.wait(timeout=30)

    assert (returncode, ledger(attempt)) == (0, REMOVED_ITS_SANDBOX)


def test_a_child_whose_teardown_hangs_after_its_worker_died_writes_its_outcome_and_exits(tmp_path, attempt):
    """No worker lives to kill the child, so the child ends itself. The worker's clean-up removes the rest."""
    (tmp_path / "images" / "base" / "tmp").mkdir(parents=True)
    with silent_endpoint(connections=1) as (url, request_out), (attempt / "worker.log").open("w") as log:
        envelope = silent_rollout(url)
        argv = sandboxed_child(tmp_path, attempt, envelope, "--hold-destroys", "--teardown-s", "1")
        with stand_in_worker(argv, encode_payload(envelope, "job-1", "attempt-1"), log) as (worker, child):
            assert request_out.wait(60), (attempt / "worker.log").read_text()

            worker.kill()
            wait_until(lambda: process_identity(child) is None, timeout_s=30, what="the exit of the child")

    outcome = json.loads((attempt / "outcome.json").read_text())
    outcome.pop("elapsed_s")
    assert outcome == {**WORKER_DIED, "attempt_id": "attempt-1"}
    assert [path.name for path in (tmp_path / "sandboxes" / "sandboxes").iterdir()] == ["sandbox-1"]
