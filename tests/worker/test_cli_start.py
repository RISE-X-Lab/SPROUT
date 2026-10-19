"""``start`` of ``python -m sprout.worker``: a Run Store worker, from its config file.

The Run Store is the isolated test database, named by ``SPROUT_RUNSTORE_DSN`` as for the worker. A
test's artifact root is under pytest's temporary directory, which the CLI refuses as volatile, so
those tests answer the volatility check with a stub. Each attempt runs the stand-in child of
``tests.worker.attempt_child``.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout.worker import cli
from sprout.worker.attempts import process_identity
from sprout.worker.kept import LEDGER_NAME, KeptSandboxes
from tests.support.tools import ManagedContainer
from tests.worker.attempt_child import child_python
from tests.worker.claimed import left_running, rollout


def config_file(tmp_path: Path, **profile: Any) -> Path:
    """A Run Store config in ``tmp_path`` whose artifact root is ``tmp_path/artifacts``."""
    path = tmp_path / "config.json"
    body = {"artifact_root": str(tmp_path / "artifacts"), "profiles": {"default": profile}}
    path.write_text(json.dumps(body))
    return path


@pytest.fixture
def worker_host(run_store, monkeypatch) -> Iterator[None]:
    """The CLI's Run Store is ``run_store``, and every artifact root survives a reboot.

    ``start`` installs handlers of SIGTERM and SIGINT. They are put back after the test."""
    monkeypatch.setenv("SPROUT_RUNSTORE_DSN", run_store.dsn)
    monkeypatch.setattr(cli, "volatile_reason", lambda path: None)
    handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    yield
    for number, handler in handlers.items():
        signal.signal(number, handler)


def test_start_refuses_an_artifact_root_that_the_os_empties_on_reboot(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"artifact_root": "/tmp/path/to/artifacts", "profiles": {}}))

    with pytest.raises(
        SystemExit, match="^/tmp/path/to/artifacts is under /tmp, which the OS empties on reboot"
    ):
        cli.main(["start", "--config", str(path)])


@pytest.mark.parametrize("concurrency", ["0", "-1"])
def test_start_refuses_a_concurrency_below_one(tmp_path, concurrency, capsys):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["start", "--config", str(tmp_path / "config.json"), "--concurrency", concurrency])

    assert (exit_.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        "worker start: error: argument --concurrency: must be a positive integer",
    )


def test_start_once_runs_one_queued_job_to_its_end_and_exits_0(run_store, worker_host, tmp_path):
    job = run_store.submit(rollout(), "queued")
    config = config_file(tmp_path, python=child_python(tmp_path))

    code = cli.main(["start", "--config", str(config), "--once"])

    assert (code, run_store.get(job["id"])["state"]) == (0, "succeeded")


def test_start_gives_the_manager_its_concurrency_its_once_and_the_kept_sandboxes_of_its_worker(
    worker_host, tmp_path, monkeypatch
):
    """The manager removes the kept sandboxes whose time passed, so it needs the ledger of the worker."""
    given = {}

    class SpyManager:
        def __init__(self, worker, *, concurrency, kept):
            given.update(concurrency=concurrency, kept_of_the_worker=kept is worker.kept)

        def run(self, *, stop, once):
            given.update(once=once)

    monkeypatch.setattr("sprout.worker.manager.WorkerManager", SpyManager)

    cli.main(["start", "--config", str(config_file(tmp_path)), "--concurrency", "24", "--once"])

    assert given == {"concurrency": 24, "kept_of_the_worker": True, "once": True}


@pytest.mark.parametrize("argv, name", [([], "default"), (["--worker-name", "gpu-1"], "gpu-1")])
def test_start_names_its_worker_after_this_host_and_the_worker_name(
    worker_host, tmp_path, monkeypatch, argv, name
):
    """A restarted worker takes its id again, and with it the jobs and kept sandboxes that it left."""
    given = {}

    class SpyManager:
        def __init__(self, worker, **_: object):
            given.update(worker_id=worker.worker_id)

        def run(self, **_: object):
            pass

    monkeypatch.setattr("sprout.worker.manager.WorkerManager", SpyManager)

    cli.main(["start", "--config", str(config_file(tmp_path)), *argv])

    assert given == {"worker_id": "%s/%s" % (os.uname().nodename, name)}


def test_start_exits_with_the_reason_before_it_registers_when_a_profile_fails_its_check(
    run_store, worker_host, tmp_path
):
    """Each job of the profile would fail at the worker, so the worker claims none."""
    config = config_file(tmp_path, run_defaults={"backend": {"backend": "local"}})

    with pytest.raises(SystemExit, match="^Profile default: run_defaults names the backend 'local'"):
        cli.main(["start", "--config", str(config), "--once"])

    with run_store.transaction() as cursor:
        cursor.execute("SELECT count(*) AS workers FROM rs_workers")
        assert cursor.fetchone()["workers"] == 0


def test_start_gives_its_worker_the_grades_that_its_grade_profiles_name(worker_host, tmp_path, monkeypatch):
    """The registration of the worker records them, and a claim matches each grade against them."""
    rows, parser = tmp_path / "tasks.jsonl", tmp_path / "log_parsers.py"
    rows.write_text('{"instance_id": "a"}\n')
    parser.write_text("NAME_TO_PARSER = {}\n")
    dataset_sha256 = hashlib.sha256(rows.read_bytes()).hexdigest()
    grader_revision = "sha256:" + hashlib.sha256(parser.read_bytes()).hexdigest()
    defaults = {
        "dataset_path": str(rows),
        "dataset_sha256": dataset_sha256,
        "parser_path": str(parser),
        "grader_revision": grader_revision,
    }
    given = {}

    class SpyManager:
        def __init__(self, worker, **_: object):
            given.update(grades=worker.grades)

        def run(self, **_: object):
            pass

    monkeypatch.setattr("sprout.worker.manager.WorkerManager", SpyManager)

    cli.main(["start", "--config", str(config_file(tmp_path, grade_defaults=defaults))])

    assert given == {"grades": ((dataset_sha256, grader_revision),)}


@pytest.mark.parametrize("name", ["", "a/b"])
def test_start_refuses_a_worker_name_that_is_empty_or_has_a_slash(tmp_path, name, capsys):
    with pytest.raises(SystemExit) as exit_:
        cli.main(["start", "--config", str(tmp_path / "config.json"), "--worker-name", name])

    assert (exit_.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        "worker start: error: argument --worker-name: must be nonempty and without a slash",
    )


def test_start_exits_with_the_reason_when_its_worker_is_live_in_another_session(
    run_store, worker_host, tmp_path
):
    worker_id = "%s/default" % os.uname().nodename
    run_store.register_worker(worker_id, host=os.uname().nodename, session="another")

    with pytest.raises(SystemExit, match="^Worker %s is live in another session$" % worker_id):
        cli.main(["start", "--config", str(config_file(tmp_path)), "--once"])


def test_start_exits_with_the_reason_when_the_process_of_the_other_session_still_runs(
    run_store, worker_host, tmp_path
):
    worker_id = "%s/default" % os.uname().nodename
    live = process_identity(os.getpid())
    run_store.register_worker(worker_id, host=os.uname().nodename, session="another", process=live)

    with pytest.raises(SystemExit, match="^Worker %s is live in another session$" % worker_id):
        cli.main(["start", "--config", str(config_file(tmp_path)), "--once"])


def exited_process() -> dict[str, Any]:
    """The identity of a process of this host that has exited, as the row of a crashed worker keeps it."""
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    identity = process_identity(process.pid)
    process.kill()
    process.wait()
    assert identity is not None
    return identity


#: The process identities that the row of a crashed worker can keep. None of them runs on this host now.
GONE_PROCESSES = {
    "exited": exited_process,
    "of an earlier boot": lambda: {**process_identity(os.getpid()), "boot_id": "an-earlier-boot"},
}


@pytest.mark.parametrize("gone", GONE_PROCESSES.values(), ids=GONE_PROCESSES.keys())
def test_start_right_after_a_crash_takes_the_row_over_from_the_session_whose_process_is_gone(
    run_store, worker_host, tmp_path, gone
):
    """A worker that crashed and starts again at once must not wait 600 s for its own row."""
    worker_id = "%s/default" % os.uname().nodename
    run_store.register_worker(worker_id, host=os.uname().nodename, session="crashed", process=gone())

    code = cli.main(["start", "--config", str(config_file(tmp_path)), "--once"])

    assert (code, run_store.worker_heartbeat(worker_id, "crashed")) == (0, False)


def test_start_removes_the_kept_sandboxes_of_its_artifact_root_whose_time_has_passed(
    worker_host, tmp_path, fake_podman, fake_btrfs_store
):
    sandbox = "sprout-0000000000c3"
    sockets = fake_btrfs_store / "sockets" / sandbox
    fake_podman.containers_exist(
        ManagedContainer(name=sandbox, store=fake_btrfs_store, sockets=sockets, created=1700000000)
    )
    backend = {"backend": "podman", "podman": {"store": str(fake_btrfs_store)}}
    kept = KeptSandboxes(tmp_path / "artifacts" / LEDGER_NAME)
    kept.keep(
        sandbox_id=sandbox, keep_until="2026-01-01T00:00:00+00:00", attempt_id="rollout", backend=backend
    )
    config = config_file(tmp_path)

    cli.main(["start", "--config", str(config), "--once"])

    assert (["rm", "-f", sandbox] in fake_podman.calls(), kept.held()) == (True, set())


def test_start_under_the_id_of_a_dead_session_first_settles_the_jobs_that_it_left(
    run_store, worker_host, tmp_path
):
    """The new session takes the row and the leases over, then publishes what the old attempt left."""
    worker_id = "%s/default" % os.uname().nodename
    outcome = {"status": "completed", "final_text": "done before the crash"}
    job = left_running(run_store, tmp_path / "artifacts", rollout(), outcome=outcome, by=worker_id)
    with run_store.transaction() as cursor:
        cursor.execute("UPDATE rs_workers SET seen_at=clock_timestamp() - interval '610 seconds'")

    code = cli.main(["start", "--config", str(config_file(tmp_path)), "--once"])

    finished = run_store.get(job["id"])
    assert (code, finished["state"], finished["result"]["final_text"]) == (
        0,
        "succeeded",
        "done before the crash",
    )
