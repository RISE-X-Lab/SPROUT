"""``up``, ``down``, ``status`` and ``logs`` of a podman service, on the podman that the doctor names.

The interpreter in services.yaml is a stub that answers as the worker's doctor, and the podman that
it names is a stub that reports the container state that the test sets. A second stub podman is first
on PATH, so that no call reaches the host's podman.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import pytest

from sprout.services import __main__ as cli
from tests.services.helpers import StubProgram, doctor_report, make_stub, table_rows, write_spec

ABSENT = {"err": "Error: no such container sprout-services-db\n", "code": 125}


@dataclass(frozen=True)
class ContainerHost:
    #: The podman that the sandboxes run on. The worker's doctor names it, and it is not on PATH.
    podman: StubProgram
    #: The interpreter that services.yaml names. It answers as ``python -m sprout.worker doctor --json``.
    python: StubProgram


@pytest.fixture
def host(tmp_path, monkeypatch) -> ContainerHost:
    podman = make_stub(tmp_path / "sandboxes" / "podman")
    python = make_stub(tmp_path / "venv" / "python")
    python.answer(out=doctor_report(podman_bin=str(podman.path)))
    path_podman = make_stub(tmp_path / "bin" / "podman")
    path_podman.answer(err="the podman on PATH is not the podman of the sandboxes\n", code=125)
    monkeypatch.setenv("PATH", "%s%s%s" % (path_podman.path.parent, os.pathsep, os.environ["PATH"]))
    return ContainerHost(podman=podman, python=python)


def write_db(paths, host: ContainerHost, **fields) -> None:
    """services.yaml with one podman service, ``db``, whose probe is the state that podman reports."""
    db = {
        "kind": "podman",
        "image": "docker.io/library/postgres:16-alpine",
        "container": "sprout-services-db",
        "port": 15432,
        "health": {"kind": "none"},
        **fields,
    }
    write_spec(paths, {"db": db}, python=str(host.python.path))


def lifecycle(podman: StubProgram) -> list[list[str]]:
    """Each call to podman other than ``inspect``, in call order."""
    return [call for call in podman.calls() if call[0] != "inspect"]


def test_up_creates_a_missing_container_with_podman_run(paths, host, capsys):
    host.podman.answer(when="inspect", **ABSENT)
    write_db(paths, host)

    code = cli.main(["up", "db"])

    assert (code, capsys.readouterr().out.splitlines(), lifecycle(host.podman)) == (
        0,
        [
            "db: creating container sprout-services-db from docker.io/library/postgres:16-alpine "
            "(podman run)",
            "db: up (container sprout-services-db, )",
        ],
        [["run", "-d", "--name", "sprout-services-db", "docker.io/library/postgres:16-alpine"]],
    )


def test_up_starts_a_stopped_container_and_does_not_create_it_again(paths, host, capsys):
    """After a reboot the container exists and is stopped. A second ``podman run`` of its name fails."""
    host.podman.answer(when="inspect", out="exited 0\n")
    write_db(paths, host)

    code = cli.main(["up", "db"])

    assert (code, capsys.readouterr().out.splitlines(), lifecycle(host.podman)) == (
        0,
        [
            "db: starting container sprout-services-db (podman start)",
            "db: up (container sprout-services-db, )",
        ],
        [["start", "sprout-services-db"]],
    )


def test_up_leaves_a_running_healthy_container_alone(paths, host, capsys):
    host.podman.answer(when="inspect", out="running 4242\n")
    write_db(paths, host)

    code = cli.main(["up", "db"])

    assert (code, capsys.readouterr().out, lifecycle(host.podman)) == (
        0,
        "db: already healthy (container sprout-services-db running)\n",
        [],
    )


def test_up_fails_when_the_container_stops_before_it_answers_its_probe(
    paths, host, tmp_path, monkeypatch, capsys
):
    pg_isready = make_stub(tmp_path / "client" / "pg_isready")
    pg_isready.answer(out="127.0.0.1:15432 - no response\n", code=2)
    monkeypatch.setenv("PATH", "%s%s%s" % (pg_isready.path.parent, os.pathsep, os.environ["PATH"]))
    host.podman.answer(when="inspect", out="exited 1\n")
    write_db(paths, host, health={"kind": "pg_isready", "port": 15432}, timeout_s=5)

    code = cli.main(["up", "db"])

    lines = capsys.readouterr().out.splitlines()
    assert (code, len(lines)) == (1, 2)
    assert re.fullmatch(
        r"db: FAILED: container sprout-services-db is not healthy after \d+ s "
        r"\(container sprout-services-db is exited\); see `podman logs sprout-services-db`",
        lines[1],
    ), lines[1]


def test_down_stops_a_running_container_and_never_removes_it(paths, host, capsys):
    """The data directory is a bind mount, and ``up`` starts the stopped container again."""
    host.podman.answer(when="inspect", out="running 4242\n")
    write_db(paths, host)

    code = cli.main(["down", "db"])

    assert (code, capsys.readouterr().out.splitlines(), lifecycle(host.podman)) == (
        0,
        ["db: stopping container sprout-services-db (podman stop)", "db: stopped"],
        [["stop", "--time", "20", "sprout-services-db"]],
    )


@pytest.mark.parametrize(
    "state, printed",
    [
        ({"out": "exited 0\n"}, "db: container sprout-services-db is exited\n"),
        (ABSENT, "db: container sprout-services-db does not exist\n"),
    ],
    ids=["stopped", "missing"],
)
def test_down_of_a_container_that_is_not_running_stops_nothing(paths, host, capsys, state, printed):
    host.podman.answer(when="inspect", **state)
    write_db(paths, host)

    code = cli.main(["down", "db"])

    assert (code, capsys.readouterr().out, lifecycle(host.podman)) == (0, printed, [])


def test_a_failed_podman_stop_fails_down(paths, host, capsys):
    host.podman.answer(when="inspect", out="running 4242\n")
    host.podman.answer(when="stop", err="Error: simulated stop failure\n", code=125)
    write_db(paths, host)

    code = cli.main(["down", "db"])

    assert (code, capsys.readouterr().out.splitlines()[1]) == (
        1,
        "db: FAILED: podman stop failed (exit 125); last lines of %s:" % paths.log_file("db"),
    )


@pytest.mark.parametrize("command", [["up", "db"], ["down", "db"]])
def test_without_a_doctor_report_a_container_command_fails_with_the_doctors_reason(
    paths, host, capsys, command
):
    host.python.answer(err="ModuleNotFoundError: No module named 'sprout'\n", code=1)
    write_db(paths, host)

    code = cli.main(command)

    assert (code, capsys.readouterr().out) == (
        1,
        "db: FAILED: %s -m sprout.worker doctor --json gave no report (exit 1): "
        "ModuleNotFoundError: No module named 'sprout'\n" % host.python.path,
    )


def test_status_shows_the_pid_that_podman_reports_for_a_running_container(paths, host, capsys):
    host.podman.answer(when="inspect", out="running 4242\n")
    write_db(paths, host)

    cli.main(["status"])

    assert table_rows(capsys.readouterr().out)[1] == [
        "db",
        "podman",
        "4242",
        "ok: container sprout-services-db running",
        "15432",
    ]


@pytest.mark.parametrize("args, flags", [([], []), (["-f"], ["-f"])], ids=["tail", "follow"])
def test_logs_of_a_container_is_podman_logs_and_exits_with_its_status(paths, host, args, flags):
    """The container's output is in podman. The service log holds only what the CLI did."""
    host.podman.answer(when="logs", code=3)
    write_db(paths, host)

    code = cli.main(["logs", "db", *args])

    assert (code, lifecycle(host.podman)) == (3, [["logs", *flags, "sprout-services-db"]])
