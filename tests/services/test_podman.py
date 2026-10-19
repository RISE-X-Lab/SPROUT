"""The podman verbs of a container service, against a stub podman that records each call and its environment.

The stub answers as the test sets, so each test arranges the state that podman reports. The shared fake
podman of ``tests/support`` has no ``stop`` and records no environment, and both are what these tests
check. The real podman is never run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.services.podman import ContainerState, Podman
from sprout.services.spec import Context, load_spec, parse_spec, template_path
from sprout.services.supervisor import StartError
from tests.services.helpers import StubProgram, make_stub

DB_STATE = [
    "inspect",
    "--type",
    "container",
    "--format",
    "{{.State.Status}} {{.State.Pid}}",
    "sprout-services-db",
]


@pytest.fixture
def sandbox_podman(tmp_path: Path) -> StubProgram:
    """The podman that the sandboxes run on. It is not on PATH."""
    return make_stub(tmp_path / "sandboxes" / "podman")


def podman_of(paths, podman_bin: str) -> Podman:
    return Podman(paths.log_file("db"), paths.service("db"), {"PATH": "/usr/bin:/bin"}, podman_bin=podman_bin)


def context(paths) -> Context:
    return Context(
        paths=paths,
        python="/path/to/python",
        secret=lambda name: "secret-value-of-" + name,
        host=lambda name: "/path/to/" + name,
        environ={},
    )


def template_db():
    return load_spec(template_path("services.yaml")).get("db")


# --- inspect ------------------------------------------------------------------


def test_inspect_asks_podman_for_the_state_and_the_pid_of_the_container(paths, sandbox_podman):
    sandbox_podman.answer(when="inspect", out="running 4242\n")

    state = podman_of(paths, str(sandbox_podman.path)).inspect("sprout-services-db")

    assert (state, sandbox_podman.calls()) == (ContainerState("running", 4242), [DB_STATE])


@pytest.mark.parametrize(
    "answer, expected",
    [
        ({"out": "exited 0\n"}, ContainerState("exited", None)),
        (
            {"err": "Error: no such container sprout-services-db\n", "code": 125},
            ContainerState("absent", None),
        ),
    ],
)
def test_inspect_reads_a_stopped_or_unknown_container_from_what_podman_reports(
    paths, sandbox_podman, answer, expected
):
    sandbox_podman.answer(when="inspect", **answer)

    assert podman_of(paths, str(sandbox_podman.path)).inspect("sprout-services-db") == expected


@pytest.mark.parametrize("podman_bin", ["/path/to/absent/podman", "podman"])
def test_inspect_without_a_podman_reports_the_container_absent(paths, tmp_path, monkeypatch, podman_bin):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert podman_of(paths, podman_bin).inspect("sprout-services-db") == ContainerState("absent", None)


def test_inspect_writes_nothing_to_the_service_log(paths, sandbox_podman):
    """``status`` inspects the container on each call."""
    sandbox_podman.answer(when="inspect", out="running 4242\n")

    podman_of(paths, str(sandbox_podman.path)).inspect("sprout-services-db")

    assert not paths.log_file("db").exists()


# --- run ----------------------------------------------------------------------


def test_run_creates_the_template_database_with_the_mount_expanded_and_each_variable_by_name(
    paths, sandbox_podman
):
    """podman reads a source without a leading ``/``, such as an unexpanded ``{state}``, as a named volume."""
    podman_of(paths, str(sandbox_podman.path)).run(template_db(), context(paths))

    assert sandbox_podman.calls() == [
        [
            "run",
            "-d",
            "--name",
            "sprout-services-db",
            "-p",
            "127.0.0.1:15432:5432",
            "-v",
            "%s/db/data:/var/lib/postgresql/data:Z" % paths.state,
            "-e",
            "POSTGRES_USER",
            "-e",
            "POSTGRES_DB",
            "-e",
            "POSTGRES_PASSWORD",
            "docker.io/library/postgres:16-alpine",
        ]
    ]


def test_run_gives_podman_the_value_of_each_variable_in_its_environment(paths, sandbox_podman):
    """``-e NAME`` without a value makes podman copy the variable from its own environment."""
    podman_of(paths, str(sandbox_podman.path)).run(template_db(), context(paths))

    env = sandbox_podman.environments()[0]
    assert (env["POSTGRES_USER"], env["POSTGRES_DB"], env["POSTGRES_PASSWORD"]) == (
        "sprout_runstore",
        "sprout_runstore",
        "secret-value-of-db-password",
    )


def test_the_service_log_of_a_run_holds_no_secret_value(paths, sandbox_podman):
    podman_of(paths, str(sandbox_podman.path)).run(template_db(), context(paths))

    assert "secret-value-of-db-password" not in paths.log_file("db").read_text()


def test_run_makes_the_host_side_of_a_bind_mount_before_podman_sees_it(paths, sandbox_podman):
    """A source that podman makes belongs to whoever the container's first process is."""
    podman_of(paths, str(sandbox_podman.path)).run(template_db(), context(paths))

    assert (paths.state / "db" / "data").is_dir()


def test_a_named_volume_goes_to_podman_as_written_and_no_directory_is_made_for_it(
    paths, sandbox_podman, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    body = {
        "kind": "podman",
        "image": "postgres",
        "container": "pg",
        "volumes": ["dbdata:/var/lib/postgresql/data"],
    }
    service = parse_spec({"services": {"db": body}}).get("db")

    podman_of(paths, str(sandbox_podman.path)).run(service, context(paths))

    assert (sandbox_podman.calls(), (tmp_path / "dbdata").exists()) == (
        [["run", "-d", "--name", "pg", "-v", "dbdata:/var/lib/postgresql/data", "postgres"]],
        False,
    )


# --- start, stop, logs --------------------------------------------------------


def test_start_starts_the_stopped_container_by_its_name(paths, sandbox_podman):
    podman_of(paths, str(sandbox_podman.path)).start("sprout-services-db")

    assert sandbox_podman.calls() == [["start", "sprout-services-db"]]


def test_stop_gives_the_container_20_seconds_before_podman_kills_it(paths, sandbox_podman):
    podman_of(paths, str(sandbox_podman.path)).stop("sprout-services-db")

    assert sandbox_podman.calls() == [["stop", "--time", "20", "sprout-services-db"]]


def test_a_failed_verb_raises_start_error_with_the_last_lines_of_the_service_log(paths, sandbox_podman):
    sandbox_podman.answer(when="stop", err="Error: simulated stop failure\n", code=125)

    with pytest.raises(StartError) as caught:
        podman_of(paths, str(sandbox_podman.path)).stop("sprout-services-db")

    lines = str(caught.value).splitlines()
    assert (lines[0], "Error: simulated stop failure" in lines) == (
        "podman stop failed (exit 125); last lines of %s:" % paths.log_file("db"),
        True,
    )


@pytest.mark.parametrize(
    "follow, argv", [(False, ["logs", "sprout-services-db"]), (True, ["logs", "-f", "sprout-services-db"])]
)
def test_logs_runs_podman_logs_and_returns_its_status(paths, sandbox_podman, follow, argv):
    sandbox_podman.answer(when="logs", code=3)

    status = podman_of(paths, str(sandbox_podman.path)).logs("sprout-services-db", follow)

    assert (status, sandbox_podman.calls()) == (3, [argv])


def test_a_verb_without_a_podman_says_how_to_install_one(paths, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(StartError) as caught:
        podman_of(paths, "podman").start("sprout-services-db")

    assert str(caught.value) == (
        "no podman at podman (ops/setup_podman.sh installs one under ~/.local/opt/podman; "
        "$SPROUT_PODMAN_BIN names another)"
    )
