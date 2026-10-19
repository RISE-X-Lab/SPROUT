"""``up``, ``down``, ``restart``, ``status`` and ``logs`` of process and external services, through ``main``.

Each process service is ``SERVICE_PROGRAM`` on a free loopback port, and its probe is a GET of /health.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading

import pytest

from sprout.services import __main__ as cli
from sprout.services.spec import parse_spec
from sprout.services.supervisor import process_state, read_pid, start
from tests.services.helpers import (
    free_port,
    is_running,
    ready_probe,
    service_command,
    table_rows,
    write_spec,
)
from tests.support.waiting import wait_until

UNSET = "SPROUT_SERVICES_TEST_UNSET"


@pytest.fixture(autouse=True)
def unset_variable(monkeypatch):
    """Make sure that ``UNSET`` is not set, so that a placeholder that names it cannot be resolved."""
    monkeypatch.delenv(UNSET, raising=False)


def web(tmp_path, port: int, **fields) -> dict:
    """A process service that answers GET /health with 200 on ``port``."""
    return {
        "kind": "process",
        "command": service_command(tmp_path, "--port", str(port)),
        "port": port,
        "health": {"kind": "http", "url": "http://127.0.0.1:%d/health" % port},
        **fields,
    }


def started(paths, *names: str) -> list[str]:
    """The lines that ``up`` prints when it starts ``names`` in this order, with the pid in each pid file."""
    return [
        line
        for name in names
        for line in (
            "%s: starting (argv and output in %s)" % (name, paths.log_file(name)),
            "%s: up (pid %d, log %s)" % (name, read_pid(paths, name), paths.log_file(name)),
        )
    ]


def numbered_lines(first: int, last: int) -> str:
    return "".join("line %d\n" % number for number in range(first, last + 1))


# --- up -----------------------------------------------------------------------


def test_up_starts_each_enabled_service_after_its_dependencies(paths, tmp_path, capsys):
    write_spec(
        paths, {"api": web(tmp_path, free_port(), depends_on=["db"]), "db": web(tmp_path, free_port())}
    )

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out.splitlines()) == (0, started(paths, "db", "api"))


def test_a_started_process_gets_its_exports_and_its_env_in_its_environment(paths, tmp_path):
    record = tmp_path / "record.json"
    port = free_port()
    command = service_command(tmp_path, "--record", str(record), "--port", str(port))
    write_spec(
        paths,
        {
            "api": web(
                tmp_path, port, command=command, exports={"API_URL": "{state}/api"}, env={"API_MODE": "test"}
            )
        },
    )

    cli.main(["up"])

    env = json.loads(record.read_text())["env"]
    assert (env.get("API_URL"), env.get("API_MODE")) == (str(paths.state / "api"), "test")


def test_up_leaves_a_healthy_service_alone(paths, tmp_path, capsys):
    """A proxy that someone started by hand on its port counts as up."""
    port, ready = free_port(), tmp_path / "ready"
    subprocess.Popen(service_command(tmp_path, "--port", str(port), "--ready", str(ready)))
    wait_until(ready.exists, what="the server started by hand")
    write_spec(paths, {"proxy": web(tmp_path, port, command=service_command(tmp_path, "--exit", "9"))})

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out, paths.pid_file("proxy").exists()) == (
        0,
        "proxy: already healthy (HTTP 200)\n",
        False,
    )


def test_up_skips_each_dependent_of_a_failed_service(paths, tmp_path, capsys):
    """One fault prints one failure, not a cascade of them."""
    crash = service_command(tmp_path, "--say", "no such config", "--exit", "3")
    write_spec(
        paths,
        {
            "bad": web(tmp_path, free_port(), command=crash),
            "dep": web(tmp_path, free_port(), depends_on=["bad"]),
            "dep2": web(tmp_path, free_port(), depends_on=["dep"]),
        },
    )

    code = cli.main(["up"])

    lines = capsys.readouterr().out.splitlines()
    assert (code, lines[1], lines[-2:], paths.pid_file("dep").exists(), paths.pid_file("dep2").exists()) == (
        1,
        "bad: FAILED: bad: exited with status 3 before it answered its health probe; last lines of %s:"
        % paths.log_file("bad"),
        ["dep: skipped (bad failed)", "dep2: skipped (dep failed)"],
        False,
        False,
    )


def test_an_unset_variable_in_a_command_fails_up_with_one_line_and_skips_the_dependents(
    paths, tmp_path, capsys
):
    write_spec(
        paths,
        {
            "a": {"kind": "process", "command": ["{env:%s}" % UNSET, "-c", "pass"]},
            "b": web(tmp_path, free_port(), depends_on=["a"]),
        },
    )

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out) == (
        1,
        "a: FAILED: environment variable %s is not set\nb: skipped (a failed)\n" % UNSET,
    )


def test_a_prepare_step_runs_before_the_first_start(paths, tmp_path):
    made = paths.state / "venv-made"
    port = free_port()
    prepare = {"run": [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('made')", str(made)]}
    command = service_command(tmp_path, "--require", str(made), "--port", str(port))
    write_spec(
        paths, {"proxy": web(tmp_path, port, command=command, prepare=[{**prepare, "unless": str(made)}])}
    )

    code = cli.main(["up"])

    assert (code, made.read_text()) == (0, "made")


def test_a_prepare_step_is_skipped_while_its_unless_path_exists(paths, tmp_path):
    made = paths.state / "venv-made"
    made.parent.mkdir(parents=True)
    made.write_text("kept")
    prepare = {"run": [sys.executable, "-c", "import sys; open(sys.argv[1], 'w').write('made')", str(made)]}
    write_spec(paths, {"proxy": web(tmp_path, free_port(), prepare=[{**prepare, "unless": str(made)}])})

    code = cli.main(["up"])

    assert (code, made.read_text()) == (0, "kept")


def test_a_failed_prepare_step_fails_up_and_starts_nothing(paths, tmp_path, capsys):
    prepare = [{"run": [sys.executable, "-c", "raise SystemExit(3)"]}]
    write_spec(paths, {"proxy": web(tmp_path, free_port(), prepare=prepare)})

    code = cli.main(["up"])

    lines = capsys.readouterr().out.splitlines()
    assert (code, lines[1], paths.pid_file("proxy").exists()) == (
        1,
        "proxy: FAILED: prepare step exited 3; last lines of %s:" % paths.log_file("proxy"),
        False,
    )


def test_up_does_not_start_a_second_instance_of_a_service_that_runs_but_is_not_healthy(
    paths, tmp_path, capsys
):
    """A second instance would fail on the port and hide the real fault."""
    ready = tmp_path / "ready"
    command = service_command(tmp_path, "--ready", str(ready))
    write_spec(paths, {"web": web(tmp_path, free_port(), command=command, timeout_s=1)})
    pid = start("web", command, dict(os.environ), paths, probe=ready_probe(ready), timeout_s=20)

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out, read_pid(paths, "web")) == (
        1,
        "web: FAILED: pid %d is running but not healthy (Connection refused); "
        "`logs web`, then `restart web`\n" % pid,
        pid,
    )


def test_a_process_health_service_is_up_once_it_stays_alive_for_the_settle_time(paths, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "SETTLE_S", 0.2)
    command = service_command(tmp_path)
    write_spec(paths, {"worker": {"kind": "process", "command": command, "health": {"kind": "process"}}})

    code = cli.main(["up"])

    assert (code, process_state(paths, "worker", command).alive) == (0, True)


def test_a_process_health_service_that_exits_at_once_fails_up(paths, tmp_path, capsys):
    """A worker that dies on its config does so within the first second, before the settle time."""
    command = service_command(tmp_path, "--say", "bad config", "--exit", "2")
    write_spec(paths, {"worker": {"kind": "process", "command": command, "health": {"kind": "process"}}})

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out.splitlines()[1]) == (
        1,
        "worker: FAILED: worker: exited with status 2 before it answered its health probe; last lines of %s:"
        % paths.log_file("worker"),
    )


def test_a_bare_up_leaves_out_a_disabled_service(paths, capsys):
    write_spec(
        paths, {"aenv": {"kind": "external", "enabled": False, "start_hint": "sudo /path/to/server &"}}
    )

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out) == (0, "")


def test_up_of_an_external_service_prints_its_start_hint_and_exits_1(paths, capsys):
    """An external service needs root, so the CLI only says how to start it."""
    aenv = {
        "kind": "external",
        "enabled": False,
        "health": {"kind": "http", "url": "http://127.0.0.1:%d/sandboxes" % free_port()},
        "start_hint": "sudo setsid /path/to/aenv-bench/bin/server &\ncurl -s http://127.0.0.1:18000/sandboxes\n",
    }
    write_spec(paths, {"aenv": aenv})

    code = cli.main(["up", "aenv"])

    assert (code, capsys.readouterr().out) == (
        1,
        "aenv: not running (Connection refused); it needs root, start it by hand:\n"
        "sudo setsid /path/to/aenv-bench/bin/server &\ncurl -s http://127.0.0.1:18000/sandboxes\n",
    )


# --- down and restart ---------------------------------------------------------


def test_down_stops_the_dependents_first_and_removes_each_pid_file(paths, tmp_path, capsys):
    write_spec(
        paths, {"db": web(tmp_path, free_port()), "api": web(tmp_path, free_port(), depends_on=["db"])}
    )
    cli.main(["up"])
    db, api = read_pid(paths, "db"), read_pid(paths, "api")
    capsys.readouterr()

    code = cli.main(["down"])

    assert (code, capsys.readouterr().out.splitlines(), is_running(db), is_running(api)) == (
        0,
        ["api: stopped (pid %d)" % api, "db: stopped (pid %d)" % db],
        False,
        False,
    )
    assert (paths.pid_file("db").exists(), paths.pid_file("api").exists()) == (False, False)


def test_down_of_a_service_also_stops_its_dependents_and_nothing_else(paths, tmp_path, capsys):
    """A service whose dependency stops is broken."""
    write_spec(
        paths,
        {
            "db": web(tmp_path, free_port()),
            "api": web(tmp_path, free_port(), depends_on=["db"]),
            "proxy": web(tmp_path, free_port()),
        },
    )
    cli.main(["up"])
    db, api, proxy = read_pid(paths, "db"), read_pid(paths, "api"), read_pid(paths, "proxy")
    capsys.readouterr()

    code = cli.main(["down", "db"])

    assert (code, capsys.readouterr().out.splitlines(), is_running(proxy)) == (
        0,
        ["api: stopped (pid %d)" % api, "db: stopped (pid %d)" % db],
        True,
    )


@pytest.mark.parametrize(
    "args, printed",
    [
        (["down"], "web: not running (no pid file)\n"),
        (["down", "aenv"], "aenv: external, nothing to stop\n"),
    ],
    ids=["bare", "named"],
)
def test_down_stops_managed_services_and_says_a_named_external_one_has_nothing_to_stop(
    paths, tmp_path, capsys, args, printed
):
    write_spec(
        paths, {"aenv": {"kind": "external", "start_hint": "sudo /path/to/server &"}, "web": web(tmp_path, 1)}
    )

    code = cli.main(args)

    assert (code, capsys.readouterr().out) == (0, printed)


def test_down_without_the_command_of_a_service_signals_nothing(paths, tmp_path, capsys):
    """The command proves that the pid in the pid file is the service's."""
    other = subprocess.Popen(service_command(tmp_path))
    paths.service("a").mkdir(parents=True)
    paths.pid_file("a").write_text("%d\n" % other.pid)
    write_spec(paths, {"a": {"kind": "process", "command": ["{env:%s}" % UNSET]}})

    code = cli.main(["down"])

    assert (code, capsys.readouterr().out, other.poll()) == (
        1,
        "a: FAILED: environment variable %s is not set\n" % UNSET,
        None,
    )


def test_restart_stops_a_service_and_its_dependents_then_starts_them_in_dependency_order(
    paths, tmp_path, capsys
):
    write_spec(
        paths, {"db": web(tmp_path, free_port()), "api": web(tmp_path, free_port(), depends_on=["db"])}
    )
    cli.main(["up"])
    db, api = read_pid(paths, "db"), read_pid(paths, "api")
    capsys.readouterr()

    code = cli.main(["restart", "db"])

    assert (code, capsys.readouterr().out.splitlines()) == (
        0,
        ["api: stopped (pid %d)" % api, "db: stopped (pid %d)" % db, *started(paths, "db", "api")],
    )


# --- status and logs ----------------------------------------------------------


def test_status_shows_one_row_for_each_service_with_its_pid_and_its_health(paths, tmp_path, capsys):
    port = free_port()
    write_spec(
        paths,
        {
            "web": web(tmp_path, port),
            "idle": {"kind": "process", "command": service_command(tmp_path), "health": {"kind": "process"}},
            "broken": {"kind": "process", "command": ["{env:%s}" % UNSET]},
            "aenv": {
                "kind": "external",
                "enabled": False,
                "port": 18000,
                "start_hint": "sudo /path/to/server &",
            },
        },
    )
    cli.main(["up", "web"])
    capsys.readouterr()

    code = cli.main(["status"])

    assert (code, table_rows(capsys.readouterr().out)) == (
        0,
        [
            ["name", "kind", "pid", "health", "port"],
            ["web", "process", str(read_pid(paths, "web")), "ok: HTTP 200", str(port)],
            ["idle", "process", "-", "down: no pid file", "-"],
            ["broken", "process", "-", "down: environment variable %s is not set" % UNSET, "-"],
            ["aenv", "external", "-", "disabled", "18000"],
        ],
    )


def test_status_never_makes_a_missing_secret(paths, capsys):
    """Only ``init``, ``up`` and ``env`` make a secret."""
    probe = {
        "kind": "http",
        "url": "http://127.0.0.1:%d/v1/jobs" % free_port(),
        "headers": {"Authorization": "Bearer {secret:runstore-token}"},
    }
    write_spec(paths, {"api": {"kind": "process", "command": ["true"], "health": probe}})

    cli.main(["status"])

    assert (table_rows(capsys.readouterr().out)[1], paths.secrets.exists()) == (
        [
            "api",
            "process",
            "-",
            "down: secret 'runstore-token' does not exist yet: run `python -m sprout.services init`",
            "-",
        ],
        False,
    )


@pytest.mark.parametrize("args, first", [(["-n", "3"], 98), ([], 51)])
def test_logs_prints_the_last_lines_of_the_service_log(paths, capsys, args, first):
    """The default is the last 50 lines."""
    write_spec(paths, {"web": {"kind": "process", "command": ["true"]}})
    paths.service("web").mkdir(parents=True)
    paths.log_file("web").write_text(numbered_lines(1, 100))

    code = cli.main(["logs", "web", *args])

    assert (code, capsys.readouterr().out) == (0, numbered_lines(first, 100))


def lines_of(stream) -> list[str]:
    """The lines that ``stream`` gives. A thread reads them, so the list grows while the test waits."""
    lines: list[str] = []

    def read() -> None:
        for line in stream:
            lines.append(line)

    threading.Thread(target=read, daemon=True).start()
    return lines


def test_logs_follow_prints_each_line_that_arrives_until_ctrl_c(paths, request):
    write_spec(paths, {"web": {"kind": "process", "command": ["true"]}})
    paths.service("web").mkdir(parents=True)
    paths.log_file("web").write_text("before\n")
    follower = subprocess.Popen(
        [sys.executable, "-m", "sprout.services", "logs", "web", "-f"], stdout=subprocess.PIPE, text=True
    )
    request.addfinalizer(follower.kill)
    printed = lines_of(follower.stdout)
    wait_until(lambda: printed == ["before\n"], what="the last lines of the log")

    with paths.log_file("web").open("a") as log:
        log.write("after\n")
    wait_until(lambda: printed == ["before\n", "after\n"], what="the line that arrived")
    follower.send_signal(signal.SIGINT)

    assert follower.wait(timeout=10) == 0


def test_logs_of_a_service_without_a_log_says_so(paths, capsys):
    write_spec(paths, {"web": {"kind": "process", "command": ["true"]}})

    code = cli.main(["logs", "web"])

    assert (code, capsys.readouterr().err) == (1, "no log yet at %s\n" % paths.log_file("web"))


# --- the environment of a process ---------------------------------------------


def test_a_process_gets_the_shell_without_unset_env_then_its_dependencies_exports_then_its_env(paths):
    spec = parse_spec(
        {
            "services": {
                "db": {
                    "kind": "process",
                    "command": ["true"],
                    "exports": {"DATA": "{state}/db", "SHARED": "db"},
                },
                "api": {
                    "kind": "process",
                    "command": ["true"],
                    "depends_on": ["db"],
                    "exports": {"CLOSE": "api-export"},
                    "env": {"CLOSE": "api-env"},
                    "unset_env": ["FORCE_COLOR"],
                },
            }
        }
    )
    manager = cli.Manager(
        spec, paths, {"PATH": "/usr/bin", "FORCE_COLOR": "3", "SHARED": "shell", "CLOSE": "shell"}
    )

    env = manager.environment(spec.get("api"), manager.ctx)

    assert env == {"PATH": "/usr/bin", "SHARED": "db", "CLOSE": "api-env", "DATA": str(paths.state / "db")}
