"""Wiring: every subcommand through argparse, against a spec of throwaway children.

The "services" are `python -c` servers on free ports; podman, the network and
the real ~/.config are never touched (the `paths` fixture moves every root).
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import sys

from sprout.services import __main__ as cli
from sprout.services import secrets
from sprout.services.spec import template_path

from tests.services.conftest import MARKER, free_port, server_command, sleeper_command, write_spec


def _web_spec(port: int) -> dict:
    return {
        "web": {"kind": "process", "command": server_command(port), "port": port,
                "health": {"kind": "http", "url": "http://127.0.0.1:%d/health" % port},
                "exports": {"WEB_TOKEN": "{secret:web-token}", "WEB_URL": "http://127.0.0.1:%d" % port}},
        "sleeper": {"kind": "process", "depends_on": ["web"], "command": sleeper_command(),
                    "env": {"MARK": "{secret:web-token}"}, "health": {"kind": "process"}},
    }


def _alive(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_init_writes_the_configs_and_secrets_and_never_overwrites(paths, capsys):
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert (paths.spec.read_text() == template_path("services.yaml").read_text()
            and (paths.config / "litellm.yaml").read_text() == template_path("litellm.yaml").read_text())
    runstore = json.loads((paths.config / "runstore.json").read_text())
    assert runstore["artifact_root"] == str(paths.state / "runstore-artifacts")
    assert runstore["profiles"]["mini"]["run_defaults"]["runtime_bin"] == str(paths.runtime_binary())
    # Every runtime_bin at any depth, whichever backend a profile runs on.
    assert runstore["profiles"]["mini"]["run_defaults"]["backend"]["podman"]["runtime_bin"] == str(paths.runtime_binary())
    assert (runstore["profiles"]["mini-microvm"]["run_defaults"]["backend"]["microvm"]["runtime_bin"]
            == str(paths.runtime_binary()))
    assert {p["python"] for p in runstore["profiles"].values()} == {paths.default_python()}
    rl_driver = json.loads((paths.config / "rl_driver.json").read_text())
    assert rl_driver["ledger"] == str(paths.state / "rl-driver" / "groups.sqlite3")
    assert rl_driver["driver_token_env"] == "SPROUT_RL_DRIVER_TOKEN" and rl_driver["runstore_url"] == "http://127.0.0.1:18110"
    names = sorted(f.name for f in paths.secrets.iterdir())
    assert names == ["db-password", "litellm-master-key", "rl-driver-token", "runstore-token"]
    assert stat.S_IMODE(paths.secrets.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o600 for f in paths.secrets.iterdir())
    assert out.count("wrote  ") == 4 and out.count("made   secret") == 4 and "eval \"$(python -m sprout.services env)\"" in out
    assert not any(secrets.require(paths, name) in out for name in names), "init never prints a value"

    (paths.config / "litellm.yaml").write_text("model_list: []\n")
    before = {name: secrets.require(paths, name) for name in names}
    assert cli.main(["init"]) == 0
    out = capsys.readouterr().out
    assert (paths.config / "litellm.yaml").read_text() == "model_list: []\n"
    assert out.count("kept   ") == 4 + 4 and "wrote  " not in out and "made   secret" not in out
    assert {name: secrets.require(paths, name) for name in names} == before
    home = paths.config.parent / "home"
    assert not (home / ".config").exists() and not (home / ".local").exists()


def test_up_status_down_round_trip_in_dependency_order(paths, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SETTLE_S", 0.3)
    port = free_port()
    write_spec(paths, _web_spec(port))
    assert cli.main(["up"]) == 0
    out = capsys.readouterr().out
    assert out.index("web: up (pid") < out.index("sleeper: up (pid"), out
    web_pid, sleeper_pid = (int(paths.pid_file(name).read_text()) for name in ("web", "sleeper"))
    token = secrets.require(paths, "web-token")

    assert cli.main(["status"]) == 0
    table = capsys.readouterr().out
    assert re.search(r"^web\s+\| process\s+\| %d\s+\| ok: HTTP 200\s+\| %d$" % (web_pid, port), table, re.M), table
    assert re.search(r"^sleeper\s+\| process\s+\| %d\s+\| ok: pid %d\s+\| -$" % (sleeper_pid, sleeper_pid), table, re.M), table
    assert token not in table
    assert "run: " not in paths.log_file("web").read_text()   # no prepare steps: only the start is logged

    assert cli.main(["up"]) == 0
    assert capsys.readouterr().out.count("already healthy") == 2
    assert (web_pid, sleeper_pid) == tuple(int(paths.pid_file(name).read_text()) for name in ("web", "sleeper"))

    assert cli.main(["down"]) == 0
    out = capsys.readouterr().out
    assert out.index("sleeper: stopped (pid %d)" % sleeper_pid) < out.index("web: stopped (pid %d)" % web_pid), out
    assert not _alive(web_pid) and not _alive(sleeper_pid)
    assert not paths.pid_file("web").exists() and not paths.pid_file("sleeper").exists()
    assert cli.main(["status"]) == 0
    assert capsys.readouterr().out.count("down: ") == 2
    assert cli.main(["down"]) == 0
    assert capsys.readouterr().out.count("not running (no pid file)") == 2


def test_down_of_a_dependency_takes_its_dependents_and_restart_brings_them_back(paths, capsys, monkeypatch):
    monkeypatch.setattr(cli, "SETTLE_S", 0.3)
    port = free_port()
    write_spec(paths, _web_spec(port))
    assert cli.main(["up"]) == 0
    first = int(paths.pid_file("web").read_text())
    capsys.readouterr()
    assert cli.main(["restart", "web"]) == 0
    out = capsys.readouterr().out
    # dependents down first, then the dependency up before them; one line per action
    assert [line.split(":")[0] for line in out.splitlines()] == ["sleeper", "web", "web", "web", "sleeper", "sleeper"], out
    assert "web: stopped (pid %d)" % first in out and int(paths.pid_file("web").read_text()) != first
    assert cli.main(["down", "sleeper"]) == 0
    assert capsys.readouterr().out.strip().startswith("sleeper: stopped") and paths.pid_file("web").exists()
    assert cli.main(["down"]) == 0


def test_env_prints_each_export_once_quoted_for_the_shell(paths, capsys):
    port = free_port()
    write_spec(paths, _web_spec(port))
    assert cli.main(["env"]) == 0
    lines = capsys.readouterr().out.splitlines()
    parsed = dict(shlex.split(line)[1].split("=", 1) for line in lines)
    assert all(shlex.split(line)[0] == "export" for line in lines)
    assert [line.split("=")[0] for line in lines] == ["export WEB_TOKEN", "export WEB_URL"]
    token = secrets.require(paths, "web-token")
    assert parsed == {"WEB_TOKEN": token, "WEB_URL": "http://127.0.0.1:%d" % port}
    assert cli.main(["status"]) == 0 and token not in capsys.readouterr().out


def test_a_failed_start_shows_the_log_and_skips_its_dependents(paths, capsys):
    port = free_port()
    write_spec(paths, {
        "bad": {"kind": "process", "command": [sys.executable, "-c", "import sys; print('no such config'); sys.exit(3)", MARKER],
                "health": {"kind": "http", "url": "http://127.0.0.1:%d/health" % port}},
        "dep": {"kind": "process", "depends_on": ["bad"], "command": sleeper_command(), "health": {"kind": "process"}},
    })
    assert cli.main(["up"]) == 1
    out = capsys.readouterr().out
    assert "bad: FAILED: bad: exited with status 3" in out and "no such config" in out
    assert "dep: skipped (bad failed)" in out and not paths.pid_file("dep").exists()


def test_an_unresolvable_placeholder_is_one_failed_line_never_a_traceback(paths, capsys, monkeypatch):
    """The file is valid; this shell lacks the variable. Every command names the
    service and goes on: a traceback from `status` would hide the other rows."""
    monkeypatch.delenv("SVC_TEST_NOPE", raising=False)
    write_spec(paths, {
        "a": {"kind": "process", "command": ["{env:SVC_TEST_NOPE}", "-c", "pass"], "health": {"kind": "process"}},
        "b": {"kind": "process", "depends_on": ["a"], "command": sleeper_command(), "health": {"kind": "process"}},
        "c": {"kind": "process", "command": sleeper_command(), "env": {"X": "{env:SVC_TEST_NOPE}"}, "health": {"kind": "process"}},
        "store": {"kind": "check", "check": "btrfs_store", "path": "{env:SVC_TEST_NOPE}/store"},
    })
    why = "environment variable SVC_TEST_NOPE is not set"
    assert cli.main(["status"]) == 0
    table = capsys.readouterr().out
    assert re.search(r"^a\s+\| process\s+\| -\s+\| down: %s\s+\| -$" % re.escape(why), table, re.M), table
    assert re.search(r"^store\s+\| check\s+\| -\s+\| fail: %s\s+\| -$" % re.escape(why), table, re.M), table
    assert cli.main(["up"]) == 1
    out = capsys.readouterr().out
    assert "a: FAILED: %s" % why in out and "b: skipped (a failed)" in out and "c: FAILED: %s" % why in out
    assert "store: fail %s" % why in out
    assert not paths.pid_file("b").exists() and not paths.pid_file("c").exists()
    assert cli.main(["down", "a"]) == 1
    assert "a: FAILED: %s" % why in capsys.readouterr().out        # without the command nothing is signalled
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert re.search(r"^fail  a\s+%s$" % re.escape(why), out, re.M) and re.search(r"^fail  store\s+%s$" % re.escape(why), out, re.M)


def test_up_of_an_external_service_prints_the_hint_and_starts_nothing(paths, capsys):
    write_spec(paths, {"aenv": {"kind": "external", "enabled": False,
                                "health": {"kind": "http", "url": "{env:AENV_SERVER_URL}/sandboxes"},
                                "start_hint": "sudo setsid env API_ADDR=127.0.0.1:18000 /opt/aenv-bench/bin/server &"}})
    assert cli.main(["up"]) == 0 and capsys.readouterr().out == ""        # disabled: a bare `up` skips it
    assert cli.main(["up", "aenv"]) == 1
    out = capsys.readouterr().out
    assert "aenv: not running (environment variable AENV_SERVER_URL is not set); it needs root" in out
    assert "sudo setsid env API_ADDR=127.0.0.1:18000 /opt/aenv-bench/bin/server &" in out
    assert not paths.state.exists() or not list(paths.state.iterdir())
    assert cli.main(["status"]) == 0
    assert re.search(r"^aenv\s+\| external\s+\| -\s+\| disabled\s+\| -$", capsys.readouterr().out, re.M)


def test_doctor_reports_a_store_that_is_not_btrfs(paths, capsys, tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    write_spec(paths, {"store": {"kind": "check", "check": "btrfs_store", "path": str(store)},
                       "missing": {"kind": "check", "check": "btrfs_store", "path": str(tmp_path / "nowhere")}})
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    store_line = next(line for line in out.splitlines() if line.startswith("fail  store"))
    assert str(store) in store_line and ("not btrfs" in store_line or "cannot tell" in store_line), store_line
    assert re.search(r"^fail  missing\s+%s does not exist; make it once as root" % re.escape(str(tmp_path / "nowhere")), out, re.M)
    assert re.search(r"^(ok|warn|fail)\s+keyring\s+kernel\.keys\.maxkeys", out, re.M)
    assert re.search(r"^(ok|fail)\s+runtime\s+", out, re.M) and re.search(r"^(ok|fail)\s+podman\s+", out, re.M)


def test_missing_spec_and_unknown_names_are_refused(paths, capsys):
    assert cli.main(["status"]) == 1
    assert "run `python -m sprout.services init` first" in capsys.readouterr().err
    write_spec(paths, {"a": {"kind": "process", "command": ["true"]}})
    assert cli.main(["up", "nope"]) == 1
    assert "unknown service nope (services.yaml has: a)" in capsys.readouterr().err
    paths.spec.write_text("services:\n  a:\n    kind: nope\n")
    assert cli.main(["status"]) == 1
    assert "service 'a', field 'kind'" in capsys.readouterr().err


def test_help_lists_every_subcommand(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "FORCE_COLOR"}
    result = subprocess.run([sys.executable, "-m", "sprout.services", "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ("init", "up", "down", "restart", "status", "logs", "env", "doctor"))
    assert "Eight commands" in result.stdout, "the package docstring is the description"


def test_logs_prints_the_tail_of_the_service_log(paths, capsys):
    write_spec(paths, {"a": {"kind": "process", "command": ["true"]}})
    assert cli.main(["logs", "a"]) == 1 and "no log yet" in capsys.readouterr().err
    paths.service("a").mkdir(parents=True)
    paths.log_file("a").write_text("".join("line %d\n" % i for i in range(100)))
    assert cli.main(["logs", "a", "-n", "3"]) == 0
    assert capsys.readouterr().out == "line 97\nline 98\nline 99\n"


def test_the_podman_check_reports_the_executable_and_env_exports_both_checks(paths, capsys, tmp_path, monkeypatch):
    """`status` and `doctor` show which podman answers --version; `env` exports the
    resolved podman and store, the way an experiment shell gets them."""
    fake = tmp_path / "bin" / "podman"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\necho 'podman version 9.9.9'\n")
    fake.chmod(0o755)
    store = tmp_path / "store"
    store.mkdir()
    monkeypatch.setenv("SPROUT_PODMAN_BIN", str(fake))
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(store))
    write_spec(paths, {"podman": {"kind": "check", "check": "podman", "exports": {"SPROUT_PODMAN_BIN": "{podman_bin}"}},
                       "store": {"kind": "check", "check": "btrfs_store", "exports": {"SPROUT_PODMAN_STORE": "{store}"}}})
    assert cli.main(["status"]) == 0
    out = capsys.readouterr().out
    assert re.search(r"^podman\s+\| check\s+\| -\s+\| ok: podman 9.9.9 \(%s\)" % re.escape(str(fake)), out, re.M), out
    assert cli.main(["env"]) == 0
    parsed = dict(shlex.split(line)[1].split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert parsed == {"SPROUT_PODMAN_BIN": str(fake), "SPROUT_PODMAN_STORE": str(store)}
    # a podman that is not there is a failed check, named
    monkeypatch.setenv("SPROUT_PODMAN_BIN", str(tmp_path / "nowhere" / "podman"))
    assert cli.main(["doctor"]) == 1
    assert "is not an executable podman" in capsys.readouterr().out
