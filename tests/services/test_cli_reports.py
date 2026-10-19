"""``env``, ``doctor``, the check services, and what each command refuses, through ``main``.

A check service shows a verdict of the worker's doctor. The interpreter in services.yaml is a stub
that answers as ``python -m sprout.worker doctor --json``.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
import yaml

from sprout.services import __main__ as cli
from sprout.services import secrets
from sprout.services.spec import template_path
from tests.services.helpers import (
    StubProgram,
    doctor_report,
    free_port,
    make_stub,
    service_command,
    table_rows,
    write_spec,
)

UNSET = "SPROUT_SERVICES_TEST_UNSET"
DOCTOR = ["-m", "sprout.worker", "doctor", "--json"]


@pytest.fixture(autouse=True)
def unset_variable(monkeypatch):
    """Make sure that ``UNSET`` is not set, so that a placeholder that names it cannot be resolved."""
    monkeypatch.delenv(UNSET, raising=False)


@pytest.fixture
def python(tmp_path) -> StubProgram:
    """The interpreter that services.yaml names. Each test sets the report that it prints."""
    return make_stub(tmp_path / "venv" / "python")


def template_checks() -> dict:
    """The ``podman`` and ``store`` services of the packaged services.yaml, as a host's file has them."""
    services = yaml.safe_load(template_path("services.yaml").read_text())["services"]
    return {"podman": services["podman"], "store": services["store"]}


# --- env ----------------------------------------------------------------------


def test_env_prints_an_export_line_for_each_export_of_each_service_in_the_order_of_the_file(paths, capsys):
    dsn = "host=127.0.0.1 port=15432 password={secret:db-password}"
    proxy = {"OPENAI_BASE_URL": "http://127.0.0.1:4000/v1"}
    write_spec(
        paths,
        {
            "db": {"kind": "process", "command": ["true"], "exports": {"SPROUT_RUNSTORE_DSN": dsn}},
            "proxy": {"kind": "process", "command": ["true"], "exports": proxy},
        },
    )

    code = cli.main(["env"])

    assert (code, capsys.readouterr().out) == (
        0,
        "export SPROUT_RUNSTORE_DSN='host=127.0.0.1 port=15432 password=%s'\n"
        "export OPENAI_BASE_URL=http://127.0.0.1:4000/v1\n" % secrets.require(paths, "db-password"),
    )


def test_env_with_a_placeholder_that_cannot_be_resolved_prints_no_line_and_exits_1(paths, capsys):
    """A shell that evaluates a part of the lines would run with a part of the variables."""
    write_spec(
        paths,
        {
            "proxy": {
                "kind": "process",
                "command": ["true"],
                "exports": {"OPENAI_BASE_URL": "http://127.0.0.1:4000/v1"},
            },
            "aenv": {"kind": "external", "exports": {"AENV_SERVER_URL": "{env:%s}" % UNSET}},
        },
    )

    code = cli.main(["env"])

    captured = capsys.readouterr()
    assert (code, captured.out, captured.err) == (1, "", "environment variable %s is not set\n" % UNSET)


def test_env_exports_the_podman_and_the_store_that_the_workers_doctor_resolved(paths, python, capsys):
    python.answer(out=doctor_report(podman_bin="/path/to/podman", store="/path/to/store"))
    write_spec(paths, template_checks(), python=str(python.path))

    code = cli.main(["env"])

    assert (code, capsys.readouterr().out) == (
        0,
        "export SPROUT_PODMAN_BIN=/path/to/podman\nexport SPROUT_PODMAN_STORE=/path/to/store\n",
    )


# --- check services -----------------------------------------------------------


def test_status_shows_the_workers_verdict_on_each_check_service(paths, python, capsys):
    python.answer(
        out=doctor_report(checks={"store": ("fail", "store /path/to/store is on ext4 mounted with rw")})
    )
    write_spec(paths, template_checks(), python=str(python.path))

    code = cli.main(["status"])

    assert (code, table_rows(capsys.readouterr().out)[1:]) == (
        0,
        [
            ["podman", "check", "-", "ok: podman 9.9.9 (/path/to/podman)", "-"],
            ["store", "check", "-", "fail: store /path/to/store is on ext4 mounted with rw", "-"],
        ],
    )


def test_one_command_runs_the_workers_doctor_once(paths, python):
    python.answer(out=doctor_report())
    write_spec(paths, template_checks(), python=str(python.path))

    cli.main(["status"])

    assert python.calls() == [DOCTOR]


def test_a_check_with_a_path_asks_the_doctor_about_that_path_with_its_placeholders_expanded(
    paths, python, monkeypatch
):
    monkeypatch.setenv("SPROUT_TEST_STORES", "/path/to/stores")
    python.answer(out=doctor_report())
    write_spec(
        paths,
        {"store": {"kind": "check", "check": "btrfs_store", "path": "{env:SPROUT_TEST_STORES}/second"}},
        python=str(python.path),
    )

    cli.main(["status"])

    assert python.calls() == [[*DOCTOR, "--store", "/path/to/stores/second"]]


def test_a_check_whose_path_cannot_be_resolved_fails_and_does_not_stop_the_report(paths, python, capsys):
    python.answer(out=doctor_report())
    write_spec(
        paths,
        {"store": {"kind": "check", "check": "btrfs_store", "path": "{env:%s}/store" % UNSET}},
        python=str(python.path),
    )

    code = cli.main(["status"])

    assert (code, table_rows(capsys.readouterr().out)[1]) == (
        0,
        ["store", "check", "-", "fail: environment variable %s is not set" % UNSET, "-"],
    )


def test_up_of_a_failed_check_skips_each_service_that_depends_on_it(paths, python, tmp_path, capsys):
    """The worker's sandboxes need the store that the store check checks."""
    python.answer(
        out=doctor_report(checks={"store": ("fail", "store /path/to/store is on ext4 mounted with rw")})
    )
    worker = {"kind": "process", "command": service_command(tmp_path), "depends_on": ["store"]}
    write_spec(paths, {"store": template_checks()["store"], "worker": worker}, python=str(python.path))

    code = cli.main(["up"])

    assert (code, capsys.readouterr().out, paths.pid_file("worker").exists()) == (
        1,
        "store: fail store /path/to/store is on ext4 mounted with rw\nworker: skipped (store failed)\n",
        False,
    )


# --- doctor -------------------------------------------------------------------


def uv_on_path(tmp_path, monkeypatch) -> StubProgram:
    """A uv, alone on PATH."""
    uv = make_stub(tmp_path / "bin" / "uv")
    monkeypatch.setenv("PATH", str(uv.path.parent))
    return uv


def test_doctor_prints_uv_then_the_workers_other_checks_then_each_service(
    paths, python, tmp_path, monkeypatch, capsys
):
    uv = uv_on_path(tmp_path, monkeypatch)
    python.answer(
        out=doctor_report(
            checks={
                "store": ("fail", "store /path/to/store is on ext4 mounted with rw"),
                "keyring": ("warn", "kernel.keys.maxkeys=200 < 20000"),
            }
        )
    )
    port = free_port()
    web = {
        "kind": "process",
        "command": ["true"],
        "health": {"kind": "http", "url": "http://127.0.0.1:%d/" % port},
    }
    off = {"kind": "process", "command": ["true"], "enabled": False}
    write_spec(
        paths, {"podman": template_checks()["podman"], "web": web, "off": off}, python=str(python.path)
    )

    code = cli.main(["doctor"])

    assert (code, capsys.readouterr().out.splitlines()) == (
        1,
        [
            "ok    uv               %s" % uv.path,
            "fail  store            store /path/to/store is on ext4 mounted with rw",
            "ok    runtime          /path/to/sprout-runtime",
            "warn  keyring          kernel.keys.maxkeys=200 < 20000",
            "ok    podman           podman 9.9.9 (/path/to/podman)",
            "fail  web              Connection refused",
            "skip  off              disabled in services.yaml",
        ],
    )


def test_doctor_exits_0_when_the_only_problems_are_warnings(paths, python, tmp_path, monkeypatch):
    uv_on_path(tmp_path, monkeypatch)
    python.answer(out=doctor_report(checks={"keyring": ("warn", "kernel.keys.maxkeys=200 < 20000")}))
    write_spec(paths, template_checks(), python=str(python.path))

    assert cli.main(["doctor"]) == 0


def test_doctor_skips_a_disabled_check_service(paths, python, tmp_path, monkeypatch, capsys):
    """docs/SERVICES.md: ``enabled: false`` leaves a service out of a bare up, of status and of doctor."""
    uv_on_path(tmp_path, monkeypatch)
    python.answer(
        out=doctor_report(checks={"store": ("fail", "store /path/to/store is on ext4 mounted with rw")})
    )
    write_spec(paths, {"store": {**template_checks()["store"], "enabled": False}}, python=str(python.path))

    code = cli.main(["doctor"])

    assert (code, capsys.readouterr().out.splitlines()[-1]) == (
        0,
        "skip  store            disabled in services.yaml",
    )


# --- what each command refuses ------------------------------------------------


@pytest.mark.parametrize(
    "command", [["up"], ["down"], ["restart", "db"], ["status"], ["logs", "db"], ["env"], ["doctor"]]
)
def test_each_command_without_services_yaml_exits_1_and_says_to_run_init(paths, capsys, command):
    code = cli.main(command)

    assert (code, capsys.readouterr().err) == (
        1,
        "no %s: run `python -m sprout.services init` first\n" % paths.spec,
    )


def test_an_invalid_services_yaml_is_refused_with_the_service_and_the_field_named(paths, capsys):
    paths.config.mkdir(parents=True)
    paths.spec.write_text("services:\n  db:\n    kind: postgres\n")

    code = cli.main(["status"])

    assert (code, capsys.readouterr().err.startswith("%s: service 'db', field 'kind': " % paths.spec)) == (
        1,
        True,
    )


@pytest.mark.parametrize("command", [["up", "nope"], ["down", "nope"], ["restart", "nope"], ["logs", "nope"]])
def test_a_command_that_names_an_unknown_service_exits_1_and_lists_the_known_ones(paths, capsys, command):
    write_spec(
        paths,
        {"db": {"kind": "process", "command": ["true"]}, "api": {"kind": "process", "command": ["true"]}},
    )

    code = cli.main(command)

    assert (code, capsys.readouterr().err) == (1, "unknown service nope (services.yaml has: db, api)\n")


def test_python_dash_m_sprout_services_exits_with_the_status_of_the_command(paths):
    done = subprocess.run(
        [sys.executable, "-m", "sprout.services", "status"],
        env={key: value for key, value in os.environ.items() if key != "FORCE_COLOR"},
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (done.returncode, done.stdout, done.stderr) == (
        1,
        "",
        "no %s: run `python -m sprout.services init` first\n" % paths.spec,
    )
