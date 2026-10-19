"""The worker's doctor as the services read it: one child process for each set of flags in a command,
and a failed check, never an exception, when the doctor gives no report.

Most tests name a stub interpreter that prints a fixed report. One test runs the real
``python -m sprout.worker doctor --json`` on the fake podman, btrfs and findmnt, because the JSON
is a wire format between the worker and the services.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from sprout.services.health import HostError, HostReport, Probe, WorkerDoctor
from tests.services.helpers import StubProgram, doctor_report, make_stub

DOCTOR = ["-m", "sprout.worker", "doctor", "--json"]
#: A report with each status: the store check fails and the keyring check warns.
MIXED_REPORT = doctor_report(
    checks={
        "store": ("fail", "store /path/to/store is on ext4 mounted with rw"),
        "keyring": ("warn", "kernel.keys.maxkeys=200 < 20000"),
    }
)


def stub_python(tmp_path: Path) -> StubProgram:
    """An interpreter that answers ``-m sprout.worker doctor --json`` as the test sets."""
    return make_stub(tmp_path / "venv" / "python")


def doctor_of(python: StubProgram) -> WorkerDoctor:
    return WorkerDoctor(python=str(python.path), environ={})


def test_the_report_holds_the_podman_the_store_and_each_check_with_its_status(tmp_path):
    python = stub_python(tmp_path)
    python.answer(out=MIXED_REPORT)

    report = doctor_of(python).report()

    assert report == HostReport(
        podman_bin="/path/to/podman",
        store="/path/to/store",
        checks={
            "podman": Probe(True, "podman 9.9.9 (/path/to/podman)"),
            "store": Probe(False, "store /path/to/store is on ext4 mounted with rw"),
            "runtime": Probe(True, "/path/to/sprout-runtime"),
            "keyring": Probe(False, "kernel.keys.maxkeys=200 < 20000", warning=True),
        },
    )


def test_the_doctor_runs_once_for_the_checks_and_facts_of_one_command(tmp_path):
    python = stub_python(tmp_path)
    python.answer(out=MIXED_REPORT)
    doctor = doctor_of(python)

    doctor.check(kind="podman", path="")
    doctor.check(kind="btrfs_store", path="")
    doctor.fact("podman_bin")
    doctor.other_checks(["podman"])

    assert python.calls() == [DOCTOR]


@pytest.mark.parametrize(
    "kind, flag, check", [("podman", "--podman-bin", "podman"), ("btrfs_store", "--store", "store")]
)
def test_a_check_with_a_path_shows_the_doctors_verdict_on_that_path(tmp_path, kind, flag, check):
    python = stub_python(tmp_path)
    python.answer(out=doctor_report(checks={check: ("ok", "checked /path/to/other")}))

    result = doctor_of(python).check(kind=kind, path="/path/to/other")

    assert (result, python.calls()) == (
        Probe(True, "checked /path/to/other"),
        [[*DOCTOR, flag, "/path/to/other"]],
    )


def test_a_report_without_the_check_of_a_service_fails_that_check(tmp_path):
    python = stub_python(tmp_path)
    report = {"podman_bin": "/path/to/podman", "store": "/path/to/store", "checks": {}}
    python.answer(out=json.dumps(report))

    result = doctor_of(python).check(kind="btrfs_store", path="")

    assert result == Probe(False, "the worker's doctor reports no store check")


def test_other_checks_are_the_checks_that_no_check_service_shows_in_the_order_of_the_report(tmp_path):
    python = stub_python(tmp_path)
    python.answer(out=MIXED_REPORT)

    others = doctor_of(python).other_checks(["podman"])

    assert others == [
        ("store", Probe(False, "store /path/to/store is on ext4 mounted with rw")),
        ("runtime", Probe(True, "/path/to/sprout-runtime")),
        ("keyring", Probe(False, "kernel.keys.maxkeys=200 < 20000", warning=True)),
    ]


def test_a_doctor_that_gives_no_report_fails_a_check_with_its_last_line(tmp_path):
    python = stub_python(tmp_path)
    python.answer(
        err="Traceback (most recent call last):\nModuleNotFoundError: No module named 'sprout'\n", code=1
    )

    result = doctor_of(python).check(kind="podman", path="")

    assert result == Probe(
        False,
        "%s -m sprout.worker doctor --json gave no report (exit 1): "
        "ModuleNotFoundError: No module named 'sprout'" % python.path,
    )


def test_without_a_report_the_other_checks_are_one_failed_worker_check(tmp_path):
    python = stub_python(tmp_path)
    python.answer(err="boom\n", code=1)

    others = doctor_of(python).other_checks([])

    assert others == [
        (
            "worker",
            Probe(False, "%s -m sprout.worker doctor --json gave no report (exit 1): boom" % python.path),
        )
    ]


def test_a_doctor_that_gave_no_report_is_not_asked_again_in_the_same_command(tmp_path):
    python = stub_python(tmp_path)
    python.answer(err="boom\n", code=1)
    doctor = doctor_of(python)

    doctor.check(kind="podman", path="")
    doctor.other_checks([])
    with pytest.raises(HostError, match="gave no report"):
        doctor.fact("store")

    assert python.calls() == [DOCTOR]


@pytest.mark.parametrize(
    "out",
    [
        "Traceback (most recent call last):\n",
        "[]",
        json.dumps({"store": "/path/to/store", "checks": {}}),
        json.dumps(
            {"podman_bin": "p", "store": "s", "checks": {"podman": {"status": "maybe", "detail": "x"}}}
        ),
        json.dumps({"podman_bin": "p", "store": "s", "checks": {"podman": {"status": "ok", "detail": 5}}}),
    ],
)
def test_output_of_another_shape_counts_as_no_report(tmp_path, out):
    python = stub_python(tmp_path)
    python.answer(out=out, err="last words\n")

    result = doctor_of(python).check(kind="podman", path="")

    assert result == Probe(
        False, "%s -m sprout.worker doctor --json gave no report (exit 0): last words" % python.path
    )


def test_an_interpreter_that_does_not_exist_fails_a_check_with_the_os_error(tmp_path):
    python = tmp_path / "absent" / "python"

    result = WorkerDoctor(python=str(python), environ={}).check(kind="podman", path="")

    assert result == Probe(
        False,
        "%s -m sprout.worker doctor --json: [Errno 2] No such file or directory: '%s'" % (python, python),
    )


def test_the_report_of_the_real_doctor_reads_as_the_services_expect(fake_podman, fake_btrfs_store):
    """The fake findmnt reports the mount that ``ops/btrfs_store.sh`` makes, so the store check passes."""
    doctor = WorkerDoctor(python=sys.executable, environ=os.environ)

    report = doctor.report(("--store", str(fake_btrfs_store)))

    assert (report.podman_bin, report.store, report.checks["store"], sorted(report.checks)) == (
        "podman",
        str(fake_btrfs_store),
        Probe(True, "%s: btrfs, user_subvol_rm_allowed" % fake_btrfs_store),
        ["keyring", "podman", "runtime", "store"],
    )
