"""``python -m sprout.worker doctor``: the checks of the host that a worker runs on.

The podman is a shell script in the test's ``bin`` directory, and the store is a directory that the
fake ``btrfs`` and ``findmnt`` of ``tests/support`` accept. Only the keyring check reads the host,
so the tests leave its status out of their exact comparisons. docs/SERVICES.md describes the report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from sprout.worker.cli import main
from sprout.worker.doctor import Check, HostReport

PODMAN_VERSION = "#!/bin/sh\necho 'podman version 9.9.9'\n"


def podman_script(path: Path, body: str = PODMAN_VERSION) -> Path:
    """An executable at ``path`` that runs the shell ``body``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def test_the_json_report_names_each_check_and_the_podman_and_store_that_it_checked(
    tmp_path, fake_btrfs_store, fake_runtime_bin, capsys
):
    """``python -m sprout.services`` reads this object, so its keys are a wire format."""
    podman = podman_script(tmp_path / "opt" / "podman")
    argv = [
        "--podman-bin",
        str(podman),
        "--store",
        str(fake_btrfs_store),
        "--runtime-bin",
        str(fake_runtime_bin),
    ]

    code = main(["doctor", "--json", *argv])

    report = json.loads(capsys.readouterr().out)
    keyring = report["checks"].pop("keyring")
    assert (code, sorted(keyring)) == (0, ["detail", "status"])
    assert report == {
        "podman_bin": str(podman),
        "store": str(fake_btrfs_store),
        "checks": {
            "podman": {"status": "ok", "detail": "podman 9.9.9 (%s)" % podman},
            "store": {"status": "ok", "detail": "%s: btrfs, user_subvol_rm_allowed" % fake_btrfs_store},
            "runtime": {"status": "ok", "detail": str(fake_runtime_bin)},
        },
    }


def test_without_flags_the_doctor_checks_the_podman_and_the_store_that_every_run_uses(
    tmp_path, fake_btrfs_store, fake_runtime_bin, monkeypatch, capsys
):
    """The services export these two variables to the worker and to each experiment shell."""
    podman = podman_script(tmp_path / "opt" / "podman")
    monkeypatch.setenv("SPROUT_PODMAN_BIN", str(podman))
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(fake_btrfs_store))

    main(["doctor", "--json", "--runtime-bin", str(fake_runtime_bin)])

    report = json.loads(capsys.readouterr().out)
    assert (report["podman_bin"], report["store"]) == (str(podman), str(fake_btrfs_store))
    assert (report["checks"]["podman"]["status"], report["checks"]["store"]["status"]) == ("ok", "ok")


def test_each_failed_check_is_a_fail_line_and_the_doctor_exits_1(
    tmp_path, fake_btrfs_store, fake_findmnt, capsys
):
    fake_findmnt.report(fstype="ext4", options="rw,relatime")
    absent = tmp_path / "absent"
    argv = ["--podman-bin", str(absent / "podman"), "--store", str(fake_btrfs_store)]

    code = main(["doctor", *argv, "--runtime-bin", str(absent / "sprout-runtime")])

    lines = capsys.readouterr().out.splitlines()
    assert (code, lines[:3]) == (
        1,
        [
            "fail  podman   %s/podman is not an executable podman (ops/setup_podman.sh installs one under "
            "~/.local/opt/podman; $SPROUT_PODMAN_BIN names another)" % absent,
            "fail  store    store %s is on ext4 mounted with rw,relatime; the podman backend needs btrfs "
            "mounted with user_subvol_rm_allowed (sudo ops/btrfs_store.sh <mount point> <image file> <size>)"
            % fake_btrfs_store,
            "fail  runtime  %s/sprout-runtime is missing (cd runtime && go generate ./... && go build -o "
            "sprout-runtime .)" % absent,
        ],
    )
    assert re.fullmatch(r"(ok|warn)  +keyring  kernel\.keys\.maxkeys.*", lines[3]), lines[3:]


def test_a_podman_whose_version_command_fails_fails_its_check_with_the_exit_code_and_the_error(
    tmp_path, fake_btrfs_store, fake_runtime_bin, capsys
):
    podman = podman_script(
        tmp_path / "opt" / "podman", "#!/bin/sh\necho 'cannot reach the user service' >&2\nexit 125\n"
    )
    argv = [
        "--podman-bin",
        str(podman),
        "--store",
        str(fake_btrfs_store),
        "--runtime-bin",
        str(fake_runtime_bin),
    ]

    code = main(["doctor", "--json", *argv])

    check = json.loads(capsys.readouterr().out)["checks"]["podman"]
    assert (code, check) == (
        1,
        {"status": "fail", "detail": "%s --version exited 125: cannot reach the user service" % podman},
    )


def test_a_podman_named_without_a_directory_is_looked_up_on_path(
    tmp_path, fake_btrfs_store, fake_runtime_bin, capsys
):
    """The pool runs a bare name from PATH, so the doctor checks the podman that the pool would run."""
    podman = podman_script(tmp_path / "bin" / "podman")
    argv = [
        "--podman-bin",
        "podman",
        "--store",
        str(fake_btrfs_store),
        "--runtime-bin",
        str(fake_runtime_bin),
    ]

    main(["doctor", "--json", *argv])

    check = json.loads(capsys.readouterr().out)["checks"]["podman"]
    assert check == {"status": "ok", "detail": "podman 9.9.9 (%s)" % podman}


@pytest.mark.parametrize(
    "checks, ok",
    [
        ({"podman": Check("ok", "podman 9.9.9"), "keyring": Check("warn", "maxkeys=200")}, True),
        ({"podman": Check("ok", "podman 9.9.9"), "store": Check("fail", "not btrfs")}, False),
        ({"store": Check("fail", "not btrfs"), "keyring": Check("warn", "maxkeys=200")}, False),
    ],
    ids=["a warning", "a failure", "a warning and a failure"],
)
def test_a_warning_does_not_fail_the_report_and_a_failure_does(checks, ok):
    report = HostReport(podman_bin="podman", store="/path/to/store", checks=checks)

    assert report.ok is ok
