"""``volatile_reason``: a journal under a directory that the OS empties on reboot is lost with it."""

from __future__ import annotations

import pytest

from sprout.contract.storage import volatile_reason

#: What follows the path and its directory in each reason.
LOSS = "which the OS empties on reboot -- a journal is the only record a run leaves"


@pytest.mark.parametrize(
    "path, resolved, root",
    [
        ("/tmp", "/tmp", "/tmp"),
        ("/var/tmp/x/deep/run.jsonl", "/var/tmp/x/deep/run.jsonl", "/var/tmp"),
        ("/dev/shm/j.jsonl", "/dev/shm/j.jsonl", "/dev/shm"),
        ("/run/sprout/runs", "/run/sprout/runs", "/run"),
        ("/path/to/../../tmp/runs", "/tmp/runs", "/tmp"),
    ],
    ids=["tmp", "var-tmp", "dev-shm", "run", "dot-dot-into-tmp"],
)
def test_a_path_under_a_directory_that_the_os_empties_is_volatile(path, resolved, root):
    reason = volatile_reason(path)

    assert reason == "%s is under %s, %s" % (resolved, root, LOSS)


def test_a_relative_path_is_resolved_against_the_working_directory(monkeypatch):
    monkeypatch.chdir("/")

    reason = volatile_reason("tmp/runs")

    assert reason == "/tmp/runs is under /tmp, %s" % LOSS


def test_a_path_that_only_starts_with_the_letters_of_such_a_directory_is_kept():
    assert volatile_reason("/tmpdir/runs") is None
