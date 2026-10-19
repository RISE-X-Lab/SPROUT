"""``python -m sprout.worker reap``: what it frees from a resource ledger, what it leaves, what it refuses.

Podman is the fake podman of ``tests/support``, and its store is one that the fake ``btrfs`` and
``findmnt`` accept. AgentENV is a closed local port, so it lists no microVM.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from sprout.worker.cli import main
from sprout.worker.run.resources import ResourceLedger
from tests.support.tools import FakePodman, ManagedContainer

SANDBOX = "sprout-0000000000b2"


@pytest.fixture(autouse=True)
def no_agentenv(monkeypatch) -> None:
    monkeypatch.setenv("AENV_SERVER_URL", "http://127.0.0.1:1")


def ledger_of_a_dead_run(path: Path, *, store: Path | None) -> Path:
    """A ledger in which the finished run ``run-1`` claimed SANDBOX on podman and never released it."""
    with ResourceLedger(path).run("run-1") as claim:
        claim.sandbox(SANDBOX, backend="podman", store=str(store) if store else None)
    return path


def running(podman: FakePodman, store: Path) -> None:
    """Make SANDBOX a running container of ``store``."""
    sockets = store / "sockets" / SANDBOX
    podman.containers_exist(ManagedContainer(name=SANDBOX, store=store, sockets=sockets, created=1700000000))


def test_a_dry_run_lists_each_orphan_with_its_reason_and_frees_nothing(
    tmp_path, fake_podman, fake_btrfs_store, capsys
):
    ledger = ledger_of_a_dead_run(tmp_path / "resources.jsonl", store=fake_btrfs_store)
    running(fake_podman, fake_btrfs_store)

    code = main(["reap", "--ledger", str(ledger), "--dry-run"])

    assert (code, capsys.readouterr().out) == (
        0,
        "  sandbox   podman   %s  orphan of run run-1 (pid %d gone)\n" % (SANDBOX, os.getpid())
        + "\n1 sandbox(es) would be freed (dry run)\n",
    )
    assert ["rm", "-f", SANDBOX] not in fake_podman.calls()


def test_a_reap_frees_each_orphan_of_the_ledger(tmp_path, fake_podman, fake_btrfs_store, capsys):
    ledger = ledger_of_a_dead_run(tmp_path / "resources.jsonl", store=fake_btrfs_store)
    running(fake_podman, fake_btrfs_store)

    code = main(["reap", "--ledger", str(ledger)])

    assert (code, capsys.readouterr().out) == (
        0,
        "  sandbox   podman   %s  orphan of run run-1 (pid %d gone)\n" % (SANDBOX, os.getpid())
        + "\nfreed 1 sandbox(es)\n",
    )
    assert ["rm", "-f", SANDBOX] in fake_podman.calls()


def test_a_sandbox_that_cannot_be_freed_is_named_and_the_reap_exits_1(
    tmp_path, fake_podman, fake_btrfs_store, capsys
):
    ledger = ledger_of_a_dead_run(tmp_path / "resources.jsonl", store=fake_btrfs_store)
    running(fake_podman, fake_btrfs_store)
    fake_podman.fail("rm")

    code = main(["reap", "--ledger", str(ledger)])

    assert (code, capsys.readouterr().err) == (
        1,
        "podman rm: Error: simulated rm failure\nfailed: %s\n" % SANDBOX,
    )


def test_a_kept_sandbox_is_left_and_counted(tmp_path, fake_podman, fake_btrfs_store, capsys):
    ledger = tmp_path / "resources.jsonl"
    with ResourceLedger(ledger).run("rollout") as claim:
        claim.sandbox(SANDBOX, keep=True, backend="podman", keep_until="2999-01-01T00:00:00+00:00")
    running(fake_podman, fake_btrfs_store)

    code = main(["reap", "--ledger", str(ledger)])

    assert (code, capsys.readouterr().out) == (0, "nothing to reap (1 kept)\n")


def test_a_rootfs_whose_container_is_gone_is_found_in_the_store_of_its_claim(
    tmp_path, fake_podman, fake_btrfs_store
):
    """A destroy that failed half-way removed the container and left the rootfs."""
    ledger = ledger_of_a_dead_run(tmp_path / "resources.jsonl", store=fake_btrfs_store)
    rootfs = fake_btrfs_store / "sandboxes" / SANDBOX
    rootfs.mkdir(parents=True)

    code = main(["reap", "--ledger", str(ledger)])

    assert (code, rootfs.exists()) == (0, False)


def test_a_rootfs_whose_claim_names_no_store_is_found_in_a_store_given_with_the_flag(
    tmp_path, fake_podman, fake_btrfs_store
):
    ledger = ledger_of_a_dead_run(tmp_path / "resources.jsonl", store=None)
    rootfs = fake_btrfs_store / "sandboxes" / SANDBOX
    rootfs.mkdir(parents=True)

    code = main(["reap", "--ledger", str(ledger), "--store", str(fake_btrfs_store)])

    assert (code, rootfs.exists()) == (0, False)


def test_a_sweep_by_age_alone_is_refused_without_an_age(tmp_path, fake_podman, capsys):
    """Without --older-than, --include-unknown would free every sandbox of the host, a colleague's too."""
    code = main(["reap", "--ledger", str(tmp_path / "resources.jsonl"), "--include-unknown"])

    assert (code, capsys.readouterr().err, fake_podman.calls()) == (
        2,
        "refusing --include-unknown without --older-than\n",
        [],
    )
