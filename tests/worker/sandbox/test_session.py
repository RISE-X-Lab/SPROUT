"""``SandboxSession``: one sandbox's lifetime, its tool channel, its snapshots, a re-board and its files.

A session builds its pool from its backend section. Here ``use_pool`` makes it build a ``StubPool``,
which records what the session asks of it (``tests/worker/sandbox/pools.py``). The tests of
misconfiguration and of the image config use the real ``build_pool`` and a stub ``regctl``.
``test_live_podman.py`` runs a session on the real podman backend.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.worker.sandbox.session import OWNER_AGENT_ID, SandboxSession
from sprout.worker.tools.result import ToolResult
from sprout_sandbox import Snapshot
from tests.worker.sandbox.pools import NEVER, NetworkedPool, SquashingPool, StubPool, StubRuntime, created

IMAGE = "docker.io/library/python:3.11-slim"

# --- the lifetime of the sandbox -------------------------------------------------------


def test_a_session_names_the_sandbox_that_it_created(use_pool):
    use_pool(StubPool())
    session = SandboxSession(quiet=True)
    before = session.sandbox_id

    ready = session.create(IMAGE)

    assert (before, ready, session.sandbox_id, session.create_error) == ("unknown", True, "sandbox-1", "")


def test_a_create_that_the_backend_refuses_returns_false_with_the_reason(use_pool):
    """A quiet caller has no warning to read, so it puts ``create_error`` in its own error."""
    use_pool(StubPool(refuse={"spawn_from_image"}))
    session = SandboxSession(quiet=True)

    ready = session.create(IMAGE)

    assert (ready, session.create_error, session.sandbox_id) == (
        False,
        "RuntimeError: spawn_from_image refused",
        "unknown",
    )


@pytest.mark.parametrize("start", [SandboxSession.create, SandboxSession.attach], ids=["create", "attach"])
def test_a_misconfigured_backend_fails_the_start_with_the_setting_that_is_wrong(start):
    session = SandboxSession(quiet=True, backend={"backend": "docker"})

    ready = start(session, IMAGE)

    assert (ready, session.create_error) == (
        False,
        "BackendError: unknown backend 'docker'; the backends are 'microvm', 'podman'",
    )


@pytest.mark.parametrize(
    "quiet, stderr",
    [(True, ""), (False, "  ! Failed to create sandbox: spawn_from_image refused\n")],
    ids=["quiet", "not-quiet"],
)
def test_only_a_session_that_is_not_quiet_reports_a_failure_on_stderr(use_pool, capsys, quiet, stderr):
    use_pool(StubPool(refuse={"spawn_from_image"}))

    SandboxSession(quiet=quiet).create(IMAGE)

    assert capsys.readouterr().err == stderr


def test_destroy_removes_the_sandbox_and_leaves_the_session_without_one(use_pool):
    pool = use_pool(StubPool())
    session = created()

    session.destroy()

    assert pool.operations[-1] == ("destroy", "sandbox-1")
    assert session.execute("shell", {"command": "ls"}) == ToolResult(False, "", "No active sandbox")


def test_a_destroy_that_fails_is_no_exception_and_the_session_lets_the_sandbox_go(use_pool):
    """The sandbox leaks until its TTL or ``sprout.worker reap``, which is no reason to fail a teardown."""
    pool = use_pool(StubPool(refuse={"destroy"}))
    session = created()

    session.destroy()

    assert (pool.operations[-1], session.sandbox_id) == (("destroy", "sandbox-1"), "unknown")


# --- the tool channel ----------------------------------------------------------------------


@pytest.mark.parametrize("run", [SandboxSession.execute, SandboxSession.call], ids=["execute", "call"])
def test_each_call_reaches_the_runtime_as_the_owner_and_an_agent_id_in_its_arguments_is_dropped(
    use_pool, run
):
    """A model that writes an ``agent_id`` argument cannot name itself."""
    pool = use_pool(StubPool())
    session = created()

    result = run(session, "shell", {"command": "ls", "agent_id": "spoofed"})

    assert result == ToolResult(success=True, output="ran shell")
    assert pool.runtimes["sandbox-1"].calls == [("shell", {"command": "ls", "timeout": 300}, OWNER_AGENT_ID)]


@pytest.mark.parametrize(
    "tool, args, sent",
    [
        ("shell", {"command": "ls"}, {"command": "ls", "timeout": 42}),
        ("shell", {"command": "ls", "timeout": 5}, {"command": "ls", "timeout": 5}),
        ("text_editor", {"command": "view", "path": "/a"}, {"command": "view", "path": "/a"}),
    ],
    ids=["shell", "shell-with-its-own-timeout", "another-tool"],
)
def test_a_shell_call_gets_the_given_timeout_unless_it_names_its_own(use_pool, tool, args, sent):
    pool = use_pool(StubPool())
    session = created()

    session.execute(tool, args, timeout=42.0)

    assert pool.runtimes["sandbox-1"].calls == [(tool, sent, OWNER_AGENT_ID)]


def test_execute_without_a_sandbox_gives_a_failed_result():
    assert SandboxSession(quiet=True).execute("shell", {"command": "ls"}) == ToolResult(
        False, "", "No active sandbox"
    )


def test_call_without_a_sandbox_raises():
    with pytest.raises(RuntimeError, match="^No active sandbox$"):
        SandboxSession(quiet=True).call("shell", {"command": "ls"})


def test_execute_returns_a_transport_failure_as_a_failed_result(use_pool):
    """For the owner's own commands: a reset, an extraction, a test run."""
    use_pool(
        StubPool(runtimes={"sandbox-1": StubRuntime(answers={"ls": ConnectionResetError("reset by peer")})})
    )
    session = created()

    result = session.execute("shell", {"command": "ls"})

    assert result == ToolResult(success=False, output="", error="reset by peer")


def test_call_raises_a_transport_failure(use_pool):
    """The agent's caller must tell "the command failed" from "nobody knows whether it ran"."""
    use_pool(
        StubPool(runtimes={"sandbox-1": StubRuntime(answers={"ls": ConnectionResetError("reset by peer")})})
    )
    session = created()

    with pytest.raises(ConnectionResetError, match="^reset by peer$"):
        session.call("shell", {"command": "ls"})


# --- snapshots ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options, operation",
    [
        ({}, ("snapshot", "sandbox-1", None, True)),
        ({"name": "final", "disk_only": False}, ("snapshot", "sandbox-1", "final", False)),
    ],
    ids=["default", "named-with-memory"],
)
def test_a_snapshot_is_disk_only_unless_asked_and_the_sandbox_keeps_running(use_pool, options, operation):
    """A replay restores the disk and re-feeds the transcript, so a memory image at each step buys nothing."""
    pool = use_pool(StubPool())
    session = created()

    snapshot = session.snapshot(**options)

    assert snapshot == Snapshot(id="sandbox-1-snapshot", disk_only=operation[3])
    assert (pool.operations[-1], session.sandbox_id) == (operation, "sandbox-1")


def test_a_snapshot_that_fails_gives_none_instead_of_failing_the_episode(use_pool):
    use_pool(StubPool(refuse={"snapshot"}))
    session = created()

    assert session.snapshot() is None


def test_a_backend_that_cannot_snapshot_gives_no_snapshot_and_is_not_asked_for_one(use_pool):
    pool = use_pool(StubPool(can_snapshot=False))
    session = created()

    snapshot = session.snapshot()

    assert (session.supports_snapshot(), snapshot, pool.operations[-1][0]) == (
        False,
        None,
        "spawn_from_image",
    )


def test_a_session_without_a_sandbox_gives_no_snapshot():
    session = SandboxSession(quiet=True)

    assert (session.supports_snapshot(), session.snapshot()) == (False, None)


def test_a_backend_that_can_squash_gives_the_flattened_snapshot(use_pool):
    use_pool(SquashingPool())
    session = created()

    squashed = session.squash_snapshot(Snapshot(id="snap-7", rootfs_layers=130), name="flat")

    assert squashed == Snapshot(id="snap-7-flat", rootfs_layers=1)


@pytest.mark.parametrize(
    "pool_class, options",
    [(StubPool, {}), (SquashingPool, {"refuse": {"squash"}}), (SquashingPool, {"can_snapshot": False})],
    ids=["no-squash-like-podman", "squash-fails", "cannot-snapshot"],
)
def test_a_squash_that_the_backend_cannot_do_gives_back_the_snapshot_that_it_was_given(
    use_pool, pool_class, options
):
    """A deep chain still works. Only the checkpoints of its children cost more."""
    use_pool(pool_class(**options))
    session = created()
    snapshot = Snapshot(id="snap-7", rootfs_layers=130)

    assert session.squash_snapshot(snapshot) is snapshot


# --- a re-board ---------------------------------------------------------------------------


@pytest.mark.parametrize("snapshot", [Snapshot(id="snap-7"), "snap-7"], ids=["snapshot", "bare-id"])
def test_a_reboard_continues_on_a_sandbox_from_the_snapshot_and_destroys_the_old_one(use_pool, snapshot):
    """The agent does not see the swap: its next call goes to the replacement."""
    pool = use_pool(StubPool())
    session = created()

    moved = session.swap_sandbox(snapshot)
    session.execute("shell", {"command": "ls"})

    assert (moved, session.sandbox_id) == (True, "sandbox-2")
    assert pool.operations[1:] == [("spawn", "snap-7"), ("destroy", "sandbox-1")]
    assert pool.runtimes["sandbox-2"].calls[-1] == (
        "shell",
        {"command": "ls", "timeout": 300},
        OWNER_AGENT_ID,
    )


def test_a_swap_listener_learns_of_the_replacement_before_the_old_sandbox_is_destroyed(use_pool):
    """A listener that learned late would serve a call into a destroyed sandbox."""
    pool = use_pool(StubPool())
    session = created()
    heard = []
    session.on_swap.append(lambda replacement: heard.append((replacement.sandbox_id, list(pool.operations))))

    session.swap_sandbox("snap-7")

    assert heard == [("sandbox-2", [("spawn_from_image", IMAGE, None), ("spawn", "snap-7")])]


def test_a_swap_listener_that_raises_stops_neither_the_reboard_nor_the_listeners_after_it(use_pool):
    use_pool(StubPool())
    session = created()
    heard = []

    def broken(replacement: object) -> None:
        raise RuntimeError("listener bug")

    session.on_swap.extend([broken, heard.append])

    moved = session.swap_sandbox("snap-7")

    assert (moved, session.sandbox_id, len(heard)) == (True, "sandbox-2", 1)


def test_a_reboard_whose_spawn_fails_keeps_the_sandbox(use_pool):
    """A pool that removes a replacement whose runtime never answers raises from the spawn."""
    pool = use_pool(StubPool(refuse={"spawn"}))
    session = created()

    moved = session.swap_sandbox("snap-7")

    assert (moved, session.sandbox_id, pool.operations[-1]) == (False, "sandbox-1", ("spawn", "snap-7"))


def test_a_reboard_target_whose_runtime_does_not_answer_is_destroyed_and_the_sandbox_kept(use_pool):
    """Every call to an unreachable replacement would fail. The old sandbox only costs a deeper chain."""
    pool = use_pool(StubPool(runtimes={"sandbox-2": StubRuntime(starting_calls=NEVER)}))
    session = created()
    heard = []
    session.on_swap.append(heard.append)

    moved = session.swap_sandbox("snap-7")

    assert (moved, session.sandbox_id, heard) == (False, "sandbox-1", [])
    assert pool.operations[1:] == [("spawn", "snap-7"), ("destroy", "sandbox-2")]


def test_a_reboard_that_cannot_destroy_the_old_sandbox_still_moves(use_pool):
    """The replacement serves the calls. The old sandbox is a leak for its TTL or the reaper."""
    use_pool(StubPool(refuse={"destroy"}))
    session = created()

    moved = session.swap_sandbox("snap-7")

    assert (moved, session.sandbox_id) == (True, "sandbox-2")


@pytest.mark.parametrize("quiet", [True, False], ids=["quiet", "not-quiet"])
def test_a_progress_line_cannot_break_the_reboard_that_it_reports(use_pool, quiet):
    """A progress line names the sandbox, also when the sandbox object cannot name itself."""
    use_pool(StubPool(nameless=True))
    session = SandboxSession(quiet=quiet)
    session.create(IMAGE)

    moved = session.swap_sandbox("snap-7")

    assert (moved, session.sandbox_id) == (True, "unknown")


# --- a sandbox that another session kept ------------------------------------------------


def test_attach_drives_a_running_sandbox_and_destroy_then_removes_it(use_pool):
    """A grade attaches to the sandbox that its rollout kept."""
    pool = use_pool(StubPool())
    session = SandboxSession(quiet=True)

    attached = session.attach("sandbox-kept")
    session.execute("shell", {"command": "ls"})
    session.destroy()

    assert (attached, session.create_error) == (True, "")
    assert pool.runtimes["sandbox-kept"].calls[-1] == (
        "shell",
        {"command": "ls", "timeout": 300},
        OWNER_AGENT_ID,
    )
    assert pool.operations == [("attach", "sandbox-kept"), ("destroy", "sandbox-kept")]


def test_attach_waits_for_a_runtime_that_is_still_starting(use_pool):
    use_pool(StubPool(runtimes={"sandbox-kept": StubRuntime(starting_calls=1)}))
    session = SandboxSession(quiet=True)

    assert session.attach("sandbox-kept") is True


def test_attach_to_a_sandbox_that_does_not_answer_forgets_it_without_destroying_it(use_pool):
    """What is left of it belongs to its owner's ledger, which reclaims it."""
    pool = use_pool(StubPool(runtimes={"sandbox-kept": StubRuntime(starting_calls=NEVER)}))
    session = SandboxSession(quiet=True)

    attached = session.attach("sandbox-kept")

    assert (attached, session.create_error, session.sandbox_id) == (
        False,
        "sandbox sandbox-kept does not answer",
        "unknown",
    )
    assert pool.operations == [("attach", "sandbox-kept"), ("forget", "sandbox-kept")]


# --- host files ---------------------------------------------------------------------------


def test_an_upload_copies_a_host_file_into_the_sandbox(use_pool, tmp_path):
    pool = use_pool(StubPool())
    session = created()

    uploaded = session.upload_file(tmp_path / "model.patch", "/tmp/model.patch")

    assert (uploaded, session.transfer_error) == (True, "")
    assert pool.operations[-1] == (
        "upload_file",
        "sandbox-1",
        str(tmp_path / "model.patch"),
        "/tmp/model.patch",
    )


def test_a_download_copies_a_guest_file_to_the_host_within_its_timeout(use_pool, tmp_path):
    pool = use_pool(StubPool())
    session = created()

    downloaded = session.download_file("/logs/verifier.tar.gz", tmp_path / "verifier.tar.gz", timeout=17)

    assert (downloaded, session.transfer_error) == (True, "")
    assert (tmp_path / "verifier.tar.gz").read_bytes() == b"the bytes of /logs/verifier.tar.gz"
    assert pool.operations[-1] == (
        "download_file",
        "sandbox-1",
        "/logs/verifier.tar.gz",
        str(tmp_path / "verifier.tar.gz"),
        17,
    )


def upload(session: SandboxSession, path: Path) -> bool:
    return session.upload_file(path, "/tmp/x")


def download(session: SandboxSession, path: Path) -> bool:
    return session.download_file("/tmp/x", path)


@pytest.mark.parametrize(
    "transfer, refused", [(upload, "upload_file"), (download, "download_file")], ids=["upload", "download"]
)
def test_a_transfer_that_fails_returns_false_with_the_reason(use_pool, tmp_path, transfer, refused):
    """A quiet caller puts ``transfer_error`` in its own error."""
    use_pool(StubPool(refuse={refused}))
    session = created()

    moved = transfer(session, tmp_path / "x")

    assert (moved, session.transfer_error) == (False, "RuntimeError: %s refused" % refused)


def without_a_sandbox() -> SandboxSession:
    return SandboxSession(quiet=True)


@pytest.mark.parametrize(
    "transfer, reason",
    [
        (upload, "no sandbox, or its backend cannot upload"),
        (download, "no sandbox, or its backend cannot download"),
    ],
    ids=["upload", "download"],
)
@pytest.mark.parametrize("session_of", [without_a_sandbox, created], ids=["no-sandbox", "backend-cannot"])
def test_a_transfer_that_the_session_cannot_make_returns_false_with_the_reason(
    use_pool, tmp_path, transfer, reason, session_of
):
    pool = use_pool(StubPool(can_transfer=False))
    session = session_of()

    moved = transfer(session, tmp_path / "x")

    assert (moved, session.transfer_error, pool.operations[1:]) == (False, reason, [])


# --- the network ------------------------------------------------------------------------


def test_the_network_of_a_running_sandbox_is_switched_through_its_pool(use_pool):
    """A Harbor task's verifier phase can have another network policy than its agent phase."""
    pool = use_pool(NetworkedPool())
    session = created()

    session.set_network(False)
    session.set_network(True)

    assert pool.operations[1:] == [("set_network", "sandbox-1", False), ("set_network", "sandbox-1", True)]


def test_a_backend_that_cannot_switch_a_running_sandboxs_network_says_so(use_pool):
    """A silent no-op would run a no-network verifier online."""
    use_pool(StubPool())
    session = created(backend={"backend": "microvm"})

    with pytest.raises(
        NotImplementedError, match="^backend 'microvm' cannot change a running sandbox's network$"
    ):
        session.set_network(False)


def test_the_network_cannot_be_switched_without_a_sandbox():
    with pytest.raises(RuntimeError, match="^No active sandbox$"):
        SandboxSession(quiet=True).set_network(True)


# --- the image's config -------------------------------------------------------------------

#: A regctl that logs its arguments and prints the config of an image whose USER is not root.
STUB_REGCTL = """#!/bin/sh
echo "$@" >> "$REGCTL_LOG"
echo '{"User":"haproxy","Env":["PATH=/usr/bin","HAPROXY_VERSION=3.0.30"],"WorkingDir":"/var/lib/haproxy"}'
"""


def test_a_microvm_session_reads_the_images_config_from_its_registry(tmp_path, monkeypatch):
    """A microVM keeps no config of its image, so the session asks ``regctl`` for the reference."""
    regctl = tmp_path / "regctl"
    regctl.write_text(STUB_REGCTL)
    regctl.chmod(0o755)
    monkeypatch.setenv("SPROUT_REGCTL", str(regctl))
    monkeypatch.setenv("REGCTL_LOG", str(tmp_path / "regctl.log"))
    session = SandboxSession(quiet=True, backend={"backend": "microvm"})

    config = session.image_config("docker.io/library/haproxy:3.0")

    assert config == {
        "User": "haproxy",
        "Env": ["PATH=/usr/bin", "HAPROXY_VERSION=3.0.30"],
        "WorkingDir": "/var/lib/haproxy",
    }
    assert (tmp_path / "regctl.log").read_text() == (
        "image config docker.io/library/haproxy:3.0 --format {{json .Config}}\n"
    )


def test_a_podman_session_without_a_sandbox_cannot_read_an_image_config():
    """A podman sandbox keeps the config of its image. Without one, there is nothing to read."""
    with pytest.raises(RuntimeError, match="^No active sandbox$"):
        SandboxSession(quiet=True).image_config(IMAGE)
