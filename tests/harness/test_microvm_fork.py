"""The rollback stack's acceptance test, against a real AgentENV.

Everything else about forking is covered with fakes. This is the one test that
answers the question the fakes cannot: **does branching actually isolate?** The
first fork demo, run on Docker (which cannot snapshot), showed exactly the failure
this prevents -- branch 1 edited a file and branch 2, sharing the filesystem,
reported "the file no longer contains the bug".

Skipped unless AENV_SERVER_URL is reachable and the sprout-runtime binary is built,
so a normal test run is unaffected. Each test creates and destroys its own
sandboxes and never touches anything it did not make: a shared AgentENV is likely
to be hosting somebody's long-running task.

    AENV_SERVER_URL=http://127.0.0.1:18000 AENV_API_KEY=... \
        pytest tests/harness/test_microvm_fork.py -v
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from urllib.parse import urlparse

import pytest

REPO = Path(__file__).resolve().parents[2]
RUNTIME_BIN = REPO / "runtime" / "sprout-runtime"
IMAGE = "python:3.11-slim"
#: The template build on a cold image is minutes, not seconds.
CREATE_TIMEOUT_HINT = "first run builds a template from the image; expect minutes"


def _server() -> "str | None":
    url = os.environ.get("AENV_SERVER_URL")
    if not url:
        return None
    parsed = urlparse(url)
    try:
        with socket.create_connection((parsed.hostname, parsed.port or 80), timeout=3):
            return url
    except OSError:
        return None


pytestmark = [
    pytest.mark.skipif(not RUNTIME_BIN.exists(),
                       reason="needs runtime/sprout-runtime (cd runtime && go build -o sprout-runtime .)"),
    pytest.mark.skipif(_server() is None,
                       reason="needs a reachable AENV_SERVER_URL (+ AENV_API_KEY)"),
    pytest.mark.slow,
]


def _backend() -> dict:
    return {"backend": "microvm", "microvm": {
        "server_url": os.environ["AENV_SERVER_URL"],
        "api_key": os.environ.get("AENV_API_KEY", ""),
        "runtime_bin": str(RUNTIME_BIN),
        "from_image": True,
        "sandbox_ttl": 1800,
    }}


class Owned:
    """Sessions this test made, destroyed on the way out.

    Explicit rather than a fixture per session: a leaked microVM costs real
    memory on a shared host, and `harness reap` is a backstop, not a plan.
    """

    def __init__(self):
        self.sessions = []

    def create(self, image_or_snapshot: str):
        from sprout.harness.sandbox.session import SandboxSession

        session = SandboxSession(quiet=True, backend=_backend())
        assert session.create(image_or_snapshot), \
            "create(%s) failed (%s)" % (image_or_snapshot, CREATE_TIMEOUT_HINT)
        self.sessions.append(session)
        return session

    def close(self):
        for session in self.sessions:
            try:
                session.destroy()
            except Exception:  # noqa: BLE001 - keep destroying the rest
                pass


@pytest.fixture
def owned():
    holder = Owned()
    try:
        yield holder
    finally:
        holder.close()


def _ls(session, path: str = "/work") -> list:
    return sorted((session.execute("shell", {"command": "ls %s" % path}).output or "").split())


# --- the mechanism ---------------------------------------------------------
def test_a_microvm_session_supports_snapshots(owned):
    """The capability the rollback pair needs, asserted against the real backend."""
    session = owned.create(IMAGE)
    assert session.supports_snapshot() is True


def test_snapshot_reports_its_layer_chain(owned):
    """The layer counts are what the Checkpointer watches to detect server-side
    compaction, so an empty report would silently disable re-boarding."""
    session = owned.create(IMAGE)
    session.execute("shell", {"command": "mkdir -p /work && echo x > /work/f"})
    snapshot = session.snapshot(disk_only=True)

    assert snapshot.id
    assert snapshot.disk_only is True
    assert snapshot.rootfs_layers > 0
    assert snapshot.memory_layers == 0      # disk_only skips the memory image


# --- the acceptance test ---------------------------------------------------
def test_branches_from_one_snapshot_are_isolated(owned):
    """Three forks of one environment: each sees the inherited state and its own
    writes, and none sees a sibling's.

    This is the capability the whole stack exists for. On a backend that cannot
    snapshot, the same demo produced cross-contamination.
    """
    parent = owned.create(IMAGE)
    parent.execute("shell", {"command": "mkdir -p /work && echo BASE > /work/base.txt"})
    assert _ls(parent) == ["base.txt"]

    snapshot = parent.snapshot(disk_only=True)
    parent.destroy()                        # the snapshot outlives its maker
    owned.sessions.remove(parent)

    names = ("ALPHA", "BETA", "GAMMA")
    branches = {}
    for name in names:
        branch = owned.create(snapshot.id)
        branch.execute("shell", {"command": "echo %s > /work/branch_%s.txt" % (name, name)})
        branches[name] = branch

    for name, branch in branches.items():
        listing = _ls(branch)
        assert "base.txt" in listing, "%s lost the inherited state" % name
        assert "branch_%s.txt" % name in listing, "%s lost its own write" % name
        siblings = ["branch_%s.txt" % other for other in names if other != name]
        leaked = [f for f in siblings if f in listing]
        assert leaked == [], "%s can see sibling writes: %s" % (name, leaked)


def test_a_branch_inherits_state_written_before_the_snapshot(owned):
    """Restoring must carry the work forward -- a branch that starts clean would
    make forking pointless."""
    parent = owned.create(IMAGE)
    parent.execute("shell", {"command":
                             "mkdir -p /work && printf 'line1\\nline2\\n' > /work/history.txt"})
    snapshot = parent.snapshot(disk_only=True)

    branch = owned.create(snapshot.id)
    content = branch.execute("shell", {"command": "cat /work/history.txt"}).output or ""
    assert "line1" in content and "line2" in content


def test_writes_after_a_snapshot_are_not_in_it(owned):
    """The snapshot is a point in time, not a live mirror of the sandbox."""
    parent = owned.create(IMAGE)
    parent.execute("shell", {"command": "mkdir -p /work && echo before > /work/before.txt"})
    snapshot = parent.snapshot(disk_only=True)
    parent.execute("shell", {"command": "echo after > /work/after.txt"})

    branch = owned.create(snapshot.id)
    listing = _ls(branch)
    assert "before.txt" in listing
    assert "after.txt" not in listing, "the snapshot captured a later write"


def test_a_branch_of_a_branch_keeps_both_generations(owned):
    """Forking is not limited to one level: a tree search needs to branch a
    branch, and each generation's layers stack."""
    root = owned.create(IMAGE)
    root.execute("shell", {"command": "mkdir -p /work && echo gen0 > /work/gen0.txt"})
    first = root.snapshot(disk_only=True)

    child = owned.create(first.id)
    child.execute("shell", {"command": "echo gen1 > /work/gen1.txt"})
    second = child.snapshot(disk_only=True)

    grandchild = owned.create(second.id)
    listing = _ls(grandchild)
    assert "gen0.txt" in listing and "gen1.txt" in listing, listing
