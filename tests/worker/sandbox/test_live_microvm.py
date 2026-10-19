"""A ``SandboxSession`` on a real AgentENV server: branches from one snapshot are isolated.

The fakes cannot answer whether branching isolates. A fork demo on a backend that cannot snapshot
showed the failure: a sibling saw the edit of another branch. Large tests: they need a reachable
``$AENV_SERVER_URL`` (with ``$AENV_API_KEY``) and the runtime binary, and skip without them. Each test
destroys each sandbox that it made, because a shared AgentENV hosts the tasks of other people. The
first create builds a template of the image, which takes minutes::

    AENV_SERVER_URL=http://127.0.0.1:18000 AENV_API_KEY=... \
        pytest tests/worker/sandbox/test_live_microvm.py -m slow
"""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from urllib.parse import urlsplit

import pytest

from sprout.worker.sandbox.commands import checked
from sprout.worker.sandbox.session import SandboxSession
from tests.support.paths import RUNTIME_BIN

pytestmark = pytest.mark.slow

IMAGE = "docker.io/library/python:3.11-slim"


def agentenv_problem(url: str | None) -> str | None:
    """Why the AgentENV server at ``url`` cannot serve a large test, or None."""
    if not url:
        return "needs an AgentENV server in $AENV_SERVER_URL, and its key in $AENV_API_KEY"
    target = urlsplit(url)
    try:
        with socket.create_connection((target.hostname, target.port or 80), timeout=3):
            return None
    except OSError as error:
        return "the AgentENV server at $AENV_SERVER_URL does not answer: %s" % error


class Sessions:
    """The microVM sessions of one test. ``new`` makes one from an image or a snapshot id."""

    def __init__(self, url: str) -> None:
        self.backend = {
            "backend": "microvm",
            "microvm": {
                "server_url": url,
                "runtime_bin": str(RUNTIME_BIN),
                "from_image": True,
                "sandbox_ttl": 1800,
            },
        }
        self.made: list[SandboxSession] = []

    def new(self, image_or_snapshot: str = IMAGE) -> SandboxSession:
        session = SandboxSession(quiet=True, backend=self.backend)
        self.made.append(session)
        assert session.create(image_or_snapshot), session.create_error
        return session


@pytest.fixture
def sessions() -> Iterator[Sessions]:
    """Destroy each session of the test, also when the test fails."""
    if not RUNTIME_BIN.is_file():
        pytest.skip(
            "needs the runtime binary: cd runtime && go generate ./... && go build -o sprout-runtime ."
        )
    url = os.environ.get("AENV_SERVER_URL")
    problem = agentenv_problem(url)
    if problem:
        pytest.skip(problem)
    made = Sessions(str(url))
    yield made
    for session in made.made:
        session.destroy()


def listing(session: SandboxSession) -> str:
    return checked(session, "ls /work")


def test_a_disk_snapshot_reports_its_layer_chain_and_no_memory(sessions):
    """The checkpointer detects a compaction of the chain by its layer count."""
    session = sessions.new()
    checked(session, "mkdir -p /work && echo x > /work/f")

    snapshot = session.snapshot()

    assert (snapshot.disk_only, snapshot.rootfs_layers > 0, snapshot.memory_layers) == (True, True, 0)


def test_branches_of_one_snapshot_have_its_files_and_do_not_see_each_others_writes(sessions):
    """The snapshot outlives the sandbox that it was taken of. A later write of that sandbox is not in it."""
    parent = sessions.new()
    checked(parent, "mkdir -p /work && echo base > /work/base")
    snapshot = parent.snapshot()
    checked(parent, "echo late > /work/late")
    parent.destroy()
    alpha, beta = sessions.new(snapshot.id), sessions.new(snapshot.id)

    checked(alpha, "echo alpha > /work/alpha")
    checked(beta, "echo beta > /work/beta")

    assert (listing(alpha), listing(beta)) == ("alpha\nbase\n", "base\nbeta\n")


def test_a_branch_of_a_branch_keeps_both_generations(sessions):
    """A tree search branches a branch. Each generation's layers stack on the one before."""
    root = sessions.new()
    checked(root, "mkdir -p /work && echo 0 > /work/gen0")
    child = sessions.new(root.snapshot().id)
    checked(child, "echo 1 > /work/gen1")

    grandchild = sessions.new(child.snapshot().id)

    assert listing(grandchild) == "gen0\ngen1\n"
