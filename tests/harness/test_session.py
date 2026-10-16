"""SandboxSession: lifecycle, snapshots, re-boarding and the tool seam.

The orchestrator owns one session per run. These pin what its callers rely on:
create reports rather than raises, a re-board is invisible to callers and told
to listeners first, a snapshot failure never fails the episode, and every tool
call goes out under the owner's identity.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, List

import pytest

from sprout.harness_core.result import ToolResult
from sprout.harness_sandbox.session import OWNER_AGENT_ID, SandboxSession, shell, shell_outcome


# --- fakes -----------------------------------------------------------------
class FakeResult:
    def __init__(self, output: str = "", is_error: bool = False):
        self.output = output
        self.is_error = is_error
        self.error = None


class FakeBackend:
    """The channel the session executes through: records (tool, args, agent_id)."""

    def __init__(self, calls: List[tuple]):
        self.calls = calls

    async def call(self, tool: str, args: dict, agent_id: str = ""):
        self.calls.append((tool, args, agent_id))
        return FakeResult(output="ran %s as %s" % (tool, agent_id))


class FakeSandbox:
    """A sandbox-shaped object. Deliberately minimal -- see the id test."""

    def __init__(self, sandbox_id: str = "sb-1", reachable: bool = True):
        self.sandbox_id = sandbox_id
        self.agent_id = "agent"
        self.reachable = reachable
        self.calls: List[tuple] = []
        self.backend = FakeBackend(self.calls)

    async def call(self, tool: str, **kwargs):
        self.calls.append((tool, kwargs))
        if not self.reachable:
            raise RuntimeError("502 Bad Gateway")
        return FakeResult(output="ok")


class FakePool:
    def __init__(self, sandbox=None, replacement=None):
        self.sandbox = sandbox or FakeSandbox()
        self.replacement = replacement
        self.destroyed: List[Any] = []
        self.snapshots: List[dict] = []

    def supports_snapshot(self):
        return True

    def supports_upload(self):
        return True

    def supports_cold_start(self):
        return True

    async def spawn(self, image=None, agent_id=""):
        if self.replacement is not None:
            return self.replacement
        return self.sandbox

    async def spawn_from_image(self, image, resources=None):
        return self.sandbox

    async def destroy(self, sandbox):
        self.destroyed.append(sandbox)

    async def snapshot(self, sandbox, name=None, disk_only=True):
        self.snapshots.append({"name": name, "disk_only": disk_only})
        return type("Snap", (), {"id": "snap-1"})()


def _session(pool: FakePool, cls=SandboxSession, **kwargs) -> SandboxSession:
    """A session with its pool already in place, skipping create()."""
    session = cls(quiet=True, **kwargs)
    session._pool = pool
    session._sandbox = pool.sandbox
    return session


def test_interrupted_drive_cancels_in_original_task():
    import anyio

    session = SandboxSession(quiet=True)
    loop = session._get_loop()
    tasks = []

    async def operation():
        with anyio.fail_after(60):
            tasks.append(asyncio.current_task())
            try:
                await asyncio.sleep(60)
            finally:
                tasks.append(asyncio.current_task())

    def interrupt():
        raise KeyboardInterrupt("isolated cancellation injection")

    loop.call_later(0.01, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="isolated cancellation injection"):
            session._drive(operation())
        assert len(tasks) == 2 and tasks[0] is tasks[1]
        assert not asyncio.all_tasks(loop)
        assert session._drive(asyncio.sleep(0, result="healthy")) == "healthy"
    finally:
        loop.close()


def test_unsettled_drive_quarantines_session_until_cleanup_finishes():
    session = SandboxSession(quiet=True)
    session._cancel_grace_seconds = 0.01
    loop = session._get_loop()
    release = asyncio.Event()

    async def operation():
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    def interrupt():
        raise KeyboardInterrupt("isolated cancellation injection")

    loop.call_later(0.01, interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            session._drive(operation())
        with pytest.raises(RuntimeError, match="quarantined"):
            session._drive(asyncio.sleep(0))
        release.set()
        loop.run_until_complete(session._unsettled_operation)
        assert session._drive(asyncio.sleep(0, result="healthy")) == "healthy"
        assert not asyncio.all_tasks(loop)
    finally:
        release.set()
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


@pytest.mark.parametrize("fails", [False, True])
def test_session_download_routes_to_pool_and_reports_failure(tmp_path, fails):
    destination = tmp_path / "logs.tar.gz"

    class DownloadPool(FakePool):
        def supports_download(self):
            return True

        async def download_file(self, sandbox, source, target, timeout=None):
            assert sandbox is self.sandbox
            assert source == "/tmp/logs.tar.gz" and timeout == 17
            if fails:
                raise OSError("download failed")
            Path(target).write_bytes(b"log bytes\x00")

    session = _session(DownloadPool())
    try:
        assert session.download_file("/tmp/logs.tar.gz", destination, timeout=17) is not fails
        assert session.transfer_error == ("OSError: download failed" if fails else "")
        if not fails:
            assert destination.read_bytes() == b"log bytes\x00"
    finally:
        session._loop.close()


def test_a_failed_upload_says_why_for_a_quiet_caller(tmp_path):
    """The grader's session is quiet, so the reason travels with the answer, as for create."""
    class FullPool(FakePool):
        async def upload_file(self, sandbox, source, destination):
            raise OSError("no space left on device")

    source = tmp_path / "model.patch"
    source.write_bytes(b"diff")
    session = _session(FullPool())
    try:
        assert session.upload_file(source, "/tmp/sprout-grade-x/model.patch") is False
        assert session.transfer_error == "OSError: no space left on device"
    finally:
        session._loop.close()
    assert SandboxSession(quiet=True).upload_file(source, "/tmp/x") is False
    assert SandboxSession(quiet=True).download_file("/tmp/x", tmp_path / "x") is False


# --- creating ----------------------------------------------------------------
def test_a_failing_create_reports_false_rather_than_raising(pooled):
    """A run reports; it does not explode in its caller's face -- and the reason
    travels with the answer, because a quiet caller has nothing else to show."""
    class Broken(FakePool):
        async def spawn(self, image=None, agent_id=""):
            raise RuntimeError("backend said no")

        async def spawn_from_image(self, image, resources=None):
            raise RuntimeError("backend said no")

    pooled(Broken())
    session = SandboxSession(quiet=True)
    assert session.create("img") is False
    assert "backend said no" in session.create_error


PODMAN = {"backend": "podman", "podman": {"store": "/mnt/store", "runtime_bin": "rt"}}
SHAPE = {"cpu": 2, "memory_mb": 4096}


@pytest.mark.parametrize("backend, image, resources, expected", [
    # no backend named: the default (podman), as a plain create() with no shape takes it
    (None, "img", None, ("shaped", "img", None)),
    # podman: an image reference and a snapshot id (a branch) alike carry the shape
    (PODMAN, "docker.io/library/python:3.11-slim", SHAPE, ("shaped", "docker.io/library/python:3.11-slim", SHAPE)),
    (PODMAN, "sprout-000000000001-0003", SHAPE, ("shaped", "sprout-000000000001-0003", SHAPE)),
    # microvm without from_image: the snapshot path, and the shape stays behind
    ({"backend": "microvm"}, "snap-7", {"cpu": 2}, ("shapeless", "snap-7")),
], ids=["default", "podman-image", "podman-snapshot", "microvm"])
def test_create_takes_each_backends_spawn_path_with_the_shape_it_takes(pooled, backend, image, resources, expected):
    """A subvolume carries no shape, so the podman pool takes the resources on
    its one spawn path -- for an image reference and for a snapshot id (a
    branch) alike. Without this a task's `--cpus`/`--memory` reached only
    microVMs, and every podman container ran unlimited. The microVM path is
    unchanged: a template or snapshot has its shape baked in, and the pool
    refuses the argument, so nothing is passed. Whichever path, a successful
    create() adopts what the pool spawned: the session's id is `unknown` until
    then and the sandbox's own after."""
    class Recording(FakePool):
        def __init__(self):
            super().__init__()
            self.spawned: List[tuple] = []

        async def spawn(self, image=None, agent_id=""):
            self.spawned.append(("shapeless", image))
            return self.sandbox

        async def spawn_from_image(self, image, resources=None):
            self.spawned.append(("shaped", image, resources))
            return self.sandbox

    pool = pooled(Recording())
    session = SandboxSession(quiet=True, backend=backend)
    assert session.sandbox_id == "unknown"
    assert session.create(image, resources) is True
    assert pool.spawned == [expected]
    assert session._sandbox is pool.sandbox and session.sandbox_id == "sb-1"


# --- the bug this refactor introduced, and its fix -------------------------
def test_a_progress_line_cannot_break_the_operation_it_reports():
    """Reporting used to sit behind `if not self.quiet`, so a quiet caller never
    computed the sandbox id. Moving reporting into a helper made the argument eager
    -- and a successful re-board started raising AttributeError from a *logging*
    line on any sandbox object that could not name itself."""
    class Nameless:
        agent_id = "agent"

        async def call(self, tool, **kwargs):
            return FakeResult(output="ok")

    replacement = Nameless()
    pool = FakePool(replacement=replacement)
    for quiet in (True, False):
        session = SandboxSession(quiet=quiet)
        session._pool = pool
        session._sandbox = FakeSandbox()
        pool.destroyed.clear()
        assert session.swap_sandbox("snap-x") is True, "quiet=%s" % quiet
        assert session.sandbox_id == "unknown"


def test_swap_tells_listeners_before_the_old_sandbox_is_destroyed():
    """Whoever else holds this session's sandbox handle must follow a re-board.
    The in-process MCP server did not: the bridge swapped the session, the
    server kept serving the destroyed VM, and every tool call 404'd until the
    run ended (3 of the first 43 DeepSWE runs, the moment chain compaction
    first fired). Order matters too -- learning late is one lost call. And a
    listener with a bug of its own fails neither the re-board nor the listeners
    after it, so the one that raises is registered first."""
    replacement = FakeSandbox(sandbox_id="sb-2")
    pool = FakePool(replacement=replacement)
    session = _session(pool)
    seen = []

    def boom(_sb):
        raise RuntimeError("listener bug")
    session.on_swap.append(boom)
    session.on_swap.append(lambda sb: seen.append((sb, list(pool.destroyed))))
    assert session.swap_sandbox("snap-x") is True
    assert session._sandbox is replacement
    assert seen and seen[0][0] is replacement, "a failing listener stopped the ones after it"
    assert seen[0][1] == [], "listener ran after the previous sandbox was destroyed"


# --- the checkpoint contract -----------------------------------------------
def test_the_methods_checkpointing_calls_are_present():
    """The Checkpointer (sprout.harness_checkpoint.checkpointer) duck-types on these.
    A session missing one cannot record the environment half of a branch point."""
    for name in ("snapshot", "squash_snapshot", "swap_sandbox"):
        assert callable(getattr(SandboxSession, name, None)), name
    assert not hasattr(SandboxSession, "squash_lineage_at"), \
        "no backend declares a squash depth any more (podman snapshots are subvolumes; nothing stacks)"


def test_snapshot_defaults_to_disk_only():
    """A rollout replays by restoring the disk and re-feeding the transcript, so
    paying for a memory image every step buys nothing."""
    pool = FakePool()
    session = _session(pool)
    assert session.snapshot().id == "snap-1"
    assert pool.snapshots == [{"name": None, "disk_only": True}]


def test_snapshot_returns_none_rather_than_failing_the_episode():
    """A checkpoint is an optimisation for later analysis. An episode that dies
    because a snapshot did is strictly worse than one with a gap in its history."""
    class Failing(FakePool):
        async def snapshot(self, sandbox, name=None, disk_only=True):
            raise RuntimeError("backend said no")

    assert _session(Failing()).snapshot() is None


class NoSquash(FakePool):
    """A backend that cannot snapshot at all."""

    def supports_snapshot(self):
        return False


class BrokenSquash(FakePool):
    """AgentENV refusing the squash."""

    async def squash(self, snapshot, name=None):
        raise RuntimeError("podman import failed (exit 125)")


@pytest.mark.parametrize("pool", [NoSquash, FakePool, BrokenSquash],
                         ids=["cannot-snapshot", "no-squash-like-podman", "squash-raises"])
def test_squash_returns_the_input_when_the_backend_cannot(pool):
    """A deep chain still works; it just makes its children's checkpoints more
    expensive. Returning None here would look like a lost checkpoint: the
    checkpointer relies on getting the same snapshot back, and then falls back
    to the plain re-board. Three backends cannot: one that cannot snapshot at
    all, the podman pool, whose snapshots are subvolumes with nothing to flatten
    (it has no `squash`, and neither has FakePool), and AgentENV refusing."""
    snap = object()
    assert _session(pool()).squash_snapshot(snap) is snap


def test_a_reboard_whose_replacement_never_boots_keeps_the_sandbox():
    """A pool that removes a sandbox whose runtime never answers raises out of
    spawn, so the reachability probe below never sees it. The session keeps its
    sandbox and answers False; the checkpointer records `reboarded: false` and
    the run goes on."""
    original = FakeSandbox(sandbox_id="sb-1")

    class NoBoot(FakePool):
        async def spawn(self, image=None, agent_id=""):
            raise RuntimeError("runtime in c3 did not answer within 30 s")

    pool = NoBoot(sandbox=original)
    session = _session(pool)
    assert session.swap_sandbox("snap-x") is False
    assert session._sandbox is original and session.sandbox_id == "sb-1"
    assert pool.destroyed == [], "nothing to clean up: the pool already removed its container"


def test_an_unreachable_reboard_target_is_not_adopted():
    """A disk-only snapshot cold-boots, so its runtime is only back if the
    template declares a startup command. Adopting an unreachable replacement turns
    every later tool call into a transport error; keeping the old sandbox costs
    only a deeper layer chain."""
    original = FakeSandbox(sandbox_id="sb-1")
    replacement = FakeSandbox(sandbox_id="sb-2", reachable=False)
    pool = FakePool(sandbox=original, replacement=replacement)
    session = _session(pool)
    assert session.swap_sandbox("snap-x") is False
    assert session._sandbox is original
    assert pool.destroyed == [replacement], "the useless replacement is cleaned up"


# --- the tool seam ---------------------------------------------------------
@pytest.mark.parametrize("args", [{"command": "ls"}, {"command": "ls", "agent_id": "spoofed"}],
                         ids=["plain", "spoofed"])
def test_an_agent_id_in_the_arguments_is_dropped(args):
    """Every call goes out on the owner's channel, and a model that emits an
    agent_id must not be able to name itself: the channel's identity wins, and
    the argument never reaches the runtime. The backend's answer comes back as
    the result's output."""
    pool = FakePool()
    session = _session(pool)
    result = session.execute("shell", dict(args))
    tool, sent, agent_id = pool.sandbox.calls[-1]
    assert agent_id == OWNER_AGENT_ID
    assert "agent_id" not in sent
    assert "as %s" % OWNER_AGENT_ID in result.output


def test_a_call_with_no_sandbox_is_an_error_not_a_crash():
    session = SandboxSession(quiet=True)
    result = session.execute("shell", {"command": "ls"})
    assert not result.success
    assert "No active sandbox" in (result.error or "")


# --- command outcomes --------------------------------------------------------
class ReplyingBackend:
    """Answers every call with one runtime ``shell`` reply, parsed as the SDK parses it."""

    def __init__(self, outcome: dict):
        self.outcome = {"stdout": "", "stderr": "", **outcome}
        self.calls: List[tuple] = []

    async def call(self, tool: str, args: dict, agent_id: str = ""):
        from sprout_sandbox.result import ToolResult as SdkToolResult

        self.calls.append((tool, args))
        return SdkToolResult.from_response({"content": [{"type": "text", "text": json.dumps(self.outcome)}],
                                            "isError": self.outcome.get("exit_code") != 0})


class PayloadSession:
    """A session whose result carries the outcome only as JSON text, the other shape shell() reads."""

    def __init__(self, outcome: dict):
        self.outcome = {"stdout": "", "stderr": "", **outcome}

    def execute(self, tool, args, timeout=300.0):
        return ToolResult(success=False, output=json.dumps(self.outcome))


def _replying(outcome: dict, shape: str):
    if shape == "payload":
        return PayloadSession(outcome)
    sandbox = FakeSandbox()
    sandbox.backend = ReplyingBackend(outcome)
    return _session(FakePool(sandbox=sandbox))


@pytest.mark.parametrize("shape", ["outcome", "payload"])
def test_shell_outcome_answers_a_timeout_that_shell_still_raises_on(shape):
    """A grader whose tests hang scores them 0: for shell_outcome a timeout is
    an answer, the runtime's 137 included. shell() keeps refusing one, as every
    other caller expects."""
    session = _replying({"stdout": "collected 3 items\n", "exit_code": 137, "timed_out": True}, shape)
    assert shell_outcome(session, "pytest", 5) == ("collected 3 items\n", "", 137, True)
    with pytest.raises(RuntimeError, match="timed out after 5 s"):
        shell(session, "pytest", 5)
    if shape == "outcome":
        assert session._sandbox.backend.calls[-1] == ("shell", {"command": "pytest", "timeout": 5})
    unknown = _replying({"exit_code": None, "timed_out": True}, shape)
    assert shell_outcome(unknown, "pytest", 5) == ("", "", None, True), "a timeout without an exit code"


@pytest.mark.parametrize("shape", ["outcome", "payload"])
@pytest.mark.parametrize("outcome", [
    {"exit_code": None, "running": True},
    {"exit_code": 0, "stdout_truncated": True},
    {"exit_code": 1, "stderr_truncated": True},
    {"exit_code": None},
], ids=["running", "stdout truncated", "stderr truncated", "no exit code"])
def test_an_unknown_outcome_raises_from_both(shape, outcome):
    session = _replying(outcome, shape)
    with pytest.raises(RuntimeError, match="incomplete or truncated"):
        shell_outcome(session, "true")
    with pytest.raises(RuntimeError, match="incomplete or truncated"):
        shell(session, "true")


@pytest.mark.parametrize("shape", ["outcome", "payload"])
def test_a_settled_command_reads_alike_through_both(shape):
    session = _replying({"stderr": "boom\n", "exit_code": 2}, shape)
    assert shell_outcome(session, "false") == ("", "boom\n", 2, False)
    assert shell(session, "false") == ("", "boom\n", 2)
    plain = _session(FakePool())    # bare stdout: the runtime had nothing more to say
    assert shell_outcome(plain, "ls") == ("ran shell as %s" % OWNER_AGENT_ID, "", 0, False)
    assert shell(plain, "ls") == ("ran shell as %s" % OWNER_AGENT_ID, "", 0)


# --- helper ----------------------------------------------------------------
@pytest.fixture
def pooled(monkeypatch):
    """Make ``create()`` reach a FakePool, exercising the real code path.

    Patches the two module-level functions ``create`` uses rather than reaching
    into the session, so what runs is the actual spawn logic -- template
    resolution off, cold start on.
    """
    import sprout.harness_sandbox.session as module

    def install(pool: FakePool):
        monkeypatch.setattr(module, "build_pool", lambda backend: pool)
        monkeypatch.setattr(module, "builder_from_backend", lambda backend: None)
        return pool

    return install


def test_session_snapshot_workers_do_not_reenter_the_same_loop():
    """Two threads snapshotting one session serialize on its loop, not re-enter it."""
    import asyncio
    import threading
    import warnings
    from concurrent.futures import ThreadPoolExecutor

    from sprout.harness_sandbox.session import SandboxSession

    entered, release = threading.Event(), threading.Event()

    class Pool:
        def supports_snapshot(self):
            return True

        async def snapshot(self, sandbox, **kwargs):
            entered.set()
            assert await asyncio.to_thread(release.wait, 2)
            return "snapshot"

    session = SandboxSession()
    session._pool, session._sandbox = Pool(), object()
    with warnings.catch_warnings(record=True) as caught, ThreadPoolExecutor(2) as pool:
        warnings.simplefilter("always", RuntimeWarning)
        first = pool.submit(session.snapshot)
        assert entered.wait(2)
        second = pool.submit(session.snapshot)
        try:
            assert not second.done()
        finally:
            release.set()
        assert first.result(2) == second.result(2) == "snapshot"
    session._drive(session._loop.shutdown_default_executor())
    session._loop.close()
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


# --- the network contract --------------------------------------------------
def test_set_network_reaches_the_pool_and_a_backend_without_one_says_so():
    """Harbor's per-phase policy toggles a running sandbox's egress through the
    session; the podman pool has the operation, the microVM pool has not (AgentENV
    takes an HTTP PUT), and the difference must be an error, not a silent no-op."""
    class Networked(FakePool):
        def __init__(self):
            super().__init__()
            self.toggled = []

        async def set_network(self, sandbox, allow):
            self.toggled.append((sandbox, allow))

    pool = Networked()
    session = _session(pool)
    session.set_network(False)
    session.set_network(True)
    assert pool.toggled == [(pool.sandbox, False), (pool.sandbox, True)]
    with pytest.raises(NotImplementedError, match="microvm"):
        _session(FakePool(), backend={"backend": "microvm"}).set_network(False)
    with pytest.raises(RuntimeError, match="No active sandbox"):
        SandboxSession(quiet=True).set_network(True)
