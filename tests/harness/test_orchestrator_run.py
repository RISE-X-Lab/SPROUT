"""Orchestrator: the sequence of one run, and what it guarantees.

These pin the two things the sequence exists for -- **ordering** (each step must
happen before the one that depends on it, and "too late" is silently wrong rather
than an error) and **teardown** (everything acquired is released, including when
the slot raises).

A fake slot stands in for an agent. Transport regressions use loopback HTTP;
no model or VM is involved.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, List, Optional


import pytest

from sprout.harness_core import events as E
from sprout.harness_core.journal import read_journal
from sprout.harness_core.slot import AgentSlot, SlotResult
from sprout.harness_orchestrator import Orchestrator, RunSpec

#: The real sandbox wiring, for the tests that exercise it directly; every other
#: test gets a fake sandbox from the autouse fixture below.
REAL_WIRE_SANDBOX = Orchestrator._wire_sandbox


# --- fakes -----------------------------------------------------------------
class RecordingSlot(AgentSlot):
    """Records what the orchestrator handed it, and emits a couple of events."""

    name = "fake"
    calls: List[dict] = []
    raises: Optional[Exception] = None

    def run(self, task, journal, tools):
        RecordingSlot.calls.append({
            "prompt": task.prompt,
            "cwd": task.cwd,
            "model": task.model,
            "extra": dict(task.extra),
            "env": dict(task.env),
            "tools": tools,
        })
        if RecordingSlot.raises is not None:
            raise RecordingSlot.raises
        journal.emit(E.SESSION_REF, native_session_id="ses_fake")
        if hasattr(tools.session, "call"):
            # Two closed turns of one call each, the way an agent reports them.
            for turn in ("t1", "t2"):
                tools.call("shell", {"command": "true", "working_dir": "/", "timeout": 5},
                           call_id=turn + "-c", turn_id=turn)
                journal.emit("model.turn.completed", turn_id=turn, call_ids=[turn + "-c"],
                             step=journal.tool_calls()[-1]["step"])
                tools.close_turn(turn)
        journal.emit(E.TURN_COMPLETED, usage={"input_tokens": 7})
        return SlotResult(status="completed", final_text="done",
                          usage={"input_tokens": 7}, native_session_id="ses_fake")


@pytest.fixture(autouse=True)
def fake_slot(monkeypatch):
    RecordingSlot.calls = []
    RecordingSlot.raises = None
    monkeypatch.setattr("sprout.harness_orchestrator.run.MiniSweSlot", RecordingSlot)
    yield RecordingSlot


@dataclass
class FakeProvisioned:
    sandbox_id: str = "sb-1"
    destroyed: bool = False
    session: Any = None
    keep: bool = False

    def destroy(self):
        self.destroyed = True


class FakeSandboxes(list):
    session: Any = None


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    """A sandbox without a VM. Set ``fake_sandbox.session`` to give it a
    snapshot-capable session the agent's tools run on."""
    made = FakeSandboxes()

    def wire(self, run_spec, claim):
        provisioned = FakeProvisioned(session=made.session)
        made.append(provisioned)
        if claim is not None:
            claim.sandbox(provisioned.sandbox_id, keep=run_spec.keep_sandbox)
        return provisioned

    monkeypatch.setattr(Orchestrator, "_wire_sandbox", wire)
    return made


#: What the stand-in workspace probe resolves; the real one needs a shell.
WORKSPACE = {"cwd": "/testbed", "repository_dir": None}


@pytest.fixture(autouse=True)
def fake_workspace(monkeypatch):
    """The stand-in sandbox has no shell for mini's workspace probe. Records the
    session each probe ran on and answers with ``WORKSPACE``."""
    probed = []

    def resolve(session, extra, journal):
        probed.append(session)
        return {**extra, "mini_workspace": dict(WORKSPACE)}

    monkeypatch.setattr("sprout.harness_agents.mini_swe.sandbox.resolve_workspace", resolve)
    return probed


@dataclass
class FakeGateway:
    base_url: str = "http://127.0.0.1:9999"
    stopped: bool = False
    order: List[str] = field(default_factory=list)

    def start(self):
        return self

    def stop(self):
        self.stopped = True
        self.order.append("gateway.stop")

    def env_for(self, token):
        return {"ANTHROPIC_BASE_URL": self.base_url, "ANTHROPIC_AUTH_TOKEN": "tok"}


@dataclass
class FakeSession:
    """Snapshot-capable, with a shell that always succeeds."""

    snapshots: List[str] = field(default_factory=list)
    sandbox_id: str = "sb-1"

    def supports_snapshot(self):
        return True

    def call(self, name, args, timeout=300):
        from sprout.harness_core.result import CommandOutcome, ToolResult

        return ToolResult(True, "", outcome=CommandOutcome(exit_code=0))

    def snapshot(self, name=None, disk_only=True):
        sid = "snap-%d" % (len(self.snapshots) + 1)
        self.snapshots.append(sid)
        return type("S", (), {"id": sid, "rootfs_layers": 1, "memory_layers": 0,
                              "disk_only": disk_only})()

    def swap_sandbox(self, snapshot):
        return True

    def squash_snapshot(self, snapshot, name=None):
        return snapshot


def spec(tmp_path, **kwargs) -> RunSpec:
    defaults = dict(prompt="do it", slot="mini-swe-agent", cwd=str(tmp_path),
                    journal_path=tmp_path / "j.jsonl")
    defaults.update(kwargs)
    return RunSpec(**defaults)


def controlled() -> dict:
    """``extra`` of a run with rollout controls: what makes the orchestrator start a gateway."""
    return {"rollout_contract": {"model_endpoint": "http://127.0.0.1:1", "model": "m",
                                 "deadline_at": time.time() + 60, "max_turns": 5}}


# --- the happy path --------------------------------------------------------
def test_the_workspace_is_resolved_on_the_owned_sandbox_for_every_run(tmp_path, fake_sandbox, fake_workspace):
    """mini starts in a probed working directory: the probe runs on the sandbox
    this run owns, and what it resolves is what the slot receives."""
    fake_sandbox.session = FakeSession()
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path))
    assert fake_workspace == [fake_sandbox[0].session]
    assert RecordingSlot.calls[0]["extra"]["mini_workspace"] == WORKSPACE


def test_the_journal_names_the_sandbox_its_backend_and_a_podman_runs_store(tmp_path):
    """`harness sweep` reads the backend: a podman run's snapshot ids are
    subvolume names, which AgentENV would 404 and count as deleted. They are
    subvolumes of the store, so a podman run's record names that too; sweep then
    needs nothing but the journal."""
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, sandbox_image="img", backend={"backend": "microvm"}))
    [created] = [r for r in read_journal(tmp_path / "j.jsonl") if r["type"] == E.SANDBOX_CREATED]
    assert (created["sandbox_id"], created["backend"], created["image"]) == ("sb-1", "microvm", "img")
    assert "store" not in created, "an AgentENV run's snapshots are records, not files"

    podman = {"backend": "podman", "podman": {"store": str(tmp_path / "store"),
                                              "runtime_bin": "runtime/sprout-runtime"}}
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, journal_path=tmp_path / "p.jsonl", backend=podman))
    [created] = [r for r in read_journal(tmp_path / "p.jsonl") if r["type"] == E.SANDBOX_CREATED]
    assert (created["backend"], created["store"]) == ("podman", str(tmp_path / "store"))


@pytest.mark.parametrize("bad,message", [
    (dict(slot="codex"), "unknown slot"),
    (dict(), "sandbox_image is required"),
])
def test_a_bad_spec_is_refused_before_a_sandbox_or_the_agent_exists(tmp_path, monkeypatch, bad, message):
    """Through the real sandbox wiring, with no image to create from: the slot
    check comes first, so a typo'd slot never pays for a sandbox (one wired too
    early would surface as the image error instead), and a missing image is
    named before the backend fails opaquely on create(None)."""
    import sprout.harness_sandbox.session as session_module

    session = _FakeSession()
    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", REAL_WIRE_SANDBOX)
    outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, **bad))
    assert not outcome.ok and message in outcome.error
    assert session.created is None and RecordingSlot.calls == []


# --- ordering --------------------------------------------------------------
@pytest.mark.parametrize("memory,memory_id", [(False, None), (True, "snap-3")])
def test_each_closed_turn_is_paired_and_the_run_ends_with_one_disk_snapshot(tmp_path, fake_sandbox,
                                                                              memory, memory_id):
    """...and a full (memory) snapshot follows the disk one only when the run asks."""
    fake_sandbox.session = FakeSession()
    outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, final_memory_snapshot=memory))

    records = list(read_journal(tmp_path / "j.jsonl"))
    marks = [r for r in records if r["type"] == E.CHECKPOINT_CAPTURED]
    assert [(m["turn_id"], m["disk_only"], m.get("final", False)) for m in marks] == \
        [("t1", True, False), ("t2", True, True)]
    assert all(m["session_ckpt"] == "ses_fake" for m in marks)
    assert outcome.checkpoints == 1 and outcome.final_snapshot_id == marks[-1]["snapshot_id"] == "snap-2"
    assert outcome.final_memory_snapshot_id == memory_id
    full = [r for r in records if r["type"] == "snapshot.memory"]
    assert [(r["snapshot_id"], r["after"]) for r in full] == ([(memory_id, "snap-2")] if memory else [])


# --- teardown --------------------------------------------------------------
@pytest.mark.parametrize("keep", [False, True])
def test_the_sandbox_is_destroyed_after_the_run_unless_kept(tmp_path, fake_sandbox, keep):
    """Grading and post-hoc extraction need the sandbox alive after the agent
    stops; a run that is not kept leaves nothing behind."""
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, sandbox_image="img", keep_sandbox=keep))
    assert fake_sandbox[0].destroyed is (not keep)


def test_everything_is_released_when_the_slot_raises(tmp_path, monkeypatch):
    """A crashing agent must not strand a sandbox or a gateway."""
    provisioned = FakeProvisioned()
    gateway = FakeGateway()
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda self, run_spec, claim: provisioned)
    monkeypatch.setattr("sprout.harness_gateway.GatewayServer", lambda *a, **k: gateway)
    RecordingSlot.raises = RuntimeError("agent exploded")

    outcome = Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, sandbox_image="img", extra=controlled()))

    assert not outcome.ok and "agent exploded" in outcome.error
    assert provisioned.destroyed and gateway.stopped
    # and the failure is in the journal, not only in the return value
    assert any(r.get("status") == "error" for r in read_journal(tmp_path / "j.jsonl"))


def test_a_failing_teardown_step_does_not_block_the_others(tmp_path, monkeypatch):
    class Stubborn(FakeProvisioned):
        def destroy(self):
            raise RuntimeError("backend unreachable")

    gateway = FakeGateway()
    stubborn = Stubborn()
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda self, run_spec, claim: stubborn)
    monkeypatch.setattr("sprout.harness_gateway.GatewayServer", lambda *a, **k: gateway)

    outcome = Orchestrator(out_dir=tmp_path).run(
        spec(tmp_path, sandbox_image="img", extra=controlled()))
    assert outcome.ok                 # the run itself succeeded
    assert gateway.stopped            # stopped before the sandbox failed to die


# --- the resource ledger ---------------------------------------------------
class ReboardingSession(FakeSession):
    """Every capture reports the same layer count, so the run's second
    checkpoint finds the chain compacted and re-boards. The swap is
    SandboxSession's: the replacement is adopted, the listeners are told, then
    the old sandbox is destroyed."""

    def __init__(self):
        super().__init__(sandbox_id="vm-1")
        self.on_swap: List = []
        self.destroyed: List[str] = []

    def create(self, image, resources=None):
        return True

    def swap_sandbox(self, snapshot):
        previous, self.sandbox_id = self.sandbox_id, "vm-2"
        for listener in list(self.on_swap):
            listener(self.sandbox_id)
        self.destroyed.append(previous)
        return True

    def destroy(self):
        self.destroyed.append(self.sandbox_id)


class ThreeTurns(AgentSlot):
    """Three closed turns: the first two are captured mid-run, the last by the final snapshot."""

    name = "fake"

    def run(self, task, journal, tools):
        journal.emit(E.SESSION_REF, native_session_id="ses_fake")
        for turn in ("t1", "t2", "t3"):
            tools.call("shell", {"command": "true", "working_dir": "/", "timeout": 5},
                       call_id=turn + "-c", turn_id=turn)
            journal.emit("model.turn.completed", turn_id=turn, call_ids=[turn + "-c"],
                         step=journal.tool_calls()[-1]["step"])
            tools.close_turn(turn)
        return SlotResult(status="completed", final_text="done", native_session_id="ses_fake")


def test_a_reboard_moves_the_claim_and_the_record_to_the_replacement(tmp_path, monkeypatch):
    """A re-board swaps the session onto a new sandbox and destroys the old one.
    The replacement has to be claimed before that -- a process killed afterwards
    would otherwise leave a sandbox `harness reap` cannot attribute -- and the
    release at teardown and the outcome must name it, not the destroyed one."""
    import sprout.harness_sandbox.session as session_module
    from sprout.harness_orchestrator import ResourceLedger

    session = ReboardingSession()
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", REAL_WIRE_SANDBOX)
    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    monkeypatch.setattr("sprout.harness_orchestrator.run.MiniSweSlot", ThreeTurns)
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    outcome = Orchestrator(out_dir=tmp_path, ledger=ledger).run(
        spec(tmp_path, run_id="r1", sandbox_image="img", backend={"backend": "microvm"}))

    assert outcome.ok, outcome.error
    assert [r["reboarded"] for r in read_journal(tmp_path / "j.jsonl")
            if r["type"] == E.CHECKPOINT_CAPTURED and not r.get("final")] == [False, True]
    rows = [(e["event"], e["id"]) for e in ledger.entries() if e.get("kind") == "sandbox"]
    assert rows == [("claim", "vm-1"), ("claim", "vm-2"), ("release", "vm-2")]
    replacement = [e for e in ledger.entries() if e.get("event") == "claim"][-1]
    assert (replacement["backend"], replacement["keep"]) == ("microvm", False)
    assert outcome.sandbox_id == "vm-2" and session.destroyed == ["vm-1", "vm-2"]
    assert session.on_swap == [], "the listener leaves with the run: a lent session outlives it"


# --- owning the sandbox ----------------------------------------------------
#
# The orchestrator used to require its caller to build a sandbox and a session
# and pass them in, which is the opposite of owning a run: a caller that merely
# named an image got no environment half of a rollback pair, because nothing here
# held a handle to snapshot through. These pin the ownership and
# the teardown order.

class _FakeSession:
    """A SandboxSession-shaped object: create, snapshot, destroy."""

    def __init__(self, sandbox_id="sb-owned", create_ok=True):
        self._id = sandbox_id
        self.create_ok = create_ok
        self.create_error = "backend said no" if not create_ok else ""
        self.created = None
        self.destroyed = False
        self.order = []

    @property
    def sandbox_id(self):
        # Mirrors the real one: a destroyed session cannot name itself.
        return "unknown" if self.destroyed else self._id

    def create(self, image, resources=None):
        self.created = image
        self.order.append("create")
        return self.create_ok

    def supports_snapshot(self):
        return False

    def destroy(self):
        self.order.append("destroy")
        self.destroyed = True


def _own(monkeypatch, session, **spec_kwargs):
    """Drive the real sandbox wiring with a fake session, returning (orchestrator, owned)."""
    import sprout.harness_sandbox.session as session_module

    from sprout.harness_orchestrator.run import Orchestrator, RunSpec

    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    spec = RunSpec(prompt="x", sandbox_image="img", **spec_kwargs)
    orch = Orchestrator()
    return orch, REAL_WIRE_SANDBOX(orch, spec, None), spec


def test_one_run_start_to_finish_on_the_sandbox_it_owns(tmp_path, monkeypatch):
    """The plain run -- no rollout controls, a backend that cannot snapshot --
    through the real sandbox wiring. The sandbox comes from the named image; the
    slot gets its tools on that very session, and no gateway; the outcome carries
    the slot's result and still names the sandbox after teardown; and the ledger
    reads claim, release under the same id, done -- what `harness reap` has to go
    on, since a killed *process* releases nothing."""
    import sprout.harness_sandbox.session as session_module

    from sprout.harness_orchestrator import ResourceLedger
    from sprout.harness.tools.access import ToolAccess
    from sprout.harness.tools.schema import runtime_tools

    session = _FakeSession()
    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", REAL_WIRE_SANDBOX)
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    podman = {"backend": "podman", "podman": {"store": str(tmp_path / "store"),
                                              "runtime_bin": "runtime/sprout-runtime"}}
    outcome = Orchestrator(out_dir=tmp_path, ledger=ledger).run(
        spec(tmp_path, run_id="r1", sandbox_image="img", backend=podman))

    assert session.created == "img" and session.destroyed
    [call] = RecordingSlot.calls
    assert isinstance(call["tools"], ToolAccess) and call["tools"].session is session
    assert [tool["name"] for tool in call["tools"].tools] == [t["name"] for t in runtime_tools()], \
        "the agent is handed the runtime's own declaration"
    assert "OPENAI_BASE_URL" not in call["env"], "the gateway starts only for runs with rollout controls"
    assert outcome.ok and outcome.final_text == "done"
    assert outcome.usage == {"input_tokens": 7} and outcome.native_session_id == "ses_fake"
    assert outcome.sandbox_id == "sb-owned"   # recorded at creation; the destroyed session says "unknown"
    assert outcome.checkpoints == 0 and outcome.final_snapshot_id is None
    types = [r["type"] for r in read_journal(tmp_path / "j.jsonl")]
    assert E.SESSION_REF in types and E.TURN_COMPLETED in types
    entries = list(ledger.entries())
    assert [(e["event"], e.get("id")) for e in entries] == \
        [("claim", "sb-owned"), ("release", "sb-owned"), ("run_done", None)]
    assert (entries[0]["backend"], entries[0]["store"]) == ("podman", str(tmp_path / "store"))


def test_the_ledger_claim_names_the_backend(monkeypatch, tmp_path):
    """`harness reap` frees AgentENV's sandboxes; a podman container's claim
    has to say so, or reap reads it as "already gone" forever."""
    import sprout.harness_sandbox.session as session_module

    from sprout.harness_orchestrator import ResourceLedger
    from sprout.harness_orchestrator.run import Orchestrator, RunSpec

    session = _FakeSession()
    monkeypatch.setattr(session_module, "SandboxSession", lambda **kw: session)
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    podman = {"backend": "podman", "podman": {"runtime_bin": "runtime/sprout-runtime"}}
    with ledger.run("r1") as claim:
        REAL_WIRE_SANDBOX(Orchestrator(), RunSpec(prompt="x", sandbox_image="img", backend=podman), claim)
    [entry] = [e for e in ledger.entries() if e.get("event") == "claim"]
    assert (entry["id"], entry["backend"]) == ("sb-owned", "podman")


def test_a_failed_create_does_not_leave_a_half_wired_run(monkeypatch):
    import pytest

    session = _FakeSession(create_ok=False)
    with pytest.raises(RuntimeError, match="could not create a sandbox") as exc:
        _own(monkeypatch, session)
    # The reason travels with it: the session is quiet, so its own warning went
    # nowhere, and "could not create a sandbox from <image>" alone points at the
    # image when the cause is usually a missing setting.
    assert "backend said no" in str(exc.value)


def test_the_outcome_can_still_name_its_sandbox_after_teardown():
    """Teardown runs before the outcome is assembled, and a destroyed session
    answers "unknown" -- so reading the id lazily left every completed run unable
    to say which sandbox produced it."""
    from sprout.harness_orchestrator.run import OwnedSandbox

    session = _FakeSession()
    owned = OwnedSandbox(session=session, sandbox_id=session.sandbox_id)
    owned.release()
    assert session.destroyed
    assert owned.sandbox_id == "sb-owned"


# --- lineage -----------------------------------------------------------------
@pytest.mark.parametrize("origin", [
    {"parent_run_id": "p1", "branch_step": 2, "snapshot_id": "snap-x", "direction": "try Y"},
    None,
])
def test_a_forked_runs_journal_opens_with_its_origin(tmp_path, origin):
    """First event, before anything else: whatever else the journal ends up
    holding -- including nothing, for a run killed immediately -- it says where
    the run came from. A run with no origin says nothing at all: `inspect` reads
    the event as "this is a branch" and goes looking for a parent journal."""
    Orchestrator(out_dir=tmp_path).run(spec(tmp_path, origin=origin))
    records = list(read_journal(tmp_path / "j.jsonl"))
    assert [i for i, r in enumerate(records) if r["type"] == "fork.origin"] == ([0] if origin else [])
    if origin:
        assert (records[0]["parent_run_id"], records[0]["branch_step"]) == ("p1", 2)


# --- journals must survive a reboot ------------------------------------------
def test_volatile_paths_are_named_and_persistent_ones_are_not():
    """A journal is the ONLY record a run leaves -- snapshot ids, every step, the
    grading evidence. A 32-instance batch's journals lived in /tmp when the host
    rebooted mid-regrade; hours of agent time now exist only as prose. The guard
    is a resolved-prefix check at the ENTRY POINTS only: the library layer stays
    unguarded because tests legitimately journal into pytest's /tmp fixtures."""
    from sprout.harness_core.journal import volatile_reason

    for doomed in ("/tmp/v32", "/tmp", "/var/tmp/x/deep/run.jsonl",
                   "/dev/shm/j.jsonl", "/tmp/../tmp/sneaky"):
        assert volatile_reason(doomed), doomed
    for safe in ("runs/fork-eval", "/home/user/runs", "/tmpdir/runs",
                 "/data/tmp-results"):
        assert volatile_reason(safe) is None, safe


def test_the_run_entrypoint_refuses_a_volatile_journal():
    import subprocess, sys

    proc = subprocess.run(
        [sys.executable, "-m", "sprout.harness", "run", "--slot", "mini-swe-agent",
         "--sandbox-image", "img", "--runtime-bin", "runtime/sprout-runtime",
         "--journal", "/tmp/doomed.jsonl", "prompt"],
        capture_output=True, text=True)
    assert proc.returncode != 0
    assert "refusing" in proc.stderr and "--volatile-ok" in proc.stderr


def test_the_pipeline_is_limits_then_the_tracker_then_declared_rules(tmp_path):
    """The checkpointer must read the tracker the pipeline feeds: a second,
    unfed tracker once recorded every step after the first as clean reuse."""
    from sprout.harness_core.journal import JournalWriter
    from sprout.harness_orchestrator.run import OwnedSandbox, RunSpec
    from sprout.harness.tools.interceptors import MutationTracker, ShellContext

    limits = object()
    spec = RunSpec(prompt="x", interceptors=[{"name": "shell_context", "cwd": "/app", "user": "root"}])
    with JournalWriter(tmp_path / "j.jsonl") as journal:
        tools = Orchestrator()._tool_access(
            spec, journal, OwnedSandbox(session=_FakeSession(), sandbox_id="sb"), None,
            rollout_controls=limits)
    first, tracker, declared = tools.pipeline.interceptors
    assert first is limits
    assert isinstance(tracker, MutationTracker) and tools.checkpointer.tracker is tracker
    assert isinstance(declared, ShellContext) and declared.cwd == "/app"


def test_a_failed_final_snapshot_keeps_the_sandbox_on_the_record(tmp_path, fake_sandbox):
    """Until the final snapshot exists the sandbox is the only copy of the result."""
    from sprout.harness_orchestrator import ResourceLedger

    class NoSnapshotSession(FakeSession):
        def snapshot(self, name=None, disk_only=True):
            return None

    fake_sandbox.session = NoSnapshotSession()
    ledger = ResourceLedger(tmp_path / "ledger.jsonl")
    outcome = Orchestrator(out_dir=tmp_path, ledger=ledger).run(spec(tmp_path))
    assert outcome.status == "error" and "Final snapshot failed" in outcome.error
    provisioned = fake_sandbox[0]
    assert provisioned.keep is True and not provisioned.destroyed
    retained = [r for r in read_journal(tmp_path / "j.jsonl") if r["type"] == "sandbox.retained"]
    assert retained and retained[0]["reason"] == "snapshot_pending"
    claims = [e for e in ledger.entries() if e.get("kind") == "sandbox" and e.get("event") == "claim"]
    assert claims[-1]["keep"] is True
