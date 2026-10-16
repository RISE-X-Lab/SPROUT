"""One journal -> a :class:`Trajectory`: turns, calls, captures and where a branch came from.

The reader walks the events once and groups them by turn. Checkpoint records are
paired to turns by ``turn_id`` afterwards, never by position: turn K's record is
emitted after turn K+1's model response, and the final one after
``run.finished``. Recorded paths in these files are absolute and roots move, so
:func:`relocate` finds a recorded file beside the file that names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re

from sprout.harness_checkpoint.rollback import branch_checkpoints
from sprout.harness.core.events import (
    AGENT_MESSAGE, AGENT_THINKING, CHECKPOINT_CAPTURED, RUN_FINISHED, RUN_RESULT, RUN_STARTED,
    SANDBOX_CREATED, SNAPSHOT_MEMORY, TOOL_FINISHED, TOOL_STARTED, TURN_COMPLETED,
    TURN_STARTED, USAGE_UPDATED,
)
from sprout.harness.core.journal import journal_events

#: Event types the harness emits by literal name (rollback.py names
#: ``model.turn.completed`` the same way).
FORK_ORIGIN = "fork.origin"
MINI_TURN_REJECTED = "mini.turn.rejected"
BRANCH_ASSISTANT_TURN = "branch.assistant_turn"
TOOL_REJECTED = "tool.rejected"
MODEL_TURN_COMPLETED = "model.turn.completed"
ROLLOUT_USAGE = "rollout.usage"
EXECUTION_UNCERTAIN = "execution.uncertain"
SNAPSHOTS_SWEPT = "snapshots.swept"
SNAPSHOT_RELEASED = "snapshot.released"

#: mini-swe-agent wraps every command: ``env PAGER=cat ... bash -c '<command>'``.
_MINI_WRAPPER = re.compile(r"^env (?:\w+=\S* )*bash -c '(.*)'$", re.S)
#: A leading ``set -e`` / ``set -euo pipefail`` or ``cd <dir>`` and the separator after it
#: (``;``, ``&&`` or a line break): preamble, not what the command does.
_PREAMBLE = re.compile(r"^(?:set -[A-Za-z]+(?: pipefail)?|cd [^\s;&]+)\s*(?:&&|;|\n)\s*")
_USAGE_KEYS = ("input_tokens", "output_tokens", "cached_input_tokens", "cache_creation_tokens",
               "reasoning_output_tokens", "cost_usd")


def load_events(path: Path) -> tuple[list[dict], bool]:
    """Every complete record and whether the last line is torn (a live writer's half line)."""
    events = journal_events(path)
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        data = b""
    return events, bool(data) and not data.endswith(b"\n")


@dataclass(frozen=True)
class Call:
    call_id: str
    step: int
    name: str
    command: str            # the model's command, mini's wrapper removed
    args: dict
    status: str | None      # tool.finished.status; None while running
    exit_code: int | None
    timed_out: bool
    output: str
    rejected: bool


@dataclass(frozen=True)
class Capture:
    """One ``checkpoint.captured`` (or ``snapshot.memory``) record."""
    seq: int
    step: int
    turn: int | None
    turn_id: str | None
    snapshot_id: str | None
    reason: str
    captured: bool
    disk_only: bool | None
    final: bool
    uncertain: bool
    delta_empty: bool | None
    kind: str               # disk | memory
    branch_point: bool      # closed turn with a complete pair (rollback.branch_checkpoints)


@dataclass(frozen=True)
class Turn:
    number: int
    source: str             # actor | reviewer
    turn_id: str | None
    thinking: str
    message: str
    assistant_turn: dict | None
    calls: tuple[Call, ...]
    closed: bool
    rejected: bool
    rejected_source: str | None   # mini.turn.rejected.source: schema_validation | malformed_tool_calls | no_tool_calls
    step: int | None        # the turn's last call (model.turn.completed.step)
    capture: Capture | None
    started: str | None
    finished: str | None


@dataclass(frozen=True)
class Origin:
    """``fork.origin``, the first record of a branch."""
    parent_journal: str
    branch_step: int
    snapshot_id: str
    conversation_cut: str
    branch_guidance: str
    parent_journal_sha256: str | None
    hint: str | None


@dataclass(frozen=True)
class Trajectory:
    path: Path
    run_id: str
    events: int
    torn: bool
    slot: str | None
    slot_version: str | None
    model: str | None
    task_prompt: str
    sandbox_id: str | None
    backend: str | None
    store: str | None
    image: str | None
    origin: Origin | None
    turns: tuple[Turn, ...]
    captures: tuple[Capture, ...]
    status: str | None      # the LAST run.finished; None = live or killed
    error: str | None
    final_text: str
    usage: dict
    model_calls: int | None
    tool_calls: int | None
    swept: bool
    released: tuple[str, ...]
    uncertain: bool
    first_ts: str | None
    last_ts: str | None


def command_of(args: dict) -> str:
    """The model's command text: mini's ``env ... bash -c '...'`` wrapper removed."""
    command = args.get("command") if isinstance(args, dict) else None
    if not isinstance(command, str):
        return json.dumps(args, ensure_ascii=False)
    match = _MINI_WRAPPER.match(command)
    return match.group(1).replace("'\"'\"'", "'") if match else command


def unpack_output(event: dict) -> str:
    """A finished call's text: stdout, then stderr, from ``outcome`` or the runtime's envelope."""
    outcome = event.get("outcome")
    if not isinstance(outcome, dict):
        raw = event.get("output")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            outcome = parsed if isinstance(parsed, dict) and "stdout" in parsed else {"stdout": raw}
        else:
            outcome = {"stdout": json.dumps(raw, ensure_ascii=False) if raw is not None else ""}
    text = outcome.get("stdout") or ""
    if outcome.get("stderr"):
        text += "\n[stderr]\n" + outcome["stderr"]
    return text


def _outcome(event: dict) -> tuple[int | None, bool]:
    outcome = event.get("outcome")
    if not isinstance(outcome, dict):
        raw = event.get("output")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else None
        except ValueError:
            parsed = None
        outcome = parsed if isinstance(parsed, dict) and "exit_code" in parsed else {}
        if not outcome and event.get("status") == "ok":
            return 0, False
    code = outcome.get("exit_code")
    return (code if isinstance(code, int) else None), bool(outcome.get("timed_out"))


class _TurnDraft:
    """A turn while its events are still arriving."""

    def __init__(self, number: int, source: str, started: str | None) -> None:
        self.number, self.source, self.started = number, source, started
        self.turn_id: str | None = None
        self.thinking: list[str] = []
        self.message: list[str] = []
        self.assistant_turn: dict | None = None
        self.calls: list[dict] = []
        self.closed = self.rejected = False
        self.rejected_source: str | None = None
        self.step: int | None = None
        self.finished: str | None = None

    def call(self, call_id: str) -> dict | None:
        return next((c for c in self.calls if c["call_id"] == call_id), None)

    def freeze(self, capture: Capture | None) -> Turn:
        calls = tuple(Call(**c) for c in self.calls)
        return Turn(number=self.number, source=self.source, turn_id=self.turn_id,
                    thinking="\n\n".join(self.thinking), message="\n\n".join(self.message),
                    assistant_turn=self.assistant_turn, calls=calls, closed=self.closed,
                    rejected=self.rejected, rejected_source=self.rejected_source, step=self.step, capture=capture,
                    started=self.started, finished=self.finished)


def read_trajectory(path: Path) -> Trajectory:
    path = Path(path)
    events, torn = load_events(path)
    header: dict = {"run_id": None, "slot": None, "slot_version": None, "model": None, "task_prompt": "",
                    "sandbox_id": None, "backend": None, "store": None, "image": None}
    origin = None
    drafts: list[_TurnDraft] = []
    finished: list[dict] = []
    usage_updates: list[dict] = []
    final_text, model_calls, tool_calls = "", None, None
    swept = uncertain = False
    released: list[str] = []
    raw_captures: list[dict] = []
    stamps = [e["ts"] for e in events if isinstance(e.get("ts"), str)]

    def open_turn(event: dict) -> _TurnDraft:
        draft = _TurnDraft(len(drafts) + 1, event.get("source") or "actor", event.get("ts"))
        drafts.append(draft)
        return draft

    def current(event: dict) -> _TurnDraft:
        if not drafts or drafts[-1].finished is not None and event.get("type") == TOOL_STARTED:
            return open_turn(event)
        return drafts[-1]

    for event in events:
        kind = event.get("type")
        if header["run_id"] is None and event.get("run_id"):
            header["run_id"] = event["run_id"]
        if kind == FORK_ORIGIN:
            origin = Origin(parent_journal=str(event.get("parent_journal") or ""),
                            branch_step=int(event.get("branch_step") or 0),
                            snapshot_id=str(event.get("snapshot_id") or ""),
                            conversation_cut=str(event.get("conversation_cut") or ""),
                            branch_guidance=str(event.get("branch_guidance") or ""),
                            parent_journal_sha256=event.get("parent_journal_sha256"),
                            hint=event.get("actor_hint"))
        elif kind == SANDBOX_CREATED:
            header.update(sandbox_id=event.get("sandbox_id"), backend=event.get("backend"),
                          store=event.get("store"), image=event.get("image"))
        elif kind == RUN_STARTED:
            header.update(slot=event.get("slot"), slot_version=event.get("slot_version"),
                          model=event.get("model"), task_prompt=event.get("task_prompt") or "")
        elif kind == TURN_STARTED:
            draft = open_turn(event)
            if isinstance(event.get("turn"), int):
                draft.number = event["turn"]
        elif kind in (AGENT_MESSAGE, AGENT_THINKING) and drafts:
            (drafts[-1].message if kind == AGENT_MESSAGE else drafts[-1].thinking).append(event.get("text") or "")
        elif kind == BRANCH_ASSISTANT_TURN and drafts:
            drafts[-1].assistant_turn = event.get("message")
            drafts[-1].turn_id = drafts[-1].turn_id or event.get("turn_id")
        elif kind == TOOL_STARTED:
            draft = current(event)
            draft.turn_id = draft.turn_id or event.get("turn_id")
            args = event.get("args") if isinstance(event.get("args"), dict) else {}
            draft.calls.append({"call_id": str(event.get("call_id")), "step": event.get("step") or 0,
                                "name": event.get("name") or "", "command": command_of(args), "args": args,
                                "status": None, "exit_code": None, "timed_out": False, "output": "",
                                "rejected": False})
        elif kind in (TOOL_FINISHED, TOOL_REJECTED):
            call = next((c for d in reversed(drafts) if (c := d.call(str(event.get("call_id"))))), None)
            if call is None and drafts:
                drafts[-1].calls.append(call := {
                    "call_id": str(event.get("call_id")), "step": 0, "name": event.get("name") or "",
                    "command": "", "args": {}, "status": None, "exit_code": None, "timed_out": False,
                    "output": "", "rejected": False})
            if call is None:
                continue
            if kind == TOOL_REJECTED:
                call.update(rejected=True, output=str(event.get("feedback") or ""))
            else:
                code, timed_out = _outcome(event)
                call.update(status=event.get("status"), exit_code=code, timed_out=timed_out,
                            output=unpack_output(event))
        elif kind == MODEL_TURN_COMPLETED and drafts:
            drafts[-1].closed = True
            drafts[-1].turn_id = event.get("turn_id") or drafts[-1].turn_id
            if isinstance(event.get("step"), int):
                drafts[-1].step = event["step"]
        elif kind == MINI_TURN_REJECTED and drafts:
            drafts[-1].rejected_source = str(event.get("source") or "") or None
        elif kind == TURN_COMPLETED and drafts:
            drafts[-1].finished = event.get("ts") or ""
            drafts[-1].rejected = bool(event.get("rejected"))
        elif kind in (CHECKPOINT_CAPTURED, SNAPSHOT_MEMORY):
            raw_captures.append(event)
        elif kind == RUN_RESULT:
            final_text = event.get("text") or ""
        elif kind == RUN_FINISHED:
            finished.append(event)
        elif kind == USAGE_UPDATED and isinstance(event.get("usage"), dict):
            usage_updates.append(event["usage"])
        elif kind == ROLLOUT_USAGE:
            model_calls, tool_calls = event.get("model_calls"), event.get("tool_calls")
        elif kind == EXECUTION_UNCERTAIN:
            uncertain = True
        elif kind == SNAPSHOTS_SWEPT:
            swept = True
        elif kind == SNAPSHOT_RELEASED and event.get("snapshot_id"):
            released.append(str(event["snapshot_id"]))

    eligible = {c.seq for c in branch_checkpoints(None, events=events).values()}
    turn_numbers = {d.turn_id: d.number for d in drafts if d.turn_id}
    captures = tuple(Capture(
        seq=e.get("seq") or 0, step=e.get("step") or 0, turn=turn_numbers.get(e.get("turn_id")),
        turn_id=e.get("turn_id"), snapshot_id=e.get("snapshot_id"), reason=e.get("reason") or "captured",
        captured=bool(e.get("captured", bool(e.get("snapshot_id")))), disk_only=e.get("disk_only"),
        final=bool(e.get("final")), uncertain=bool(e.get("uncertain")), delta_empty=e.get("delta_empty"),
        kind="memory" if e.get("type") == SNAPSHOT_MEMORY else "disk",
        branch_point=(e.get("seq") or 0) in eligible) for e in raw_captures)
    by_turn: dict[str, Capture] = {}
    for capture in captures:
        if capture.kind == "disk" and capture.turn_id and capture.turn_id not in by_turn:
            by_turn[capture.turn_id] = capture
    last = finished[-1] if finished else None
    if last is not None and isinstance(last.get("usage"), dict) and last["usage"]:
        usage = dict(last["usage"])
    else:
        usage = {key: sum(u.get(key) or 0 for u in usage_updates) for key in _USAGE_KEYS}
    return Trajectory(
        path=path, run_id=header["run_id"] or path.stem, events=len(events), torn=torn,
        slot=header["slot"], slot_version=header["slot_version"], model=header["model"],
        task_prompt=header["task_prompt"], sandbox_id=header["sandbox_id"], backend=header["backend"],
        store=header["store"], image=header["image"],
        origin=origin, turns=tuple(d.freeze(by_turn.get(d.turn_id)) for d in drafts), captures=captures,
        status=last.get("status") if last else None, error=last.get("error") if last else None,
        final_text=final_text, usage=usage, model_calls=model_calls, tool_calls=tool_calls,
        swept=swept, released=tuple(released), uncertain=uncertain,
        first_ts=stamps[0] if stamps else None, last_ts=stamps[-1] if stamps else None)


def turn_count(trajectory: Trajectory) -> tuple[int, int]:
    """(turns started, turns completed): ``turn 74`` of a live run is the one in flight."""
    return len(trajectory.turns), sum(1 for t in trajectory.turns if t.finished is not None)


def step_count(trajectory: Trajectory) -> int:
    return sum(len(t.calls) for t in trajectory.turns)


def parse_ts(stamp: str | None) -> datetime | None:
    if not stamp:
        return None
    try:
        return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def minutes(trajectory: Trajectory) -> float | None:
    first, last = parse_ts(trajectory.first_ts), parse_ts(trajectory.last_ts)
    if first is None or last is None:
        return None
    return (last - first).total_seconds() / 60


def summarize_command(command: str, width: int = 40) -> str:
    """One line of a command: no leading ``set -e`` or ``cd <dir>`` (joined by ``;``, ``&&``
    or a line break), whitespace collapsed."""
    text = command.strip()
    while (stripped := _PREAMBLE.sub("", text, count=1)) != text:
        text = stripped
    text = " ".join(text.split())
    return text if len(text) <= width else text[:width] + "…"


def clip(text: str, limit: int) -> str:
    """Head 2/3 and tail 1/3 of ``text`` within ``limit`` chars; 0 = unlimited.

    A test run's verdict is at the end, so the tail is kept.
    """
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + "\n… [%d chars elided] …\n" % (len(text) - limit) + text[-(limit - head):]


def relocate(recorded: str, anchor: Path, *, min_parts: int = 4) -> Path | None:
    """Where a recorded absolute path is now, looking beside ``anchor`` when it has moved.

    The longest trailing part of ``recorded`` that exists under an ancestor of
    ``anchor`` wins (nearest ancestor first), never ``anchor`` itself: longest
    first stops ``agent/trajectory.jsonl`` from resolving to a branch's own
    journal, and ``min_parts`` keeps ``trials/<t>/agent/trajectory.jsonl`` the
    shortest acceptable match.
    """
    if not recorded:
        return None
    path = Path(recorded)
    if path.exists():
        return path
    names = [part for part in path.parts if part != path.anchor]
    anchor = Path(anchor).resolve()
    for length in range(len(names), min_parts - 1, -1):
        suffix = Path(*names[-length:])
        for ancestor in anchor.parents:
            candidate = ancestor / suffix
            if candidate.exists() and candidate != anchor:
                return candidate
    return None
