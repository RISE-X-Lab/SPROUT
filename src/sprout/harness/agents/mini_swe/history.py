"""mini's native history: durable messages, closed turns and their branch points.

A branch point is a closed model turn whose last tool call has an exact
snapshot; ``index_mini`` finds them, ``reference_at`` names the prefix a branch
resumes from, and ``materialize`` writes that prefix for a new attempt.
"""

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from sprout.harness.agents.mini_swe.tools import validate_mini_tools
from sprout.harness_checkpoint.rollback import branch_checkpoints, closed_turns
from sprout.harness.core.journal import journal_events, read_journal

FORMAT = "sprout-mini-v1"
FORMATS = {FORMAT, "ash-mini-v1"}  # the second: histories written before the rename


def read_entries(data: bytes) -> list[dict]:
    return [json.loads(line) for line in data.splitlines(keepends=True)
            if line.endswith(b"\n") and line.strip()]


def training_messages(entries: list[dict]) -> list[dict]:
    messages = []
    for entry in entries:
        if entry.get("type") != "mini.message":
            continue
        native = entry["message"]
        if native.get("role") == "exit":
            continue
        message = {key: deepcopy(native[key]) for key in (
            "role", "content", "tool_calls", "tool_call_id", "reasoning_content")
            if native.get(key) is not None}
        message.setdefault("content", "")
        if not isinstance(message["content"], str):
            raise ValueError("mini training requires text content")
        if message.get("tool_calls"):
            message["tool_calls"] = [
                {"id": call["id"], "type": call["type"],
                 "function": {"name": call["function"]["name"],
                              "arguments": call["function"]["arguments"]}}
                for call in message["tool_calls"]
            ]
        messages.append(message)
    return messages


def load_prefix(reference: dict) -> list[dict]:
    path = Path(reference["path"])
    with path.open("rb") as stream:
        data = stream.read(reference.get("byte_length", -1))
    if (not data.endswith(b"\n") or hashlib.sha256(data).hexdigest() != reference["sha256"]):
        raise ValueError("mini native prefix checksum changed")
    entries = read_entries(data)
    if not entries or entries[-1].get("type") != "mini.turn":
        raise ValueError("mini restoration requires a closed model turn")
    metadata = next((e for e in entries if e.get("type") == "mini.session"), {})
    if metadata.get("format") not in FORMATS or metadata.get("mini_version") != "2.4.6":
        raise ValueError("Unsupported mini native history version")
    return entries


class History:
    def __init__(self, path: Path | str, session_id: str, *, prefix=(), workspace=None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("x", encoding="utf-8")
        metadata = {"type": "mini.session", "format": FORMAT,
                    "mini_version": "2.4.6", "session_id": session_id}
        if workspace is not None:
            metadata["workspace"] = deepcopy(workspace)
        self.append(metadata)
        self.prefix_messages = []
        for entry in prefix:
            if entry["type"] != "mini.session":
                self.append(entry)
            if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
                self.prefix_messages.append(deepcopy(entry["message"]))

    def append(self, entry: dict) -> None:
        self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self) -> None:
        self.stream.close()


#: An assistant turn written outside the actor, as its author writes it; the
#: canonical rendering (``continuation.render_reasoning``) moves its text into
#: the one key ``REASONING_TURN_KEYS`` adds.
TURN_KEYS = frozenset({"role", "content", "tool_calls"})
REASONING_TURN_KEYS = TURN_KEYS | {"reasoning_content"}


def validate_assistant_turn(message: dict, *, history: Sequence[dict] = (),
                            tools: list[dict] | None = None, allow_reasoning: bool = False) -> dict:
    """Return an owned copy; observations and provider thinking are never supplied.

    ``allow_reasoning`` admits the canonical rendering as well: the text as
    non-blank ``reasoning_content`` beside a ``content`` of at most one line,
    usually empty. Only mini's runtime passes it, since it runs a turn
    ``continuation.prepare`` may already have rendered; the turn its author
    writes is checked in its own form, without reasoning.
    """
    if tools is not None:
        tools = validate_mini_tools(tools)
    keys = set(message) if isinstance(message, dict) else None
    if allow_reasoning and keys == REASONING_TURN_KEYS:
        if message["role"] != "assistant" or not isinstance(message["content"], str) or "\n" in message["content"]:
            raise ValueError("assistant_turn with reasoning_content requires assistant role "
                             "and content of at most one line")
        reasoning = message["reasoning_content"]
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError("assistant_turn reasoning_content must be non-empty text")
    elif keys != TURN_KEYS:
        raise ValueError("assistant_turn requires exactly role, content and tool_calls"
                         + (", or those and reasoning_content" if allow_reasoning else ""))
    elif message["role"] != "assistant" or not isinstance(message["content"], str) or not message["content"].strip():
        raise ValueError("assistant_turn requires assistant role and non-empty ordinary content")
    calls = message["tool_calls"]
    if not isinstance(calls, list) or not calls:
        raise ValueError("assistant_turn requires at least one bash tool call")
    used = {call["id"] for entry in history for call in entry.get("tool_calls") or []}
    for index, call in enumerate(calls):
        if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
            raise ValueError("assistant_turn tool calls require id, type and function")
        identifier = call["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in used:
            raise ValueError("assistant_turn tool call ids must be non-empty and unique across history")
        used.add(identifier)
        function = call["function"]
        if (call["type"] != "function" or not isinstance(function, dict)
                or set(function) != {"name", "arguments"}
                or not isinstance(function["arguments"], str)):
            raise ValueError("assistant_turn supports only bash with JSON-string arguments")
        if function["name"] != "bash":
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) has forbidden tool name "
                f"{function['name']!r}; only bash is allowed. Return function.name='bash' "
                "with arguments containing exactly one string field, command. "
                "No calls from this plan have executed."
            )
        try:
            arguments = json.loads(function["arguments"])
        except ValueError as error:
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments are not valid JSON: {error}"
            ) from error
        if (not isinstance(arguments, dict) or set(arguments) != {"command"}
                or not isinstance(arguments["command"], str) or not arguments["command"].strip()):
            raise ValueError(
                f"assistant_turn.tool_calls[{index}] ({identifier}) bash arguments require exactly one non-empty string command"
            )
        if tools is not None:
            from jsonschema import Draft202012Validator, ValidationError

            try:
                Draft202012Validator(tools[0]["function"]["parameters"]).validate(arguments)
            except ValidationError as error:
                raise ValueError(f"assistant_turn does not match the recorded actor tool schema: {error.message}") from error
    return deepcopy(message)


@dataclass(frozen=True)
class NativePoint:
    message_step: int
    tool_depth: int
    snapshot_id: str
    native: dict


def read_prefix(reference: dict) -> bytes:
    length = reference["byte_length"]
    if type(length) is not int or length <= 0:
        raise ValueError("Invalid native prefix length")
    with Path(reference["path"]).open("rb") as stream:
        data = stream.read(length)
    if len(data) != length or not data.endswith(b"\n"):
        raise ValueError("Native prefix missing or truncated")
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("Native prefix checksum changed")
    return data


def valid_prefix(reference: dict) -> bool:
    try:
        read_prefix(reference)
        return True
    except (OSError, KeyError, TypeError, ValueError):
        return False


def index_native(journal: Path, transcript: Path, slot: str, session_id: str, *,
                 events: list[dict] | None = None,
                 inherited_native: dict | None = None) -> list[NativePoint]:
    """Recovery points of one attempt: closed turns with an exact snapshot."""
    if slot != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    if events is None:
        # Split as bytes, like every journal reader: str.splitlines() also
        # breaks at U+0085 and U+2028, which a tool's output keeps verbatim.
        events = journal_events(journal)
    return index_mini(journal, transcript, session_id, events=list(events),
                      inherited_native=inherited_native)


def materialize(reference: dict, cwd: Path, destination: Path, native_home: Path) -> dict:
    data = read_prefix(reference)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prefix.jsonl"
    path.write_bytes(data)
    if reference["slot"] != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    return {"native_prefix": {"path": str(path.resolve()), "sha256": reference["sha256"]}}


def index_mini(journal: Path, transcript: Path, session_id: str, *,
               events: list[dict], inherited_native: dict | None = None) -> list:
    local_calls = [e["call_id"] for e in events if e.get("type") == "tool.started"]
    positions = {call: i for i, call in enumerate(local_calls, 1)}
    admitted = set(local_calls)
    rejection_events = {e["call_id"]: e for e in events
                        if e.get("type") == "tool.rejected" and e.get("source") == "schema_validation"
                        and e.get("executed") is False and e.get("call_id")}
    rejected_turns = {e["turn_id"]: e for e in events
                      if e.get("type") == "mini.turn.rejected" and e.get("source") == "schema_validation"
                      and e.get("executed") is False and e.get("turn_id")}
    admitted.update(rejection_events)
    inherited_rejections = {}
    if inherited_native:
        if inherited_native["slot"] != "mini-swe-agent":
            raise ValueError("Inherited native prefix uses a different slot")
        for entry in read_entries(read_prefix(inherited_native)):
            admitted.update(call["id"] for call in entry.get("message", {}).get("tool_calls") or [])
            if entry.get("type") == "mini.turn" and entry.get("rejected") is True:
                inherited_rejections[entry["turn_id"]] = entry["call_ids"]
    checkpoints = branch_checkpoints(journal, events=events)
    closed = closed_turns(events)
    points, pending, seen, group = [], set(), set(), []
    checksum, offset, message_step = hashlib.sha256(), 0, 0
    for line in Path(transcript).read_bytes().splitlines(keepends=True):
        if not line.endswith(b"\n"):
            break
        row = json.loads(line)
        checksum.update(line)
        offset += len(line)
        kind = row.get("type")
        if kind == "mini.session":
            if row.get("format") not in FORMATS or row.get("mini_version") != "2.4.6":
                raise ValueError("Unsupported mini native history version")
        elif kind == "mini.message":
            message = row["message"]
            calls = message.get("tool_calls") or []
            if calls:
                if pending or group:
                    break
                group = [call["id"] for call in calls]
                if len(set(group)) != len(group) or not set(group) <= admitted or set(group) & seen:
                    break
                pending.update(group)
            if message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if call_id not in pending:
                    break
                if call_id in rejection_events and message.get("content") != rejection_events[call_id].get("feedback"):
                    break
                pending.remove(call_id)
                seen.add(call_id)
        elif kind == "mini.turn":
            if pending or row.get("call_ids") != group:
                break
            message_step += 1
            is_rejection = (bool(set(group) & set(rejection_events))
                            or row.get("turn_id") in rejected_turns
                            or row.get("turn_id") in inherited_rejections)
            if is_rejection or row.get("rejected") is True:
                if row.get("rejected") is not True:
                    break
                tid = row.get("turn_id")
                if set(group) & set(local_calls):
                    break
                if tid in inherited_rejections:
                    if inherited_rejections[tid] != group:
                        break
                elif (rejected_turns.get(tid, {}).get("call_ids") != group
                      or any(rejection_events.get(call, {}).get("turn_id") != tid for call in group)):
                    break
                group = []
                continue
            local = [call for call in group if call in positions]
            if local:
                depth = max(positions[call] for call in local)
                turn = closed.get(row.get("turn_id"), {})
                point = checkpoints.get(depth)
                if (turn.get("call_ids") != group or turn.get("step") != depth
                        or not set(local_calls[:depth]) <= seen or set(local_calls[depth:]) & seen):
                    break
                if point and point.turn_id == row["turn_id"] and point.session_ckpt == session_id:
                    points.append(NativePoint(message_step, depth, point.snapshot_id, {
                        "slot": "mini-swe-agent", "session_id": session_id,
                        "path": str(Path(transcript).resolve()), "byte_length": offset,
                        "sha256": checksum.hexdigest(), "cut": row["turn_id"],
                    }))
            group = []
        else:
            raise ValueError("Unknown mini native record")
    return points


def reference_at(journal: Path, step: int, session_id: str) -> dict | None:
    events = journal_events(Path(journal))
    reference = next((e for e in reversed(events) if e.get("type") == "session.ref"
                      and e.get("native_session_id") == session_id and e.get("transcript_path")), None)
    if reference is None:
        return None
    inherited = next((e.get("inherited_native") for e in events
                      if e.get("type") == "mini.restored"), None)
    points = index_mini(journal, reference["transcript_path"], session_id,
                        events=events, inherited_native=inherited)
    return next((point.native for point in points if point.tool_depth == step), None)


def conversation_restore(journal_path, step: int, session_ref: str) -> str | None:
    """The exact native cut at ``step``, or None when there is none."""
    if not any(row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent"
               for row in read_journal(journal_path)):
        return None
    reference = reference_at(journal_path, step, session_ref)
    return reference["cut"] if reference else None


def available_branch_points(journal_path):
    """The same candidates, indexing each native session once per inspection."""
    events = read_journal(journal_path)
    points = branch_checkpoints(journal_path, events=events)
    if not any(row.get("type") == "run.started" and row.get("slot") == "mini-swe-agent"
               for row in events):
        return {}
    # Match reference_at's last reference for each session. A local index avoids
    # reparsing the entire transcript once for every checkpoint, without keeping
    # stale state when the journal or native history changes between inspections.
    references = {
        row.get("native_session_id"): row
        for row in events
        if row.get("type") == "session.ref" and row.get("transcript_path")
    }
    inherited = next((row.get("inherited_native") for row in events
                      if row.get("type") == "mini.restored"), None)
    cuts = {}
    for session_id in dict.fromkeys(point.session_ckpt for point in points.values()):
        reference = references.get(session_id)
        if reference is None:
            continue
        indexed = index_mini(journal_path, reference["transcript_path"], session_id,
                             events=events, inherited_native=inherited)
        cuts[session_id] = {point.tool_depth: point.native["cut"] for point in indexed}
    return {step: point for step, point in points.items()
            if cuts.get(point.session_ckpt, {}).get(step)}


def actor_tools_at(journal, step: int) -> list[dict]:
    """Read the last actual tool declaration before the selected turn closed."""
    tools = None
    for event in read_journal(journal):
        if event.get("type") == "rollout.model_tools" and event.get("shape") == "chat/completions":
            tools = event.get("tools")
        if event.get("type") == "model.turn.completed" and event.get("step") == step:
            if tools is None:
                raise ValueError("No recorded actor tool schema at the selected checkpoint")
            return validate_mini_tools(tools)
    raise ValueError("No completed turn for the selected actor tool schema")
