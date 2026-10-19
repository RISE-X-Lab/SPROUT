"""Runs of the mini slot for its tests: a sandbox that runs commands with bash, and ``run_mini``.

``run_slot`` calls ``MiniSweSlot`` the way ``sprout.worker.run.runner.Runner`` calls it. It resolves
the workspace, runs the slot through a ``ToolAccess`` that checkpoints each turn, and takes the final
snapshot. ``run_mini`` does the same against a scripted model (``tests.support.http.chat_server``).
"""

from __future__ import annotations

import importlib.util
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import read_entries
from sprout.worker.agents.mini_swe.sandbox import resolve_workspace
from sprout.worker.agents.mini_swe.slot import MiniSweSlot
from sprout.worker.agents.slot import SlotResult, TaskSpec
from sprout.worker.checkpoint.checkpointer import Checkpointer
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.tools.access import ToolAccess
from sprout.worker.tools.result import CommandOutcome, ToolResult
from tests.support.http import chat_server

needs_mini = pytest.mark.skipif(
    importlib.util.find_spec("minisweagent") is None,
    reason='mini-swe-agent is not installed: pip install -e ".[mini]"',
)

#: The fields that the journal adds to each event.
ENVELOPE = {"v", "ts", "seq", "run_id", "agent_id", "sandbox_id"}
#: The command that ends a mini run with a submission: the first line of its output.
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
#: The bash tool that mini 2.4.6 offers, with the argument limit that Sprout adds.
BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class Snapshot:
    """A snapshot as a ``Checkpointer`` reads it. A snapshot without a layer count never re-boards."""

    id: str
    rootfs_layers: int | None = None


def agent_command(line: str) -> str:
    """The agent's own command in a shell line that mini sends: the last argument of ``bash -c``."""
    return shlex.split(line)[-1]


class ShellSandbox:
    """A sandbox session that runs each shell call with bash in ``root``, which stands for ``/testbed``.

    ``scripted`` answers an agent command, or a whole shell line, with its result instead. ``on_call``
    sees each call first. The workspace probes answer ``/testbed``, in ``repository`` if it is given.
    """

    sandbox_id = "sandbox-fixture"

    def __init__(
        self,
        root: Path,
        *,
        scripted: dict[str, ToolResult] | None = None,
        on_call: Callable[[dict[str, Any]], None] | None = None,
        repository: str | None = None,
    ) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.scripted = scripted or {}
        self.on_call = on_call
        self.repository = repository
        #: The arguments of each shell call, in order.
        self.calls: list[dict[str, Any]] = []
        #: The files of each snapshot, by snapshot id.
        self.snapshots: dict[str, dict[str, bytes]] = {}

    @property
    def commands(self) -> list[str]:
        """The agent command of each shell call, in order."""
        return [agent_command(args["command"]) for args in self.calls]

    def execute(self, name: str, args: dict[str, Any], timeout: float = 30) -> ToolResult:
        """Answer a workspace probe: ``pwd -P`` and ``git rev-parse --show-toplevel``."""
        if args["command"] == "pwd -P":
            return ToolResult(True, args.get("working_dir", "/testbed") + "\n")
        if self.repository is None:
            return ToolResult(False, "", outcome=CommandOutcome(exit_code=128, stderr="not a git repository"))
        return ToolResult(True, self.repository + "\n")

    def call(self, name: str, args: dict[str, Any], timeout: float = 300) -> ToolResult:
        self.calls.append(dict(args))
        if self.on_call is not None:
            self.on_call(args)
        line = args["command"]
        scripted = self.scripted.get(line, self.scripted.get(agent_command(line)))
        if scripted is not None:
            return scripted
        completed = subprocess.run(
            ["bash", "-c", args["command"]], cwd=self.root, capture_output=True, text=True, timeout=60
        )
        # The runtime sends bare stdout for a command that exits 0 with nothing on stderr.
        if completed.returncode == 0 and not completed.stderr:
            return ToolResult(True, completed.stdout)
        outcome = CommandOutcome(
            exit_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr
        )
        return ToolResult(completed.returncode == 0, completed.stdout, outcome=outcome)

    def supports_snapshot(self) -> bool:
        return True

    def snapshot(self, *, disk_only: bool) -> Snapshot:
        snapshot = Snapshot(id="snapshot-%d" % (len(self.snapshots) + 1))
        self.snapshots[snapshot.id] = {
            str(path.relative_to(self.root)): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        return snapshot


@dataclass(frozen=True)
class Rollout:
    """What one run of the slot left: its result, its journal, its native history and its model requests."""

    result: SlotResult
    journal_path: Path
    sandbox: ShellSandbox
    control: RunControl
    #: The JSON body of each model request, in order.
    requests: list[dict[str, Any]] = field(default_factory=list)

    def events(self, kind: str) -> list[dict[str, Any]]:
        """The journal records of type ``kind``, in order, without the envelope."""
        return [
            {key: value for key, value in record.items() if key not in ENVELOPE}
            for record in read_journal(self.journal_path)
            if record["type"] == kind
        ]

    @property
    def transcript(self) -> Path:
        """The native history that the last ``session.ref`` of the journal names."""
        return Path(self.events("session.ref")[-1]["transcript_path"])

    def entries(self) -> list[dict[str, Any]]:
        """The entries of the native history."""
        return read_entries(self.transcript.read_bytes())


def text_reply(content: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    """A chat completion whose assistant message has ``content`` and no tool call."""
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": "fixture",
        "choices": [
            {"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }


def mini_task(
    *, base_url: str | None, home: Path, extra: dict[str, Any] | None = None, **changes: Any
) -> TaskSpec:
    """A task for the model ``fixture`` at ``base_url``, with its native history under ``home``."""
    values: dict[str, Any] = {
        "prompt": "Write the answer.",
        "cwd": "/testbed",
        "model": "fixture",
        "env": {} if base_url is None else {"OPENAI_BASE_URL": base_url},
        "timeout_s": 3600.0,
        "extra": {"native_home": str(home), **(extra or {})},
        "control": RunControl(),
    }
    values.update(changes)
    return TaskSpec(**values)


def run_slot(task: TaskSpec, sandbox: ShellSandbox, journal_path: Path) -> Rollout:
    """Run ``task`` in ``sandbox`` as the runner does, and return what the run left."""
    assert task.control is not None
    journal = JournalWriter(journal_path, run_id="run-fixture")
    try:
        tools = ToolAccess(sandbox, journal, control=task.control, checkpointer=Checkpointer(session=sandbox))
        resolved = replace(task, extra=resolve_workspace(sandbox, task.extra, journal))
        result = MiniSweSlot().run(resolved, journal, tools)
        tools.finalize()
    finally:
        journal.close()
    return Rollout(result=result, journal_path=journal_path, sandbox=sandbox, control=task.control)


def run_mini(
    directory: Path,
    replies: list[Any],
    *,
    sandbox: ShellSandbox | None = None,
    base_path: str | None = "",
    **changes: Any,
) -> Rollout:
    """Run the slot in ``directory`` against a model that answers with ``replies``.

    The model's base URL is the server's URL and ``base_path``. ``base_path`` None sets no URL.
    ``changes`` change the task of ``mini_task``.
    """
    with chat_server(replies) as server:
        base_url = None if base_path is None else server.url + base_path
        task = mini_task(base_url=base_url, home=directory / "native-home", **changes)
        rollout = run_slot(task, sandbox or ShellSandbox(directory / "sandbox"), directory / "journal.jsonl")
    return replace(rollout, requests=server.requests)


def outline(entries: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    """Each entry of a native history in short: its type, and the role and call ids of a message."""
    return [_outline(entry) for entry in entries]


def _outline(entry: dict[str, Any]) -> tuple[Any, ...]:
    if entry["type"] == "mini.turn":
        return ("turn", entry["call_ids"])
    if entry["type"] != "mini.message":
        return (entry["type"],)
    message = entry["message"]
    if message.get("tool_calls"):
        return ("message", message["role"], [call["id"] for call in message["tool_calls"]])
    if message.get("tool_call_id"):
        return ("message", message["role"], message["tool_call_id"])
    return ("message", message["role"])


def with_tool_name(reply: dict[str, Any], *, index: int, name: str) -> dict[str, Any]:
    """``reply`` with the function name of its tool call ``index`` set to ``name``."""
    reply["choices"][0]["message"]["tool_calls"][index]["function"]["name"] = name
    return reply


def stop_at(
    control: RunControl, command: str, *, stop_reason: str = "timeout"
) -> Callable[[dict[str, Any]], None]:
    """A sandbox ``on_call`` that stops the run when the agent runs ``command``."""

    def on_call(args: dict[str, Any]) -> None:
        if agent_command(args["command"]) == command:
            control.request_stop("stopped at %s" % command, stop_reason=stop_reason)

    return on_call
