"""Runs of the harness slot for its tests: a sandbox with a shell and a text editor, Qwen3.5 replies, and
``run_harness``.

``FileSandbox`` maps ``/testbed`` to a directory. Its ``text_editor`` answers as the runtime's does
(``runtime/tools/edit.go``). ``reply`` writes a chat completion whose log-probs carry the raw text, the
way SGLang returns one. ``run_harness`` calls ``HarnessSlot`` as the runner does.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sprout.contract.mini_history import read_entries
from sprout.worker.agents.harness.qwen_format import Call, FormatViolation, Turn, parse, render
from sprout.worker.agents.harness.slot import HarnessSlot, resolve_workspace
from sprout.worker.agents.slot import SlotResult, TaskSpec
from sprout.worker.checkpoint.checkpointer import Checkpointer
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.tools.access import ToolAccess
from sprout.worker.tools.interceptors import MutationTracker
from sprout.worker.tools.pipeline import ToolPipeline
from sprout.worker.tools.result import CommandOutcome, ToolResult
from tests.support.http import chat_server

#: The fields that the journal adds to each event.
ENVELOPE = {"v", "ts", "seq", "run_id", "agent_id", "sandbox_id"}


@dataclass(frozen=True)
class Snapshot:
    id: str
    rootfs_layers: int | None = None


class FileSandbox:
    """A sandbox session whose ``/testbed`` is ``root``: ``shell`` runs bash, ``text_editor`` edits files."""

    sandbox_id = "sandbox-fixture"

    def __init__(self, root: Path, on_call: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.root = root
        self.on_call = on_call
        self.root.mkdir(parents=True, exist_ok=True)
        #: The tool and arguments of each call, in order.
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.snapshots: list[str] = []

    def host(self, path: str) -> Path:
        return self.root / Path(path).relative_to("/testbed")

    def execute(self, name: str, args: dict[str, Any], timeout: float = 30) -> ToolResult:
        """Answer the workspace probes: ``/testbed`` is the working directory and the repository."""
        return ToolResult(True, args.get("working_dir", "/testbed") + "\n")

    def call(self, name: str, args: dict[str, Any], timeout: float = 300) -> ToolResult:
        self.calls.append((name, dict(args)))
        if self.on_call is not None:
            self.on_call(name, args)
        if name == "shell":
            done = subprocess.run(
                ["bash", "-c", args["command"]], cwd=self.root, capture_output=True, text=True
            )
            outcome = CommandOutcome(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)
            return ToolResult(done.returncode == 0, done.stdout, outcome=outcome)
        return self._edit(args)

    def _edit(self, args: dict[str, Any]) -> ToolResult:
        path = self.host(args["path"])
        if args["command"] == "view":
            if not path.exists():
                return ToolResult(False, "open %s: no such file or directory" % args["path"])
            lines = path.read_text().split("\n")
            first, last = args.get("view_range", [1, len(lines)])
            last = min(last, len(lines))
            return ToolResult(
                True, "".join("  %6d | %s\n" % (n, lines[n - 1]) for n in range(max(first, 1), last + 1))
            )
        if args["command"] == "write":
            existed = path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(args["file_text"])
            return ToolResult(True, "%s %s" % ("Overwrote" if existed else "Created", args["path"]))
        text = path.read_text()
        count = text.count(args["old_str"])
        if count != 1:
            return ToolResult(False, "No match found" if count == 0 else "Multiple matches (%d)" % count)
        path.write_text(text.replace(args["old_str"], args["new_str"]))
        return ToolResult(True, "Edited %s: replaced the text" % args["path"])

    def supports_snapshot(self) -> bool:
        return True

    def snapshot(self, *, disk_only: bool) -> Snapshot:
        self.snapshots.append("snapshot-%d" % (len(self.snapshots) + 1))
        return Snapshot(self.snapshots[-1])


def turn(name: str, *parameters: tuple[str, str], reasoning: str = "I know what to do next.") -> str:
    """The canonical raw text of a turn that calls ``name`` with ``parameters``."""
    return render(Turn(reasoning, "", (Call(name, parameters),)))


def reply(
    raw: str, *, finish_reason: str = "stop", tokens: tuple[int, int] = (100, 20), logprobs: bool = True
) -> dict[str, Any]:
    """The chat completion that SGLang returns for the sampled text ``raw``, parsed, with its log-probs."""
    try:
        parsed: Turn | None = parse(raw)
    except FormatViolation:
        parsed = None
    message: dict[str, Any] = {"role": "assistant", "content": raw}
    if parsed is not None:
        message = {"role": "assistant", "content": parsed.content, "reasoning_content": parsed.reasoning}
        if parsed.calls:
            message["tool_calls"] = [
                {
                    "id": "call-%d" % i,
                    "type": "function",
                    "function": {"name": c.name, "arguments": dict_json(c)},
                }
                for i, c in enumerate(parsed.calls)
            ]
    choice: dict[str, Any] = {"index": 0, "finish_reason": finish_reason, "message": message}
    if logprobs:
        pieces = [*raw, "<|im_end|>"] if finish_reason == "stop" else list(raw)
        choice["logprobs"] = {
            "content": [{"token": p, "bytes": list(p.encode()), "logprob": -0.1} for p in pieces]
        }
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "choices": [choice],
        "usage": {"prompt_tokens": tokens[0], "completion_tokens": tokens[1]},
    }


def dict_json(call: Call) -> str:
    import json

    return json.dumps(dict(call.parameters))


@dataclass(frozen=True)
class Rollout:
    """What one run of the slot left: its result, journal, native history, model requests and sandbox."""

    result: SlotResult
    journal_path: Path
    sandbox: FileSandbox
    control: RunControl
    requests: list[dict[str, Any]] = field(default_factory=list)

    def events(self, kind: str) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in record.items() if key not in ENVELOPE}
            for record in read_journal(self.journal_path)
            if record["type"] == kind
        ]

    @property
    def transcript(self) -> Path:
        return Path(self.events("session.ref")[-1]["transcript_path"])

    def entries(self) -> list[dict[str, Any]]:
        return read_entries(self.transcript.read_bytes())


def run_harness(
    directory: Path,
    replies: list[Any],
    *,
    harness: dict[str, Any] | None = None,
    timeout_s: float = 600.0,
    on_call: Callable[[str, dict[str, Any], RunControl], None] | None = None,
) -> Rollout:
    """Run the slot in ``directory`` against a model that answers with ``replies``, as the runner runs it.

    ``on_call`` sees each sandbox call with the run's control, before the call runs."""
    control = RunControl()
    sandbox = FileSandbox(
        directory / "testbed", None if on_call is None else lambda n, a: on_call(n, a, control)
    )
    journal = JournalWriter(directory / "attempt" / "trajectory.jsonl", run_id="run-fixture")
    tracker = MutationTracker()
    tools = ToolAccess(
        sandbox,  # type: ignore[arg-type]
        journal,
        pipeline=ToolPipeline([tracker]),
        control=control,
        checkpointer=Checkpointer(session=sandbox, tracker=tracker, disk_only=True),  # type: ignore[arg-type]
    )
    extra = {"harness": harness or {}, "native_home": str(directory / "attempt" / "native-home")}
    with chat_server(replies) as server:
        task = TaskSpec(
            prompt="Fix the bug in calc.py.",
            cwd=str(directory),
            model="qwen",
            env={"OPENAI_BASE_URL": server.url + "/v1", "OPENAI_API_KEY": "key"},
            timeout_s=timeout_s,
            extra=resolve_workspace(sandbox, extra, journal),  # type: ignore[arg-type]
            control=control,
        )
        try:
            result = HarnessSlot().run(task, journal, tools)
            tools.finalize()
        finally:
            journal.close()
    return Rollout(result, journal.path, sandbox, control, requests=server.requests)
