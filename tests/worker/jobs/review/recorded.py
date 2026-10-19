"""Finished mini-swe-agent attempts on disk, written by the journal and history writers that a run uses.

``record_attempt`` writes what a finished attempt leaves in its directory: the journal
``trajectory.jsonl`` (``JournalWriter``) and mini's native history ``native-home/ses-1.jsonl``
(``History``). Each turn makes one bash call for each of its commands. The checkpoint after a turn is
at the step of its last call, and names the snapshot ``<run>-snap-<step>``. ``TURNS`` is the default
attempt. Its branch points are at the steps 2, 3 and 4.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.worker.agents.mini_swe.history import History
from sprout.worker.run.journal import JournalWriter

SESSION = "ses-1"
WORKSPACE = {"cwd": "/app", "repository_dir": "/app"}
TASK = "Fix the failing test."
SYSTEM = "You are a helpful assistant that can interact with a computer."
#: The actor's tool, as mini declares it to the model.
BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


@dataclass(frozen=True)
class Turn:
    """One model turn: its text, one bash call for each command, and the output of each call."""

    content: str
    commands: tuple[str, ...]
    outputs: tuple[str, ...]


def turn(*commands: str, outputs: Sequence[str] | None = None, content: str = "Inspect the files.") -> Turn:
    """A turn whose calls run ``commands``. Each call prints ``ok`` unless ``outputs`` says otherwise."""
    return Turn(
        content=content,
        commands=commands,
        outputs=tuple(outputs) if outputs is not None else ("ok",) * len(commands),
    )


TURNS = (
    turn("echo early > answer", "cat answer", outputs=("", "early")),
    turn("echo late > answer"),
    turn("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"),
)


def record_attempt(
    directory: Path,
    turns: Sequence[Turn] = TURNS,
    *,
    run: str = "parent",
    task: str = TASK,
    status: str = "completed",
    failed_steps: Sequence[int] = (),
    final_text: str = "",
) -> Path:
    """The journal of a finished attempt named ``run``, written in ``directory`` beside its native history.

    The snapshot after each step in ``failed_steps`` failed, so its checkpoint has no snapshot."""
    native = History(directory / "native-home" / f"{SESSION}.jsonl", SESSION, workspace=WORKSPACE)
    with JournalWriter(directory / "trajectory.jsonl", run_id=run) as journal:
        journal.emit(
            "run.started", slot="mini-swe-agent", slot_version="2.4.6", model="fixture", task_prompt=task
        )
        journal.emit("session.ref", native_session_id=SESSION, transcript_path=str(native.path.resolve()))
        journal.emit("rollout.model_tools", shape="chat/completions", tools=BASH_TOOL)
        native.append(_message(role="system", content=SYSTEM))
        native.append(_message(role="user", content=task))
        for number, recorded in enumerate(turns, 1):
            _record_turn(
                journal, native, recorded, turn_id=f"turn-{number}", run=run, failed_steps=failed_steps
            )
        journal.emit("run.result", text=final_text)
        journal.emit("run.finished", status=status, error=None, usage={"prompt_tokens": 11})
    native.close()
    return journal.path


def _record_turn(
    journal: JournalWriter,
    native: History,
    recorded: Turn,
    *,
    turn_id: str,
    run: str,
    failed_steps: Sequence[int],
) -> None:
    first = len(journal.tool_calls()) + 1
    ids = ["call-%d" % step for step in range(first, first + len(recorded.commands))]
    calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
        for call_id, command in zip(ids, recorded.commands, strict=True)
    ]
    native.append(_message(role="assistant", content=recorded.content, tool_calls=calls))
    journal.emit("agent.message", text=recorded.content)
    for call_id, command, output in zip(ids, recorded.commands, recorded.outputs, strict=True):
        journal.emit(
            "tool.started", call_id=call_id, turn_id=turn_id, name="shell", args={"command": command}
        )
        journal.emit(
            "tool.finished", call_id=call_id, turn_id=turn_id, name="shell", status="ok", output=output
        )
        # mini renders an observation as a JSON document.
        observation = json.dumps({"returncode": 0, "output": output})
        native.append(_message(role="tool", tool_call_id=call_id, content=observation))
    step = first + len(ids) - 1
    native.append({"type": "mini.turn", "turn_id": turn_id, "call_ids": ids})
    journal.emit("model.turn.completed", turn_id=turn_id, call_ids=ids, step=step)
    failed = step in failed_steps
    journal.emit(
        "checkpoint.captured",
        step=step,
        turn_id=turn_id,
        snapshot_id=None if failed else "%s-snap-%d" % (run, step),
        session_ckpt=SESSION,
        reason="failed" if failed else "captured",
    )


def _message(**message: Any) -> dict[str, Any]:
    return {"type": "mini.message", "message": message}
