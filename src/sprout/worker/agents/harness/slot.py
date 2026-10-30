"""The ``sprout-harness`` slot: ``HarnessSlot`` runs a ``Session``, and ``resolve_workspace`` fixes its cwd.

The slot needs no optional dependency. It reaches the model through ``OPENAI_BASE_URL`` and
``OPENAI_API_KEY`` from ``task.env`` (the run's gateway), else from the process environment.
"""

from __future__ import annotations

import os
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from sprout.contract.harness_history import SLOT
from sprout.worker.agents.harness.config import HarnessConfig
from sprout.worker.agents.harness.history import History
from sprout.worker.agents.harness.loop import Session
from sprout.worker.agents.harness.model import ModelClient, chat_url
from sprout.worker.agents.harness.prompts import system_message, task_message
from sprout.worker.agents.harness.tools import tool_schemas
from sprout.worker.agents.slot import AgentSlot, SlotResult, SlotStatus, TaskSpec
from sprout.worker.run.control import RunAborted, RunControl, ended_by_stop_condition
from sprout.worker.run.events import RUN_RESULT
from sprout.worker.run.journal import JournalWriter

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession
    from sprout.worker.tools.access import ToolAccess

#: The version of the harness's prompts, tools and format check, as ``run.started`` records it.
HARNESS_VERSION = "1"


class HarnessSlot(AgentSlot):
    """Runs the harness on a task through the run's ``ToolAccess``."""

    name = SLOT

    def version(self) -> str | None:
        return HARNESS_VERSION

    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess | None) -> SlotResult:
        """Run ``task`` and return how it ended.

        Raises ``ValueError`` before the run starts for bad options, no ``tools``, no model, no
        workspace, or a branch, which the harness cannot restore yet."""
        config = HarnessConfig.from_extra(task.extra)
        if tools is None:
            raise ValueError("the harness needs the run's ToolAccess")
        if not task.model:
            raise ValueError("the harness requires an explicit model")
        if task.extra.get("native_prefix") or task.extra.get("assistant_turn"):
            raise ValueError("the harness cannot continue a branch point yet")
        workspace = cast(dict[str, Any] | None, task.extra.get("harness_workspace"))
        if not workspace or not str(workspace.get("cwd", "")).startswith("/"):
            raise ValueError("the harness requires a resolved sandbox workspace")
        env = {**os.environ, **task.env}
        if not env.get("OPENAI_BASE_URL"):
            raise ValueError("Set OPENAI_BASE_URL or configure the Sprout model gateway")
        session_id = uuid4().hex
        home = Path(cast(str | None, task.extra.get("native_home")) or journal.path.parent / "harness-native")
        schemas = tool_schemas(config.toolset)
        history = History(home / f"{session_id}.jsonl", session_id, workspace=workspace, tools=schemas)
        journal.emit(
            "run.started",
            slot=SLOT,
            slot_version=HARNESS_VERSION,
            model=task.model,
            task_prompt=task.prompt,
            cwd=task.cwd,
            config=asdict(config),
        )
        journal.emit("session.ref", native_session_id=session_id, transcript_path=str(history.path.resolve()))
        journal.emit("rollout.model_tools", shape="chat/completions", tools=schemas)
        control = task.control or RunControl()
        deadline = time.monotonic() + task.timeout_s
        timer = threading.Timer(
            task.timeout_s,
            lambda: control.request_stop("harness wall-time budget exhausted", stop_reason="timeout"),
        )
        timer.daemon = True
        timer.start()
        client = ModelClient(
            url=chat_url(env["OPENAI_BASE_URL"]),
            key=env.get("OPENAI_API_KEY", ""),
            model=task.model,
            journal=journal,
            control=control,
            deadline=deadline,
            parameters=config.parameters,
            logprobs=config.logprobs,
        )
        names = frozenset(t["function"]["name"] for t in schemas)
        messages = [
            {
                "role": "system",
                "content": system_message(
                    cwd=workspace["cwd"], repository_dir=workspace.get("repository_dir"), tools=names
                ),
            },
            {"role": "user", "content": task_message(task.prompt, names)},
        ]
        session = Session(
            journal=journal,
            tools=tools,
            client=client,
            history=history,
            config=config,
            control=control,
            schemas=schemas,
            messages=messages,
            cwd=workspace["cwd"],
            deadline=deadline,
            image_shell=task.image_shell,
        )
        status: SlotStatus = "error"
        error: str | None = None
        try:
            session.run()
            status = "completed"
        except RunAborted as exc:
            # A stop condition (a budget, or an invalid turn) leaves a rollout with a known end.
            status = "timeout" if ended_by_stop_condition(control.stop_reason) else "killed"
            error = str(exc) or control.reason
        except (TimeoutError, KeyboardInterrupt) as exc:
            # The attempt process turns its termination signal into KeyboardInterrupt.
            control.request_stop("harness wall-time budget exhausted", stop_reason="timeout")
            status, error = "timeout", str(exc) or control.reason
        except Exception as exc:  # noqa: BLE001 - the run reports how it ended
            status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            timer.cancel()
            history.close()
        usage: dict[str, object] = dict(client.usage.as_dict())
        journal.emit(RUN_RESULT, text="")
        journal.emit("run.finished", status=status, error=error, usage=usage, stop_reason=control.stop_reason)
        return SlotResult(
            status=status, final_text="", usage=usage, native_session_id=session_id, error=error
        )


def resolve_workspace(
    session: SandboxSession, extra: dict[str, Any], journal: JournalWriter
) -> dict[str, Any]:
    """A copy of ``extra`` with the harness's sandbox working directory, as ``harness_workspace``.

    The directory is ``extra.harness.cwd``, else the repository root of the shell's default
    directory, else that directory. Raises ``ValueError`` if a probe fails."""
    from sprout.worker.agents.mini_swe.sandbox import probe_workspace

    config = HarnessConfig.from_extra(extra)
    workspace = probe_workspace(session, config.cwd)
    resolved = deepcopy(extra)
    resolved["harness_workspace"] = dict(workspace)
    journal.emit("harness.workspace", **workspace)
    return resolved
