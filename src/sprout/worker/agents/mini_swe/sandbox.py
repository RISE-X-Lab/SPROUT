"""mini's environment in the sandbox: ``resolve_workspace`` and ``ToolEnvironment``.

``resolve_workspace`` fixes mini's working directory before any model action runs.
``ToolEnvironment`` runs each bash action there, as a ``shell`` call through the run's ``ToolAccess``.
"""

from __future__ import annotations

import math
import re
import shlex
import time
from copy import deepcopy
from typing import TYPE_CHECKING, Any, TypedDict

from sprout.contract.mini_history import load_prefix
from sprout.worker.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter
from sprout.worker.sandbox.image_shell import ImageShell
from sprout.worker.tools.access import ToolAccess

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession
    from sprout.worker.tools.result import ToolResult


class Workspace(TypedDict):
    """Where mini works in the sandbox. ``repository_dir`` is ``None`` outside a repository."""

    cwd: str
    repository_dir: str | None


def _stdout(result: ToolResult, purpose: str) -> str:
    """The one absolute path that a probe printed. Raises ``ValueError`` if the probe failed."""
    outcome = result.outcome
    if not result.success or (
        outcome is not None
        and (outcome.exit_code != 0 or outcome.running or outcome.timed_out or outcome.truncated)
    ):
        detail = result.error or (outcome.stderr if outcome is not None else "") or result.output
        raise ValueError(f"mini {purpose} failed: {detail or 'shell returned no diagnostic'}")
    text = outcome.stdout if outcome is not None else result.output
    value = text.strip()
    if not value.startswith("/") or "\n" in value or "\r" in value:
        raise ValueError(f"mini {purpose} returned an invalid absolute path: {value!r}")
    return value


def resolve_workspace(
    session: SandboxSession, extra: dict[str, Any], journal: JournalWriter
) -> dict[str, Any]:
    """A copy of ``extra`` with mini's sandbox working directory resolved, as ``mini_workspace``.

    The directory is a branch's inherited one, else the configured ``cwd``, else the repository root
    of the shell's default directory, else that directory. Raises ``ValueError`` on any conflict.
    """
    configured = extra.get("mini", {}).get("environment", {}).get("cwd")
    inherited: dict[str, Any] | None = None
    if extra.get("native_prefix"):
        entries = load_prefix(extra["native_prefix"])
        metadata = next(entry for entry in entries if entry.get("type") == "mini.session")
        inherited = metadata.get("workspace")
    if inherited and configured not in (None, inherited["cwd"]):
        raise ValueError("mini branch working directory differs from its inherited workspace")
    requested = inherited["cwd"] if inherited else configured
    if requested is not None and (not isinstance(requested, str) or not requested.startswith("/")):
        raise ValueError("mini working directory must be an absolute path or omitted for discovery")
    workspace = probe_workspace(session, requested)
    if inherited is not None and workspace != inherited:
        raise ValueError("mini restored workspace differs from the parent's recorded workspace")
    resolved = deepcopy(extra)
    resolved.setdefault("mini", {}).setdefault("environment", {})["cwd"] = workspace["cwd"]
    resolved["mini_workspace"] = workspace
    journal.emit("mini.workspace", **workspace, inherited=inherited is not None)
    return resolved


def probe_workspace(session: SandboxSession, requested: str | None) -> Workspace:
    """The working directory ``requested``, else the repository root of the shell's default directory,
    else that directory, with the repository that holds it. Raises ``ValueError`` if a probe fails."""
    arguments: dict[str, Any] = {"command": "pwd -P", "timeout": 10}
    if requested is not None:
        arguments["working_dir"] = requested
    cwd = _stdout(session.execute("shell", arguments, timeout=20), "working-directory probe")
    repository = session.execute(
        "shell",
        {
            "command": "git rev-parse --show-toplevel",
            "working_dir": cwd,
            "timeout": 10,
        },
        timeout=20,
    )
    repository_dir = _stdout(repository, "repository probe") if repository.success else None
    if requested is None and repository_dir is not None and cwd != repository_dir:
        cwd = _stdout(
            session.execute(
                "shell",
                {
                    "command": "pwd -P",
                    "working_dir": repository_dir,
                    "timeout": 10,
                },
                timeout=20,
            ),
            "repository working-directory probe",
        )
    return {"cwd": cwd, "repository_dir": repository_dir}


class ToolEnvironment:
    """mini's environment over the run's ``ToolAccess``: ``bash`` is the runtime's ``shell``.

    With ``image_shell``, each action runs as the image's user (``sprout.worker.sandbox.image_shell``).
    ``__init__`` raises ``ValueError`` if ``tools`` has no ``shell(command, working_dir, timeout)``.
    """

    def __init__(
        self,
        tools: ToolAccess,
        journal: JournalWriter,
        deadline: float,
        control: RunControl,
        config: dict[str, Any],
        *,
        image_shell: ImageShell | None,
    ) -> None:
        self.tools, self.journal = tools, journal
        self.deadline, self.control = deadline, control
        self.config = config
        self.image_shell = image_shell
        #: The model turn of the next action. The agent sets it when a turn starts.
        self.turn_id: str | None = None
        #: The repository root that the prompt names. ``None`` outside a repository.
        self.repository_dir: str | None = None
        shell = next((tool for tool in tools.tools if tool["name"] == "shell"), None)
        if shell is None or not {"command", "working_dir", "timeout"} <= set(
            shell["inputSchema"].get("properties", {})
        ):
            raise ValueError("mini requires Sprout shell(command, working_dir, timeout)")

    def execute(self, action: dict[str, Any], cwd: str = "") -> dict[str, Any]:
        """Run one bash ``action`` as a ``shell`` call, and return mini's output dict for it.

        Raises ``RunAborted`` if the run stops. Raises ``ValueError`` for an invalid action, a call
        outside a model turn, or an outcome that is not settled.
        """
        self.control.raise_if_stopped()
        call_id, command = action.get("tool_call_id"), action.get("command")
        if not isinstance(call_id, str) or not call_id or not isinstance(command, str):
            raise ValueError("Invalid mini bash action")
        env = self.config.get("env", {})
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in env):
            raise ValueError("Invalid sandbox environment variable name")
        prefix = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        line = f"env {prefix} bash -c {shlex.quote(command)}"
        working_dir = cwd or self.config["cwd"]
        if self.image_shell is not None:
            line = self.image_shell.command(line, working_dir=working_dir)
        args = {
            "command": line,
            "working_dir": working_dir,
            "timeout": max(
                1,
                min(
                    self.config.get("timeout", DEFAULT_COMMAND_TIMEOUT_S),
                    math.ceil(self.deadline - time.monotonic()),
                ),
            ),
        }
        if self.turn_id is None:
            raise ValueError("tool calls need a turn_id")
        result = self.tools.call("shell", args, call_id=call_id, turn_id=self.turn_id)
        # A rollout limit refuses the call and stops the run. Report that as a stop.
        self.control.raise_if_stopped()
        outcome = result.outcome
        if outcome is None or type(outcome.exit_code) is not int or outcome.running:
            raise ValueError(
                "Sprout shell did not return a settled command outcome: %s"
                % (result.error or result.output)[:500]
            )
        # mini's completion sentinel check and its observation template read one joined output.
        return {
            "output": outcome.stdout + outcome.stderr,
            "returncode": outcome.exit_code,
            "exception_info": "Command timed out" if outcome.timed_out else "",
            "extra": {"timed_out": outcome.timed_out},
        }

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        # No host environment or host uname enters the sandbox prompt.
        return {
            **self.config,
            "repository_dir": self.repository_dir,
            "system": "Linux",
            "release": "",
            "version": "",
            "machine": "",
            **kwargs,
        }

    def serialize(self) -> dict[str, Any]:
        return {"info": {"config": {"environment": self.config, "environment_type": "sprout-tool-access"}}}
