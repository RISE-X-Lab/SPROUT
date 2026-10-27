"""Resolve the sandbox checkout and the historical protocol's workspace alias."""

from __future__ import annotations

import shlex
from typing import Any

from sprout.worker.agents.mini_swe.sandbox import resolve_workspace as resolve_native_workspace
from sprout.worker.run.journal import JournalWriter
from sprout.worker.sandbox.session import SandboxSession


def resolve_workspace(
    session: SandboxSession, extra: dict[str, Any], journal: JournalWriter
) -> dict[str, Any]:
    """Resolve a checkout; compatibility tools additionally expose it at ``/workspace``."""
    resolved = resolve_native_workspace(session, extra, journal)
    if resolved.get("opencollab_tool_protocol", "single2") != "review50-v1":
        return resolved
    target = resolved["mini_workspace"]["cwd"]
    command = (
        "if [ -e /workspace ] || [ -L /workspace ]; then "
        f'test "$(cd /workspace && pwd -P)" = {shlex.quote(target)}; '
        f"else ln -s -- {shlex.quote(target)} /workspace; fi"
    )
    result = session.execute("shell", {"command": command, "working_dir": target, "timeout": 10}, timeout=20)
    if not result.success:
        raise ValueError(
            "OpenCollab /workspace alias is unavailable or points outside the configured checkout"
        )
    journal.emit("opencollab.workspace_alias", alias="/workspace", cwd=target)
    return resolved
