"""Optional mini-swe-agent slot. All effects go through the run's ToolAccess."""

from __future__ import annotations

import os
import threading
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from sprout.worker.agents.slot import AgentSlot, SlotResult, TaskSpec
from sprout.worker.run.journal import JournalWriter

if TYPE_CHECKING:
    from sprout.worker.tools.access import ToolAccess

#: The one mini-swe-agent version that the slot runs.
MINI_VERSION = "2.4.6"
# One thread at a time sets the process environment around mini's first import.
_IMPORT_LOCK = threading.Lock()


class MiniSweSlot(AgentSlot):
    """The slot that runs mini-swe-agent ``MINI_VERSION`` on a task."""

    name = "mini-swe-agent"

    def version(self) -> str | None:
        try:
            return version("mini-swe-agent")
        except PackageNotFoundError:
            return None

    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess | None) -> SlotResult:
        """Run ``task`` with mini through ``tools``, and return its result.

        Raises ``ValueError`` if mini ``MINI_VERSION`` is not installed, ``tools`` is ``None`` or
        ``task`` names no model.
        """
        installed = self.version()
        if installed != MINI_VERSION:
            raise ValueError(
                f'mini-swe-agent {MINI_VERSION} required (installed: {installed}); pip install -e ".[mini]"'
            )
        if tools is None:
            raise ValueError("mini-swe-agent needs the run's ToolAccess")
        if not task.model:
            raise ValueError("mini-swe-agent requires an explicit model")
        # native_home is worker-owned, a path string. sprout.runstore.specs refuses it in a job spec.
        native_home = cast(str | None, task.extra.get("native_home"))
        home = Path(native_home or journal.path.parent / "mini-native")
        home.mkdir(parents=True, exist_ok=True)
        # mini loads a user-global .env at its first import. An empty attempt-owned config directory
        # keeps a developer's credentials, model choice and process-global limits out of the run.
        with _IMPORT_LOCK:
            settings = {
                "MSWEA_GLOBAL_CONFIG_DIR": str(home / "config"),
                "MSWEA_SILENT_STARTUP": "1",
                "MSWEA_GLOBAL_COST_LIMIT": "0",
                "MSWEA_GLOBAL_CALL_LIMIT": "0",
            }
            previous = {key: os.environ.get(key) for key in settings}
            try:
                os.environ.update(settings)
                from sprout.worker.agents.mini_swe.runtime import run
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return run(task, journal, tools, home, uuid4().hex)
