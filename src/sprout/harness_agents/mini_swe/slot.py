"""Optional mini-swe-agent slot. All effects go through the run's ToolAccess."""

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import threading
from uuid import uuid4

from sprout.harness_core.journal import JournalWriter
from sprout.harness_core.slot import AgentSlot, SlotResult, TaskSpec

MINI_VERSION = "2.4.6"
_IMPORT_LOCK = threading.Lock()


class MiniSweSlot(AgentSlot):
    name = "mini-swe-agent"

    def version(self) -> str | None:
        try:
            return version("mini-swe-agent")
        except PackageNotFoundError:
            return None

    def run(self, task: TaskSpec, journal: JournalWriter, tools) -> SlotResult:
        installed = self.version()
        if installed != MINI_VERSION:
            raise ValueError(
                f'mini-swe-agent {MINI_VERSION} required (installed: {installed}); pip install -e ".[mini]"'
            )
        if tools is None:
            raise ValueError("mini-swe-agent needs the run's ToolAccess")
        if not task.model:
            raise ValueError("mini-swe-agent requires an explicit model")
        home = Path(task.extra.get("native_home") or journal.path.parent / "mini-native")
        home.mkdir(parents=True, exist_ok=True)
        # Upstream loads a user-global .env on first import. Import against an
        # empty attempt-owned directory so a developer's config cannot set RL
        # credentials, model selection or process-global limits.
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
                from sprout.harness_agents.mini_swe.runtime import run
            finally:
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
        return run(task, journal, tools, home, uuid4().hex)
