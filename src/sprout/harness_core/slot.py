"""Agent slot contract.

A *slot* runs one agent against one task and streams normalized events into a
journal. Slots are black boxes to the orchestrator: the contract below is the
whole interface. The driver lives in harness_agents/.

Design constraints:
- ``run()`` is synchronous from the caller's perspective; async drivers own
  their own event loop internally.
- Everything a slot learns goes into the journal *as it happens* -- a killed
  run must still leave a usable trajectory (the rule checkpoints follow too:
  ``ToolAccess`` journals each one as it is taken).
- Tools are reached through the run's :class:`~sprout.harness_tools.access.ToolAccess`
  -- the functions an agent may call on its sandbox; the slot decides how to
  offer them to its model and never reaches the sandbox any other way.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from sprout.harness_core.control import RunControl
from sprout.harness_core.journal import JournalWriter


@dataclass
class TaskSpec:
    """What to run. ``extra`` carries slot-specific knobs (documented per slot)."""

    prompt: str
    cwd: str
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 3600.0
    extra: dict[str, object] = field(default_factory=dict)
    control: RunControl | None = field(default=None, repr=False)


@dataclass
class SlotResult:
    status: str  # completed | error | timeout | killed
    final_text: str = ""
    usage: dict[str, object] = field(default_factory=dict)
    native_session_id: str | None = None
    error: str | None = None


class AgentSlot(ABC):
    """Contract every driver implements."""

    name: str = "abstract"

    @abstractmethod
    def run(self, task: TaskSpec, journal: JournalWriter, tools: Any) -> SlotResult:
        """Execute the task through ``tools``, streaming events into ``journal``."""

    def version(self) -> str | None:
        """Upstream version string, for run.started provenance."""
        return None
