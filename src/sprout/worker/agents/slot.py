"""The agent slot interface: ``AgentSlot``, ``TaskSpec`` and ``SlotResult``.

The drivers are the other modules of ``sprout.worker.agents``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter
from sprout.worker.sandbox.image_shell import ImageShell

if TYPE_CHECKING:
    from sprout.worker.tools.access import ToolAccess

#: How a run of a slot ended. ``SlotResult`` says what each value means.
SlotStatus = Literal["completed", "error", "timeout", "killed"]


@dataclass
class TaskSpec:
    """What a slot runs. ``extra`` carries the options of one slot, and each slot reads its own keys."""

    prompt: str
    cwd: str
    model: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 3600.0
    extra: dict[str, object] = field(default_factory=dict)
    control: RunControl | None = field(default=None, repr=False)
    #: How the agent's own commands run as the image's user (``worker.sandbox.image_shell``).
    #: None: as the runtime runs every command, as root.
    image_shell: ImageShell | None = None


@dataclass
class SlotResult:
    """How one run of a slot ended.

    ``timeout`` means that the run used up a budget, such as its time or its turns, and can still be
    graded. ``killed`` means that the run was stopped for another reason.
    """

    status: SlotStatus
    final_text: str = ""
    usage: dict[str, object] = field(default_factory=dict)
    native_session_id: str | None = None
    error: str | None = None


class AgentSlot(ABC):
    """Runs one agent on one task, and writes the agent's events to the journal as they occur.

    ``run()`` is synchronous for the caller. An async driver owns its event loop. A killed run must
    still leave a usable trajectory. The slot reaches the sandbox only through ``tools``.
    """

    name: str = "abstract"

    @abstractmethod
    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess) -> SlotResult:
        """Run ``task`` through ``tools``, and return how the run ended.

        The slot decides how to offer the tools to its model. If ``run`` raises, the run ends with
        status ``error``.
        """

    def version(self) -> str | None:
        """The installed version of the upstream agent that the slot drives, or None if there is none."""
        return None
