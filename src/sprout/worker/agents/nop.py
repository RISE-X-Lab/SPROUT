"""The ``nop`` agent: does nothing in the sandbox, like Harbor's ``nop`` agent.

A ``nop`` run leaves the sandbox as its image or snapshot made it, so its grade measures
what the task's verifier gives an untouched environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sprout.worker.agents.slot import AgentSlot, SlotResult, TaskSpec
from sprout.worker.run.journal import JournalWriter

if TYPE_CHECKING:
    from sprout.worker.tools.access import ToolAccess


class NopSlot(AgentSlot):
    """Calls no tool and completes at once."""

    name = "nop"

    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess) -> SlotResult:
        return SlotResult(status="completed")
