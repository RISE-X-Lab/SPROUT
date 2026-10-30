"""The options of one harness run, from ``RunSpec.extra["harness"]``: ``HarnessConfig``.

``on_invalid`` decides what a rejected turn does. ``feedback`` (evaluation) tells the model what was
wrong and lets it answer again. ``terminate`` (training) ends the rollout, as Arctic-SWE's training
harness does, with the stop reason of the flaw: the RL driver scores such a rollout -1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sprout.worker.agents.harness.tools import TOOLSETS

#: The sampling parameters that a run may set for its requests. The rollout contract's win at the gateway.
PARAMETERS = frozenset({"temperature", "top_p", "top_k", "max_tokens", "stop", "reasoning_effort", "seed"})
#: How ``on_invalid`` can handle a rejected turn.
MODES = ("feedback", "terminate")


@dataclass(frozen=True)
class HarnessConfig:
    """Validated harness options. ``ValueError`` from ``from_extra`` names the first bad one."""

    toolset: str = "default"
    on_invalid: str = "feedback"
    #: The prompt plus completion tokens of the latest request at which the rollout stops, 0 for none.
    max_total_tokens: int = 131_072
    #: The repeats of one call since the last successful edit that end the rollout, 0 for never.
    repetition_limit: int = 0
    #: Whether a turn that parses but differs from its canonical text is rejected, not only recorded.
    strict_roundtrip: bool = False
    #: Whether requests ask for log-probs, the source of the raw text that the format check reads.
    logprobs: bool = True
    #: The sandbox working directory. None finds it: the repository root, else the shell's directory.
    cwd: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_extra(cls, extra: Mapping[str, Any]) -> HarnessConfig:
        """The options in ``extra["harness"]``, with defaults for the rest."""
        options = extra.get("harness", {})
        if not isinstance(options, Mapping):
            raise ValueError("RunSpec.extra.harness must be an object")
        unknown = set(options) - {name for name in cls.__dataclass_fields__}
        if unknown:
            raise ValueError("Unknown harness options: %s" % ", ".join(sorted(unknown)))
        config = cls(**dict(options))
        config._validate()
        return config

    def _validate(self) -> None:
        if self.toolset not in TOOLSETS:
            raise ValueError("harness toolset must be one of %s" % ", ".join(TOOLSETS))
        if self.on_invalid not in MODES:
            raise ValueError("harness on_invalid must be feedback or terminate")
        for name in ("max_total_tokens", "repetition_limit"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("harness %s must be a non-negative integer" % name)
        for name in ("strict_roundtrip", "logprobs"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("harness %s must be a boolean" % name)
        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd.startswith("/")):
            raise ValueError("harness cwd must be an absolute path")
        if not isinstance(self.parameters, dict) or set(self.parameters) - PARAMETERS:
            raise ValueError("harness parameters may set only %s" % ", ".join(sorted(PARAMETERS)))
