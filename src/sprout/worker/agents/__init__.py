"""Agent slot drivers: what runs in a run's sandbox, chosen by ``RunSpec.agent``.

- ``mini``: the model, in mini-swe-agent (``mini_swe``). The slot imports mini-swe-agent
  lazily, so a missing optional dependency fails only the run that needs it.
- ``harness``: the model, in Sprout's own harness (``harness``): five native tools and a strict
  check of the raw model output (ADR-0008).
- ``oracle``: a Harbor task's ``solution/solve.sh`` (``oracle``).
- ``nop``: nothing (``nop``).

Every driver implements ``slot.AgentSlot``. ``RunSpec.slot`` is ``sprout-harness`` for the harness,
and ``mini-swe-agent`` for each of the other three agents.
"""
