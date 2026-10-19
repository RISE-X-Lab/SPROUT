"""Agent slot drivers: what runs in a run's sandbox, chosen by ``RunSpec.agent``.

- ``mini``: the model, in mini-swe-agent (``mini_swe``). The slot imports mini-swe-agent
  lazily, so a missing optional dependency fails only the run that needs it.
- ``oracle``: a Harbor task's ``solution/solve.sh`` (``oracle``).
- ``nop``: nothing (``nop``).

Every driver implements ``slot.AgentSlot``. ``RunSpec.slot`` must be ``mini-swe-agent`` for each
of the three agents.
"""
