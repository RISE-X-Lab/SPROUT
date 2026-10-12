"""Agent slot drivers.

mini-swe-agent (``mini_swe``) is the only agent: the orchestrator constructs
:class:`~sprout.harness.agents.mini_swe.slot.MiniSweSlot` directly, and every
entry point that accepts a slot name (``harness run``, Run Store, the RL driver)
admits only ``mini-swe-agent``. The slot imports mini-swe-agent itself lazily,
so a missing optional dependency breaks only the run that needs it.
"""
