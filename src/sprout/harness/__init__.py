"""Sprout harness core: the mini-swe-agent runtime over the sandbox stack.

Layering:

- ``harness_core``          event model v2, journal store, slot contract, control
- ``harness_sandbox``       sandbox session, the microVM and podman backends, the microVM templates
- ``harness_tools``         ToolAccess, the runtime schema, interceptor pipeline
- ``harness_checkpoint``    checkpointer, branch points
- ``harness_agents``        the mini-swe-agent driver and its native history
- ``harness_gateway``       Chat Completions inference gateway
- ``harness_orchestrator``  the shape of one run
- ``harness.atif``          journal -> ATIF v1.8 export

Design notes live in AGENTS.md. This package must not import benchmark
or method code. What a run's *answer* is stays out of here too: the harness
leaves a snapshot, and the benchmark grades it -- after the fact, re-runnable
when the grader changes.
"""
