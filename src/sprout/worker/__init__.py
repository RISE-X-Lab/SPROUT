"""Sprout harness core: the mini-swe-agent runtime over the sandbox stack.

Layering:

- ``worker.run``         one run start to finish: runner, journal and events, control, reaping, ATIF export
- ``worker.sandbox``     sandbox session, the microVM and podman backends, the microVM templates
- ``worker.tools``       ToolAccess, the runtime schema, interceptor pipeline, tool results
- ``worker.checkpoint``  checkpointer, branch points
- ``worker.agents``      the slot contract and the mini-swe-agent driver with its native history
- ``worker.gateway``     Chat Completions inference gateway

Design notes live in AGENTS.md. This package must not import benchmark
or method code. What a run's *answer* is stays out of here too: the harness
leaves a snapshot, and the benchmark grades it -- after the fact, re-runnable
when the grader changes.
"""
