"""The Run Store worker, and the stack that runs one agent in one sandbox on this host.

``python -m sprout.worker start`` claims Run Store jobs and runs each attempt in a child process
(``worker.manager``, ``worker.attempts``). ``worker.cleanup`` deletes the data of a lost attempt and
frees the sandboxes that attempts left. docs/RUNSTORE.md describes the worker. The subpackages:

- ``worker.jobs``: the jobs that an attempt runs, which are a rollout, a grade, a review and the export
- ``worker.run``: one run from start to finish, its journal and events, its control, reaping, ATIF export
- ``worker.sandbox``: the sandbox session, the microVM and podman backends, the microVM templates
- ``worker.tools``: ``ToolAccess``, the runtime schema, the interceptor pipeline, the tool results
- ``worker.checkpoint``: the checkpoints and the branch points
- ``worker.agents``: the slot interface and its drivers, which are mini-swe-agent, oracle and nop
- ``worker.gateway``: the Chat Completions inference gateway

No module of this package imports ``sprout.eval``, ``sprout.search``, ``sprout.rl_driver`` or
``sprout.services``. A run does not grade its own work. A separate grade job grades it, from the
run's final disk snapshot or in the sandbox that the run kept.
"""
