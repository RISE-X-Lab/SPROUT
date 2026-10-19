"""The ``harbor`` grade job: the worker runs a Harbor task's verifier itself.

``grade`` is the job. ``overlay`` replays a task's ``tests/Dockerfile`` in a sandbox, and
``sprout.eval`` uses it too. ``files`` moves files in and out of a sandbox. The decision is
``docs/decisions/0003-sprout-runs-harbor-verifier.md``.
"""
