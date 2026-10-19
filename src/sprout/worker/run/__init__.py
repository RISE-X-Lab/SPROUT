"""The run lifecycle: one run from start to finish, its records, and the reclaim of what dies.

    runner.py     one run: provision, gateway, checkpoints, the slot, and each exit's teardown
    journal.py    the append-only record of a run (its event types are in events.py)
    atif.py       the export of a journal as an ATIF trajectory
    control.py    the out-of-band stop of a run
    rollout.py    the limits of an externally scheduled rollout
    resources.py  what a run holds, recorded before it is used
    reap.py       the reclaim of sandboxes whose run is gone
    sweep.py      the deletion of the snapshots that a journal names

Import each name from its module. The package root imports none of them, because the runner
imports the agent slots, and the slots import ``control``, ``journal`` and ``events``.

Two adjacent concerns are not here. The pool that sandboxes come from is chosen in
``sprout.worker.sandbox.backends``. Per-call machinery, such as the decision whether a call
could change the workspace, runs once for each tool call, so it is on the tool path in
``sprout.worker.tools.interceptors``. There is no registry of live agents, because no code
asks what else runs.
"""
