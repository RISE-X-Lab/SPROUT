"""Run lifecycle: start one, run many, reclaim what dies.

    run.py        one run start to finish -- provision, gateway, checkpoints,
                  drive the slot, tear down whichever way it ends
    resources.py  what a run holds, written before it is used
    reap.py       reclaim sandboxes whose run is gone

This is the *run* granularity, and the boundary is worth stating because two
things that look adjacent are deliberately not here:

- ``harness_sandbox/backends.py`` -- which pool sandboxes come from. The
  orchestrator is one of its callers, not its owner.
- ``harness_tools/interceptors/mutation.py`` -- an interceptor, so it runs
  per *tool call*. Putting per-call machinery in a per-run component would be a
  layer mistake. What lives here is the decision to snapshot; what decides whether
  a call could have changed anything is on the tool path.

There is no registry of live agents. Nothing yet asks "what else is running", and
a state machine with one state is worse than none -- when subagent spawning
arrives, this is the package it goes in, and `run.py`'s sequence is already what a
child needs.
"""

from sprout.harness_orchestrator.reap import AgentEnvClient, PodmanClient, Reaper, parse_duration
from sprout.harness_orchestrator.resources import Resource, ResourceLedger
from sprout.harness_orchestrator.run import Orchestrator, RunOutcome, RunSpec

__all__ = [
    "Orchestrator",
    "RunSpec",
    "RunOutcome",
    "ResourceLedger",
    "Resource",
    "Reaper",
    "AgentEnvClient",
    "PodmanClient",
    "parse_duration",
]
