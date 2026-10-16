# ADR-0001: The top-level packages follow the runtime components

- **Status:** Accepted
- **Date:** 2026-10-16
- **Revert cost:** later changes build on the package boundaries, and the entry points changed
  for every host that runs them (`python -m sprout.worker start` replaces
  `python -m sprout.runstore worker` in each `~/.config/sprout/services.yaml`).

## Context

Sprout runs as separate processes: Miles talks to the RL driver, the RL driver submits jobs to
the Run Store, the Run Store keeps them in PostgreSQL, and a worker on each machine claims jobs
and runs the agent, the sandbox and the grader. The old layout did not show this. `harness/` held
seven sub-packages, `runstore/` held the worker's code, and `methods/` mixed the review job with
the local evaluation. The Run Store and the RL driver imported harness modules only because the
request formats were defined inside them.

## Decision

We give each runtime component one top-level package under `src/sprout/`:

- `contract`: the formats that the Run Store, the RL driver and the worker share. It imports no
  other package.
- `runstore`: the job records, the HTTP API and the client. It imports only `contract`.
- `worker`: the claim loop, attempt supervision and the jobs (rollout, grade, review, export),
  with the agent runtime in `worker/run`, `worker/agents`, `worker/sandbox`, `worker/tools`,
  `worker/checkpoint` and `worker/gateway`.
- `rl_driver`: the Miles adapter. It imports only `contract` and `runstore`.
- `eval`: manual evaluation. `services`: the host's long-running processes.

The read-only run viewer is `ops/report`, outside the package. `tests/harness/test_layering.py`
enforces the import direction.

## Alternatives

- **One package per layer with `<layer>_<part>` siblings:** gives about 30 top-level directories
  whose grouping is visible only in the names.
- **Groups of packages, as in DeepSeek Harness (`packages/<group>/<pkg>/`):** groups by capability
  family, not by the processes that Sprout deploys.

## Consequences

- A change can see from the path which process it affects.
- `src/` keeps at most 1000 lines per file, 10 files per directory and 5 directory levels
  (`gate/checks/src_layout.py`).
- Recorded Harbor trial configs under `~/sprout-runs` name `sprout.harbor_plugin.*`. A re-run
  from such a config fails to import.

## Links

- Branch `restructure-components`. `README.md` (Layout).
