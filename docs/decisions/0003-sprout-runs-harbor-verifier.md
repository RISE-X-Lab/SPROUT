# ADR-0003: Sprout reads Harbor tasks and runs their verifier itself

- **Status:** Accepted
- **Date:** 2026-10-16
- **Revert cost:** the rewards of reported benchmarks (DeepSWE, TerminalBench 2.1, HARD-51) come
  from a different implementation of the verifier, and the dependency on `harbor==0.22.0` goes.

## Context

Harbor owns a trial: it loads the task, starts the environment, runs the agent, runs the verifier
and writes `result.json`. Sprout provides the environment and the agent through seven adapters in
`eval/harbor/`. The sandbox, the agent, the checkpoints and the branches are Sprout's own. Harbor
adds the task format, the verifier run and the result layout.

## Decision

We read the Harbor task format (`task.toml`, `instruction.md`, `environment/`, `tests/`,
`solution/`) with a reader in `contract`. A worker grade job with benchmark `harbor` runs
`tests/test.sh` with the task's verifier timeout and network policy, and reads the reward from
`/logs/verifier/reward.txt` or `reward.json`. Before we delete the adapters and the dependency, we
check that our rewards equal Harbor's: the gate on every task of the three datasets, and a
re-grade of recorded snapshots against the recorded Harbor rewards.

## Alternatives

- **Run each Harbor trial as one worker job:** keeps Harbor in control of the lifecycle and keeps
  two ways to run an agent.
- **Keep Sprout usable inside Harbor:** no current user needs it. It can return as a thin adapter.

## Consequences

- A new benchmark in the Harbor format needs only the reader to support its `schema_version`.
- The equivalence check costs a gate run and a re-grade of each dataset.

## Links

- [ADR-0002](0002-eval-and-gate-through-run-store.md), [ADR-0004](0004-grade-in-kept-sandbox.md). `docs/HARBOR_PLUGIN.md` describes the adapters that this decision removes.
