# ADR-0002: Manual evaluation and the task gate submit jobs to the Run Store

- **Status:** Accepted
- **Date:** 2026-10-16
- **Revert cost:** the output of an evaluation changes from Harbor trial directories to Run Store
  records, and the job spec gains fields (`agent`) that stored jobs keep.

## Context

The RL driver already runs rollouts, grades, reviews and branches as Run Store jobs. Manual
evaluation (`sprout.eval.evaluate`) and the task gate (`sprout.eval.gate`) run local Harbor trials
instead, so the same task runs through two execution paths. The behaviors that this decision
serves:

1. An experiment script evaluates one task with a method and gets each attempt's reward.
2. An experiment script gates a dataset: each task must score 1 with its reference solution
   and 0 when nothing is done.
3. Miles requests scored rollouts through the RL driver (unchanged).

## Decision

We make `eval` a client of the Run Store, like the RL driver. The evaluation client builds the
rollout specs from the task and hands them to the search coordinator. The coordinator submits the
rollout, grade, review and branch jobs, and the client writes the summary. The gate submits two
rollouts for each task: one with `agent=oracle`, which runs `solution/solve.sh`, and one with
`agent=nop`, which runs nothing. Then it grades both. A task passes the gate when the oracle
scores 1 and the nop scores 0.

## Alternatives

- **Keep local Harbor trials for manual evaluation:** keeps two execution paths, so a fix in one
  path does not reach the other.

## Consequences

- `ops/report`, the evaluation summaries and `examples/qwen/run.py` must read Run Store results.
- A worker must run before any evaluation.

## Links

- [ADR-0003](0003-sprout-runs-harbor-verifier.md), [ADR-0004](0004-grade-in-kept-sandbox.md).
