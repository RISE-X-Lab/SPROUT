# ADR-0004: A verifier that shares the agent's environment grades in the kept sandbox

- **Status:** Accepted
- **Date:** 2026-10-16
- **Revert cost:** the Run Store schema gains the bound worker and the keep deadline of a job, and
  the scoring rule "a lost sandbox scores 0" applies to reported TerminalBench 2.1 results.

## Context

Some TerminalBench 2.1 verifiers need the processes that the agent started. Today a memory
snapshot keeps them for the grade. The Run Store refuses `keep_sandbox` ("Queued runs retain
snapshots, not live sandboxes"), any worker can claim any job, and a kept sandbox exists only on
the machine that ran the rollout.

## Decision

When the task's verifier shares the agent's environment, we do these steps:

1. The rollout job keeps its sandbox. The worker takes the final disk snapshot first, then keeps
   the sandbox for at most 20 minutes.
2. The Run Store records the kept sandbox and the worker that holds it.
3. The grade job names the rollout. The Run Store binds it to that worker, and only that worker
   claims it.
4. The worker runs the verifier in the kept sandbox, then destroys the sandbox.

If the sandbox is gone, or the 20 minutes pass before the grade runs, the grade records reward 0
with the reason `sandbox_lost`. The reaper removes a kept sandbox that has no grade.

## Alternatives

- **Grade in the same attempt as the rollout:** needs no binding, but joins the rollout and the
  grade, which stay separate jobs.
- **Restore a memory snapshot for the grade:** works only on microVMs, because podman snapshots
  are disk-only.

## Consequences

- A re-grade of such a task from its disk snapshot cannot restore its processes.
- A branch from a failed attempt restores the disk only.

## Links

- [ADR-0002](0002-eval-and-gate-through-run-store.md), [ADR-0003](0003-sprout-runs-harbor-verifier.md).
