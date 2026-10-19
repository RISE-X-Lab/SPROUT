# ADR-0005: Manual evaluation runs as the image's user, and no run has a tool primer

- **Status:** Accepted
- **Date:** 2026-10-17
- **Revert cost:** the decision fixes the protocol of reported experiments: the actor's prompt and
  the user that the agent's commands, `solve.sh` and `test.sh` run as. Runs made under one protocol
  cannot be repeated for free under the other.

## Context

Before [ADR-0002](0002-eval-and-gate-through-run-store.md), manual evaluation ran local Harbor
trials, and Harbor differed from Run Store rollouts in two ways:

- Harbor's agent adapter appended a tool primer to mini-swe-agent's instruction: the `bash` tool
  definition, the rules of a call, the default working directory and an example.
- Harbor ran the agent's commands, `solution/solve.sh` and `tests/test.sh` as the image's `USER`
  with the image's `ENV`. Run Store rollouts ran them as root.

The reported DeepSWE, TerminalBench 2.1 and HARD-51 numbers come from the Harbor protocol. Mini's
model request already declares the `bash` tool in its `tools` field, so the primer repeats the
definition that the model gets.

## Decision

We give mini the task's instruction as it is, with no tool primer, on every run.

We make the image's identity two settings of the backend:

- `image_user`: commands run as the image's `USER`, or as root when the image names none.
- `image_env`: commands run with the image's `ENV`.

Both default to off, so RL rollouts keep their behavior. Manual evaluation and the task gate turn
both on. `--no-image-user` and `--no-image-env` turn them off, and `summary.json` records them.
`examples/qwen/run.py` passes both to each stage, and a branch method must share them with its
baseline. A baseline manifest that predates the two keys ran Harbor trials, so it counts as both on.

## Alternatives

- **Keep the primer as a setting of a run, on for evaluation:** keeps the comparison with the
  reported numbers, but sends the tool definition twice and keeps one more protocol difference.
- **One protocol for evaluation and RL, as root:** breaks the comparison with the reported numbers
  on the user that commands run as.
- **The Harbor user and environment for RL too:** changes the sandbox user of RL in the middle of
  the research.

## Consequences

- A new evaluation and the reported numbers differ in the actor's first prompt. Compare them only
  with that difference stated.
- A branch from a parent that ran with the primer keeps it, because the branch restores the
  parent's history.
- An evaluation and an RL rollout of the same task can differ in their user and environment.
- `[[verifier.collect]]` commands, the preparation of `/logs/verifier` and the `RUN` steps of a
  verifier overlay still run as root. Harbor runs the collect commands as the image user.

## Links

- [ADR-0002](0002-eval-and-gate-through-run-store.md),
  [ADR-0003](0003-sprout-runs-harbor-verifier.md).
- Commits `e61de77` and `7be2362` (the image's user and environment), `e8be9f0` (no primer) and
  `289c076` (shared with the baseline) on branch `eval-through-runstore`.
- `src/sprout/worker/sandbox/image_shell.py`, `examples/qwen/run.py`.
