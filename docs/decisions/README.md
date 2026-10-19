# Decisions

The architecture decisions that one revert cannot undo, and the significant ones that the team
chose to record. Each file is one decision. Nobody edits an accepted decision. A later decision
supersedes it.

| ADR | Decision | Status | Date |
|---|---|---|---|
| [0001](0001-runtime-component-packages.md) | The top-level packages follow the runtime components | Accepted | 2026-10-16 |
| [0002](0002-eval-and-gate-through-run-store.md) | Manual evaluation and the task gate submit jobs to the Run Store | Accepted | 2026-10-16 |
| [0003](0003-sprout-runs-harbor-verifier.md) | Sprout reads Harbor tasks and runs their verifier itself | Accepted | 2026-10-16 |
| [0004](0004-grade-in-kept-sandbox.md) | A verifier that shares the agent's environment grades in the kept sandbox | Accepted | 2026-10-16 |
| [0005](0005-eval-runs-as-the-image-user.md) | Manual evaluation runs as the image's user, and no run has a tool primer | Accepted | 2026-10-17 |
| [0006](0006-worker-heartbeat-and-rerun-from-scratch.md) | Workers report a heartbeat, and a lost attempt runs again from the start | Accepted | 2026-10-18 |
| [0007](0007-search-affinity-and-host-loss.md) | The jobs of one search run on one host, and a lost host runs the search again | Accepted | 2026-10-18 |
