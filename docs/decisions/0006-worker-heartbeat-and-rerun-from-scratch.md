# ADR-0006: Workers report a heartbeat, and a lost attempt runs again from the start

- **Status:** Accepted
- **Date:** 2026-10-18
- **Revert cost:** the Run Store gets a table of workers, and the job states lose `quarantined`.
  The data of an attempt that runs again is deleted, so a revert cannot resume it.

## Context

Each attempt renews the lease of its own job about once a second. When a lease expires, `expire()`
quarantines the job, and any worker on any host reconciles it. Four other failures quarantine a
job until a person runs `python -m sprout.worker reconcile`:

- a configuration failure, such as a changed profile or a checksum that differs
- a cleanup that the worker cannot confirm
- an exception after the attempt's child started
- a reconcile that failed

Reconcile reads the old attempt's process in `/proc` and its `outcome.json` on the local disk. On
another host, it takes a live process for a gone one and loses the outcome. A shutdown that a
worker notices in its heartbeat loop before its watchdog quarantines the job. We estimated 2.5% to
5% of the running attempts at each shutdown. No job has been quarantined on this host yet: the Run
Store databases hold 135 jobs, all succeeded.

## Decision

We give each worker a stable id, `<host>/<name>`, and a row in a new table `rs_workers`. A worker
registers at start and reports a heartbeat every 30 s. The row records the worker's process. A worker
on the same host takes the row over at once when that process is gone. A worker that stops marks its
row stopped, and the row stays. The heartbeat renews the leases of all the
worker's running jobs to 10 minutes after it. The lease token stays and fences every write.

The Run Store acknowledges each heartbeat that it records. If a worker gets no acknowledgment for 9
minutes, counted from the send time of its last acknowledged heartbeat, it stops all its children.
If an acknowledgment says that the row is no longer the worker's, it stops them at once. A child
also stops when its worker process dies. A worker that reports no heartbeat for 10 minutes is dead,
and its jobs run again. While a worker lives, only it settles its own attempts.

We run a lost attempt again from the start of its job's spec, never from a recovery point. An
attempt is lost when its child raised, when its child left no outcome, or when its worker died. An
infrastructure failure also runs the job again. An actor failure does not. A job runs at most `max_attempts`
times, 3 by default, and then fails with the error of each attempt. The worker deletes the data of
an attempt that runs again: its directory, snapshots, recovery points and events. The attempt's row
stays, with the last 4000 characters of its log. A shutdown stops each child, releases each job
without counting its attempt, and marks the worker's row stopped.

We check the configuration when a worker starts and when a client submits a job, not in the job. The
Run Store freezes the defaults of a job's profile into its spec at submit, and a job carries no
profile hash. A worker claims only the grades whose dataset and parser checksums it has. A cleanup that the worker
cannot confirm does not hold the job: a janitor in each worker removes what the resource ledger
still claims. We remove the `quarantined` state, reconcile across hosts and the `reconcile`
command. A grade in a kept sandbox binds to the stable worker id.

## Alternatives

- **Keep a lease per job and quarantine:** keeps the race at shutdown and the jobs that wait for a
  person.
- **Resume a retry from the last recovery point:** keeps the progress of a lost attempt, but its
  snapshots can be on a lost host, and a person must check each quarantined job.
- **Reconcile on any host:** cannot see a process or a file of another host.

## Consequences

- A crashed worker's jobs wait up to 10 minutes before another worker runs them.
- A cut-off worker stops its children 1 minute before another worker can take their jobs.
- A rollout that runs again loses the progress of its lost attempt. If failures cut long
  trajectories more often, RL sees shorter trajectories than it would.
- The branch points of an attempt that runs again disappear with its snapshots.
- The Run Store HTTP API no longer reports `quarantined`. `docs/RUNSTORE.md` and its job state
  figure change.
- A grade in a kept sandbox settles as `sandbox_lost` with reward 0 when its `keep_until` passes. If
  its host is lost first, it fails as `data_lost`, and its search runs again (ADR-0007).
- A planned reboot releases the running jobs, which run again from the start. It saves no progress
  of a running attempt.
- A grade whose checksums no worker has stays queued until its client's deadline.
- The disk data of a host that never returns stays on that host.

## Links

- [ADR-0004](0004-grade-in-kept-sandbox.md), [ADR-0007](0007-search-affinity-and-host-loss.md).
- `src/sprout/runstore/store.py`, `src/sprout/worker/manager.py`, `src/sprout/worker/attempts.py`,
  `resources/runstore/schema.sql`.
