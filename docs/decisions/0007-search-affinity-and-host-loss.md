# ADR-0007: The jobs of one search run on one host, and a lost host runs the search again

- **Status:** Accepted
- **Date:** 2026-10-18
- **Revert cost:** each job records an `affinity` key, the Run Store gets a table of key bindings,
  and jobs that a lost host held end as `data_lost`. A revert leaves those records.

## Context

A rollout leaves its data on the host that ran it: the snapshots in the btrfs store, the attempt
directory, and a kept sandbox. The later jobs of a search read that data:

- A branch starts from a snapshot of its parent.
- A review reads the attempt directories of every failed root of its group.
- A grade reads the rollout's data. Of the datasets on this host, TerminalBench 2.1 (89 tasks) and
  SWE-bench Pro HARD-51 (51 tasks) grade in the kept sandbox. DeepSWE (113 tasks) collects the
  patch from the final snapshot. SWE-rebench V2 takes the diff of the final snapshot.

On another host, none of this data exists. A group has more than one root, so a rule that sends a
job to the host of its parent cannot place a review of two roots that ran on two hosts. Manual
evaluation splits one task's search into coordinator groups that every evaluation names alike:
`parent`, `review-1`, `round-1`.

## Decision

We add the field `affinity` to each job spec. Each job of one search carries one key. The RL driver
uses the id of the Miles group. Manual evaluation makes a key for each task evaluation, keeps it in
the evaluation's output, and gives it to each of the task's groups. The first worker that claims a job of a key binds the key to its host, in a new
table `rs_affinity`. After that, only a worker on that host claims a job of the key, and it claims
these jobs before the jobs without a key. A job that names another job and has no key takes the key
of that job.

A host is lost when no worker on it has reported a heartbeat for 10 minutes, whether its workers
died or stopped. The Run Store then fails each unfinished job of each key bound to that host, with
the reason `data_lost`. A grade in a kept sandbox on that host fails the same way. The Run Store keeps
every job, attempt and result. The client then runs the search again from its roots, under the key
`<key>-r<n>`, and keeps the records of the earlier run: the coordinator runs a Miles group again, and
manual evaluation runs the task evaluation again. A search runs again at most 2 times, and then it
fails. The RL driver exports nothing to Miles for a failed group.

## Alternatives

- **Each job follows the host of its parent:** cannot place a review that reads two roots.
- **One key for each coordinator group:** unrelated evaluations share the key `parent`, and the
  review of a task gets another key than its parent.
- **The Run Store runs the chain again:** the Run Store would have to know which roots, points and
  branches make a group. The coordinator owns that structure.
- **Storage that every host reads, such as S3 or NFS:** removes the problem, but needs its own
  project.

## Consequences

- All the jobs of a search run on one host, so the load spreads only by search.
- A search that runs again after a lost host loses all its work.
- A host that reboots for more than 10 minutes is lost, and its searches run again. Its kept
  sandboxes do not survive a reboot of any length.
- The microVM backend follows the same rule, though AgentENV keeps its snapshots off the host,
  because a review reads the attempt directories on the host.
- The coordinator runs an RL group again, and `evaluate.py` runs a task evaluation again.

## Links

- [ADR-0004](0004-grade-in-kept-sandbox.md), [ADR-0006](0006-worker-heartbeat-and-rerun-from-scratch.md).
- `src/sprout/search/coordinator.py`, `src/sprout/runstore/store.py`, `src/sprout/runstore/specs.py`.
