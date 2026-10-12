# Snapshot cost

Measured with `ops/bench_snapshots.py`: one sandbox from the same DeepSWE image
(2 vCPU, 8 GiB) on each backend, timing the call the checkpointer makes once per
turn, from the client, after each kind of step. Medians of 10 (of 3 for full
snapshots) on an otherwise idle host; both stores sit on the same EBS volume.

| Since the previous snapshot | `podman` | `microvm` |
|---|---|---|
| nothing written | 93 ms | 38 ms |
| 4 KiB written | 96 ms | 41 ms |
| 100 MiB written | 575 ms | 485 ms |
| full snapshot (disk and memory), fresh guest | — (disk-only) | 69 ms |
| full snapshot, after the guest wrote 1 GiB | — | 9.5 s |

A snapshot pays for what was written since the last one, so a turn that only
read costs the floor. Most of podman's floor is a safety check, not the
snapshot: `btrfs subvolume snapshot` takes about 20 ms, and the walk that makes
sure the rootfs holds no nested subvolume (a btrfs snapshot would show it empty)
takes the rest. A full snapshot stores the memory the guest has touched, page
cache included (three in a row after the 1 GiB took 9.5 s each), which is why
per-turn checkpoints are disk-only: measured on one long attempt, 37 MB per
episode disk-only versus 2.2–4.7 GB with memory.

Under load the same call waits for the other sandboxes' writes. Over the
113-task DeepSWE run on podman (up to 80 sandboxes at once on that one volume),
22,077 per-turn snapshots took a median of 153 ms, p90 1.9 s and p99 47 s;
checkpointing was 2.4% of rollout wall time, and 5 of 22,781 captures failed
(recorded as failed, never used as branch points).
