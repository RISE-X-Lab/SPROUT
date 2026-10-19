# Architecture

Sprout makes each turn of a rollout a point that a new rollout can start from. Evaluation and RL
training both build on this one ability, through the same Run Store jobs. Read this document
before you change the code. It has these parts:

- [Checkpoint every turn](#checkpoint-every-turn): how a run takes its checkpoints, and on which
  backend
- [Branch from where it went wrong](#branch-from-where-it-went-wrong): the steps of a SPROUT search
- [Processes](#processes): the Run Store, its clients and the claim loop of a worker

## Checkpoint every turn

Sprout checkpoints each turn of a rollout, so that a branch can start after any turn. A
checkpoint pairs a disk snapshot of the sandbox with the agent's history up to that turn. A run
that starts from the snapshot with that history continues after that turn. It has the same past,
and it can take a different future. The Run Store keeps each checkpoint of a rollout job as a
recovery point.

The `Runner` owns each run. The agent reaches the sandbox only through `ToolAccess`, which makes
the checkpoints. A run makes them in this order:

1. The `Runner` starts the sandbox.
2. The `Runner` gives the agent a `ToolAccess`: the functions that the agent can call on the
   sandbox. mini-swe-agent offers its model one `bash` tool, which runs as the runtime's `shell`.
3. When the agent has recorded each observation of a turn, it closes the turn.
4. Before the first tool call of the next turn, `ToolAccess` asks the `Checkpointer` for the
   checkpoint of the closed turn. Only a model request ran in between, so the disk is as the
   closed turn left it.
5. If a call of the closed turn can change the disk, the `Checkpointer` takes a new snapshot. If
   no call can, it reuses the last snapshot.
6. `ToolAccess` writes the checkpoint to the journal as a `checkpoint.captured` record.
7. At the end of the run, `ToolAccess` takes one final disk snapshot, the checkpoint of the last
   turn. The `Runner` does not destroy the sandbox before this snapshot exists.

If a capture fails, the `Checkpointer` reuses the last snapshot, and the record gives the reason
`failed`. A turn that makes no tool call gets no checkpoint. If the outcome of a tool call is
unknown, for example after a transport failure, `ToolAccess` stops the rollout. No later state
of the sandbox pairs with a known disk.

Each `shell` call counts as a change, because the text of a command does not show its effect:
`ls` reads, and `ls > out` writes. A wrong "no change" loses the state of a turn. A wrong
"change" costs one capture. [Runtime toolset support](../README.md#runtime-toolset-support) gives the rule
for each tool.

Sprout runs the sandbox on one of two backends:

| Backend | Sandbox | Snapshot |
| --- | --- | --- |
| `podman`, the default | A rootless container whose root filesystem is a btrfs subvolume. Stock podman 6 runs it, with no daemon. Each sandbox has a network of its own, so a run can switch the egress of a running sandbox. | A read-only btrfs snapshot of the subvolume. The container keeps running. The snapshot holds the disk only. |
| `microvm`, with `--backend microvm` or in a profile's `backend` | A Firecracker microVM that [AgentENV](https://github.com/dreamyang-liu/AgentENV) serves. | One layer on the chain of the VM disk. The server compacts the chain. A memory snapshot is also possible, for a grader that needs the processes of the VM. |

A branch runs on the backend of its parent. On podman, it also runs in the store of its parent,
because the snapshots of the parent are in that store.

Per-turn snapshots are disk-only, because a snapshot with memory is about 60 to 130 times larger.
In a 113-task DeepSWE run on podman, with up to 80 sandboxes at once, 22,077 per-turn snapshots
took a median of 153 ms. Checkpointing took 2.4% of the rollout wall time. On one long attempt,
disk-only snapshots used 37 MB per episode. With memory, they used 2.2–4.7 GB per episode.

### Branch by hand

`python -m sprout.worker run` refuses a journal in a directory that the operating system empties
at a reboot, for example `/tmp`. The journal is the only record of the run. To accept such a
path, add `--volatile-ok`. The `--cwd` of the run names a directory inside the sandbox, so the
rule does not apply to it.

Each snapshot id that `python -m ops.report snapshots` prints is also a sandbox image. To branch
from a snapshot by hand, run `python -m sprout.worker run` again with these two changes:

- `--sandbox-image <snapshot id>`
- `--extra`, with a `native_prefix` that names the agent's native history up to that turn. The
  `native_prefix` has these keys:
  - `path`: the history file
  - `sha256`: the checksum of the prefix
  - `byte_length`: the length of the prefix in bytes. Without it, the prefix is the whole file.

Through the Run Store, a client branches a rollout job from one of its recovery points
([Run Store](RUNSTORE.md)).

## Branch from where it went wrong

SPROUT branches a failed rollout after the steps that it got right, instead of sampling again
from the start. A failed rollout rarely goes wrong at its first step, and the agent usually
believes that it succeeded. A new rollout from the start repeats each step that the failed one
got right.

A SPROUT search grows a tree of rollouts from one failed parent. Each round reviews every
attempt so far, so a later round can branch from a branch of an earlier round.

<p align="center">
  <img src="figures/architecture-fork-tree.svg" width="900" alt="The rollouts of one SPROUT search, drawn as a tree. Each line is one rollout, and each dot on it is a checkpoint: a disk snapshot of the sandbox and the agent's history up to that turn. The parent rollout starts at the task, takes checkpoints C1 to C7 and fails with reward 0. In round 1, the review picks C2 and C5, and one branch starts at each. Each branch starts with the reviewer's turn, drawn as a diamond, and then the actor continues. Both round 1 branches fail with reward 0. In round 2, the review reads all three attempts and picks a checkpoint on the branch from C2. The round 2 branch from it passes with reward 1, and the search stops."/>
</p>
<p align="center"><sub>Round 1 branches from two checkpoints of the failed parent, and round 2 branches from a checkpoint of a round 1 branch.</sub></p>

`python -m sprout.eval.evaluate --method sprout` does these steps:

1. The evaluation submits the parent rollout and its grade. With `--parent-output`, it reads the
   parent of that evaluation instead.
2. If an attempt resolved the task, the evaluation stops.
3. The evaluation submits a review job over every attempt so far.
4. In the review job, an analyst reads each failed attempt on its own. The analyst reads these
   parts of the attempt:
   - the trajectory
   - the patch
   - the grade
5. A reviewer reads every analysis and picks checkpoints. Each checkpoint is a step of one
   attempt.
6. With `--branch-guidance assistant-turn`, the default, the reviewer writes the first assistant
   turn of each branch. The prompts ask for a turn with these commands, in this order:
   1. a probe that shows the defect
   2. the fix
   3. a check: the same probe again, and the existing tests nearest the change

   With `user-hint`, the reviewer writes a hint instead. With `none`, the branch gets neither.
7. The evaluation submits a branch job from each picked checkpoint, with a grade job for each
   branch.
8. If a round remains, the evaluation goes to step 2. `--rounds` sets the number of rounds.

If the review picks no checkpoint, the evaluation stops and writes the reason to `stop_reason`
in `summary.json`. If a picked checkpoint has no recovery point, the evaluation skips it and
records it in `plan-round<N>.json`. If a rollout job does not succeed, or a grade gives no
reward, the evaluation writes the error to `summary.json` and stops with that error.

The reviewer's turn runs through the normal tool path, so the history of the branch and its disk
agree. Then the actor continues from that turn. The analyst and the reviewer can use another
model than the actor, with `--review-model` and `--review-endpoint`. Every prompt of the review
is in [prompts.yaml](../resources/review/prompts.yaml). The summary of each evaluation records the
sha256 of that file.

## Processes

Sprout runs as separate processes around one job queue, the Run Store. Clients submit jobs to
the Run Store, and workers claim the jobs from it. No process assigns a job to a worker.

<p align="center">
  <img src="figures/architecture.svg" width="1000" alt="The processes of Sprout and the calls between them. Each arrow points from the caller to the callee. Miles, the external RL trainer, sends POST /rollout-groups to the RL driver (sprout.rl_driver). Three Run Store clients, the RL driver, manual evaluation (sprout.eval.evaluate) and the task gate (sprout.eval.gate), each run the search coordinator (sprout.search), which submits jobs to the HTTP API of the Run Store (/v1/jobs) and polls them. The API reads and writes PostgreSQL, which holds the jobs, attempts, leases and recovery points. A worker on each machine (python -m sprout.worker start) runs a claim loop (WorkerManager), which calls claim(), worker_heartbeat() and finish() on PostgreSQL directly. No process assigns jobs to workers. The claim loop starts an attempt child process for each job: a rollout, a grade or a review. In a rollout, the agent (mini-swe-agent, oracle or nop) makes tool calls to the tools (ToolAccess), which check each call and take the checkpoints. The tools call tools/call over JSON-RPC in the sandbox, a rootless podman container on btrfs or an AgentENV microVM, where sprout-runtime serves the tools. The agent sends its model requests through the model gateway to the external model endpoint, an OpenAI-compatible Chat Completions API."/>
</p>
<p align="center"><sub>Three clients submit jobs through the HTTP API of the Run Store, and each worker claims jobs from its PostgreSQL database.</sub></p>

The figure has these components:

| Component | Command or code | What it does |
| --- | --- | --- |
| Miles | an RL trainer outside this repository | Asks the RL driver for scored rollout groups with `POST /rollout-groups`. |
| RL driver | `python -m sprout.rl_driver` | Keeps each group in a SQLite ledger and runs it as Run Store jobs. Returns the messages without hints, and the reward. |
| Manual evaluation | `python -m sprout.eval.evaluate` | Evaluates one task. |
| Task gate | `python -m sprout.eval.gate` | Checks that the oracle of each task scores 1 and that doing nothing scores 0. |
| Run Store | `python -m sprout.runstore serve` | Keeps the jobs, attempts, leases, journals and recovery points in PostgreSQL. Serves the HTTP API on `127.0.0.1:18110`. |
| Worker | `python -m sprout.worker start` | Claims jobs and runs each attempt in a child process. |
| Agent | `src/sprout/worker/agents/` | Runs in a rollout: mini-swe-agent with a model, `oracle`, which runs the task's `solution/solve.sh`, or `nop`, which does nothing. |
| Tools | `ToolAccess` | Checks each tool call against `runtime/schema/tools.json`, runs it in the sandbox and takes the checkpoints. |
| Sandbox | `src/sprout/worker/sandbox/` and `src/sprout_sandbox/` | A rootless podman container or an AgentENV microVM. `sprout-runtime` serves the tools inside it. |
| Model endpoint | external | Serves OpenAI-compatible Chat Completions. A rollout reaches it through the model gateway of its attempt. A review job calls it directly. |

The three clients run the search coordinator, `sprout.search`. The coordinator submits the jobs
of each group and polls them. Then it advances the search. It runs no job itself. After a
restart, it follows the jobs that its stored groups name, and it does not submit them again.
Each job of one search carries one affinity key, so the jobs of one search run on one host. The RL
driver uses the id of the Miles group as the key. Manual evaluation makes one key for each task
evaluation, and the gate one for each task of a gate run. If that host is lost, the search runs
again under a new key, at most 2 times. The coordinator runs an RL group again from its roots.
`evaluate.py` runs a task evaluation again from a new parent, and the gate runs the lost check
again ([RL_DRIVER.md](RL_DRIVER.md#host-affinity), [EVALUATION.md](EVALUATION.md#a-lost-host)).

A job has one of three kinds:

| Kind | What its attempt does |
| --- | --- |
| `rollout` | Runs an agent in a sandbox through the `Runner`. A branch is a rollout job with a `parent_point` and a `continuation`. |
| `grade` | Runs a verifier on the work of one rollout: `harbor` for a Harbor task, `swe-rebench-v2` for SWE-rebench V2. |
| `review` | Sends the requests of the analyst and the reviewer to the model endpoint. It runs nothing in a sandbox. |

The Run Store's HTTP API reads and writes its PostgreSQL database. A worker does not use the API.
Its claim loop, `WorkerManager`, calls the database directly. The worker id is `<host>/<name>`,
and `--worker-name` sets the name. At its start, the worker checks its profiles, and it refuses to
start on an error. Then the manager registers the worker with the checksums of the grade files that
its profiles name, and settles the jobs that an earlier session of the worker left. A thread of the
manager reports a heartbeat every 30 s, which renews the leases of the worker's jobs. The loop runs
these steps until the worker stops:

1. The worker calls `settle_dead_workers()`. Each running job of a worker that stopped, or that
   has no heartbeat for 600 s, runs again from its spec, once its lease has expired. If no worker
   of the host of the job's affinity key reported a heartbeat for 600 s, because its workers died
   or stopped, the job fails as `data_lost` instead, also while it is queued.
2. While the worker has a free slot, it calls `claim()`. `--concurrency` sets the number of
   slots. The default is 1.
3. `claim()` leases the next queued job that the worker can run to the worker. It locks the row of
   the job with `FOR UPDATE SKIP LOCKED`, so two workers never claim the same job. These rules
   decide which jobs the worker can run:
   - The first claim of a job of an affinity key binds the key to the host of the worker. Only
     the workers of that host claim the later jobs of the key, before the other jobs.
   - A `swe-rebench-v2` grade goes only to a worker whose grade profiles name its dataset and
     its parser.
4. The worker starts the attempt as a child process, from the module
   `sprout.worker.jobs.rollout`. The child holds no database credentials.
5. While the attempt runs, the worker calls `progress()` at most every 5 s. The worker stops the
   attempt in these cases:
   - The attempt reaches its deadline.
   - `progress()` reports a stop request.
   - The worker shuts down.
   - No heartbeat was acknowledged for 540 s.
6. When the attempt ends, the worker records its result with `finish()`. If the worker shuts down,
   it calls `release()` instead. The job goes back to the queue, and the attempt does not count.

Between two passes, the worker waits 1 s, or less when an attempt ends.

An attempt that is lost, for example with its worker, runs its job again from the start of its
spec. The worker deletes the data of the attempt first. If the worker cannot confirm that the
sandboxes of an attempt are gone, it records them, and its clean-up frees them later.
[Run Store](RUNSTORE.md) gives the rerun rules and the other failure paths.

A grade job runs the task's verifier in one of two ways:

- If the verifier shares the agent's environment, the rollout keeps its sandbox. The grade runs
  the verifier in that sandbox, on the same worker.
- If the verifier has its own environment, the grade starts a sandbox from the final disk
  snapshot and copies the artifacts out. Then the verifier runs in a sandbox from the task's
  `tests/Dockerfile`.

In the first way, the grade must run on the worker that holds the sandbox
([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)):

1. After its final disk snapshot, the rollout keeps its sandbox for 20 minutes.
2. The Run Store binds the grade job to the worker that keeps the sandbox. Only that worker can
   claim the grade, and it claims bound grades before other jobs.
3. The grade runs the verifier in the kept sandbox. Then it destroys the sandbox.

If the 20 minutes pass before a claim, the Run Store finishes the grade with reward 0 and the
reason `sandbox_lost`. If the bound worker dies or stops, the grade waits for a worker under the
same id until then. If the host is lost first, the grade fails as `data_lost`. After its time
passes, the worker removes each kept sandbox that no grade took.

The same checkpoints feed RL training. The [RL driver](RL_DRIVER.md) serves
[Miles](https://github.com/radixark/miles) rollout groups from the Run Store, and branches a
failed rollout at the checkpoints that the review picks. The worker grades
[SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2) with a port of its official
harness. The training export of each rollout marks each message that no policy sampled: the
restored prefix and each inserted turn. The export also leaves out the hint of a branch.
