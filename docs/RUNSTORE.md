# Run Store

The Run Store is the job queue of Sprout: an authenticated HTTP API over PostgreSQL. A worker claims
each job and runs each attempt of it in a child process on its host. A lost attempt runs again
from the start of its spec. The jobs of one affinity key run on one host, and fail as `data_lost` if
that host is lost. A job has one of three kinds:

- `rollout`: one agent run in a sandbox ([Agents](#agents))
- `grade`: the verdict of a benchmark on one rollout ([Official grading](#official-grading))
- `review`: hindsight over finished attempts, with a model and without a sandbox ([Review](#review))

The [RL driver](RL_DRIVER.md) and the [evaluation](EVALUATION.md) are its clients. Read
[Failure handling](#failure-handling) before you investigate a job that did not succeed.

## Job flow

A client and a worker never talk to each other. Each one talks only to the API or to the database.
The figure shows the calls of one attempt in order.

<p align="center">
  <img src="figures/runstore-job-flow.svg" width="1080" alt="A sequence diagram of one Run Store job with seven lifelines: client, Run Store API, PostgreSQL, WorkerManager, Worker, attempt child and sandbox. WorkerManager and Worker are inside one box, the sprout.worker start process. 1: the client sends POST /v1/jobs to the API. 2: the API calls Store.submit(), and the job is queued. 3: WorkerManager calls Store.claim(), and the job is running with a lease of 600 s, which the heartbeats of WorkerManager renew. 4: WorkerManager calls run_claimed() on Worker. 5: Worker calls freeze_payload(). 6: Worker starts the attempt child and sends the payload on stdin. In a loop until the child exits, 7: the child sends tool calls to the sandbox, and 8: Worker calls progress() at most every 5 s. 9: the child leaves outcome.json and the journal for Worker, a file relation drawn dashed. 10: Worker calls _cleanup() to check that each claimed sandbox is gone. 11: Worker calls finish(), and the job is settled. 12: the client calls Client.wait(), which polls the result."/>
</p>
<p align="center"><sub>One attempt of a job, from <code>POST /v1/jobs</code> to the settled state, with the calls numbered in order.</sub></p>

The attempt child is a process of the module `sprout.worker.jobs.rollout`. It writes `outcome.json`
and its journal `trajectory.jsonl` in `<artifact_root>/<job id>/<attempt id>/`. Each `progress()`
call records the phase and returns a stop request. The worker also copies new journal events to
`rs_events`.

## Boundaries

The Run Store serves one trusted operator, and each client submits tasks that run code. So keep the
listener on `127.0.0.1`, or put it behind an authenticated proxy.

### Owned sandboxes: podman by default, or microVMs

The `backend` of the effective spec, the request `spec` merged over its profile's defaults, selects
where a rollout or a grade runs. Before an attempt starts, `owned_backend()` refuses every backend
but these two, and a rollout without a `sandbox_image`:

| Backend | Sandbox | Check after each attempt |
|---|---|---|
| `podman`, the default | A rootless container on the btrfs `store` that `ops/setup_podman.sh` makes. | No container, network or rootfs subvolume remains. The worker removes a leftover as `python -m sprout.worker reap` does, then looks again. |
| `microvm` | An AgentENV microVM. | The worker deletes the sandbox, and AgentENV then answers 404. Without a `server_url`, the check fails. |

The worker checks each sandbox in the attempt's ledger `resources.jsonl`, except a kept one. If the
check fails, the worker still publishes the outcome, and frees the sandbox later
([Supervision](#supervision)). The worker and its children run the first podman of this list:

1. `podman_bin` in the `podman` section of the effective `backend`
2. `$SPROUT_PODMAN_BIN`, which the services' `podman` check sets
3. the podman that `ops/setup_podman.sh` installed
4. `podman` on `PATH`

After you change the podman, restart the worker.

### Agents

The `agent` of a rollout says what runs in its sandbox, through `sprout.worker.run.runner.Runner`:

- `mini`, the default: the model, in the slot `mini-swe-agent`
- `harness`: the model, in the slot `sprout-harness`, Sprout's own harness
  ([ADR-0008](decisions/0008-sprout-harness.md)). `extra.harness` sets its options, such as
  `on_invalid` (`feedback` or `terminate`), `toolset` and `max_total_tokens`.
- `oracle`: the `solution/solve.sh` of the Harbor task `harbor_task`, as Harbor's `oracle` runs it
- `nop`: nothing

Each run has a journal and a final disk snapshot. The worker sets the run's identity and paths, such
as `run_id` and `cwd`, so `JobSpec.validate()` rejects a request that sets them.

### Kept sandboxes

A rollout with `keep_sandbox: true` keeps its sandbox for `KEEP_SANDBOX_S` (1200 s), so that a
`harbor` grade can run in it ([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)). If the attempt
completes or is truncated with a final snapshot, its result has `kept_sandbox` with a UTC
`keep_until`. The worker then records the sandbox in `<artifact_root>/kept-sandboxes.jsonl`, and the
cleanup leaves it. Otherwise the cleanup removes the sandbox. Once a minute, the worker removes each
recorded sandbox that no grade took by its `keep_until`.

The Run Store accepts a grade with `in_place_job` only after that rollout finished with a
`kept_sandbox`. The grade is then bound to `bound_worker`, the worker that kept the sandbox:

| Case | Result |
|---|---|
| The bound worker claims the grade before `keep_until`. | The grade takes the sandbox over, and the cleanup after it removes the sandbox. If the sandbox is gone, the reward is 0 with the reason `sandbox_lost`. If the attempt ends before its child starts, the worker keeps the sandbox again. |
| `keep_until` passes while the grade is queued. | Any worker's next claim or sweep finishes the grade `succeeded`, with reward 0 and `sandbox_lost`. |
| The bound worker dies or stops. | The grade waits for a worker under the same id. |
| The host of the grade's key is lost before `keep_until`. | The grade fails as `data_lost` ([Lost hosts](#lost-hosts)). |
| No attempt is left after a lost attempt or an infrastructure failure. | The grade finishes `succeeded`, with `sandbox_lost`. |

### Recovery points are closed messages

A branch starts only from a point where the agent's history and its disk agree. So a recovery point
has these parts:

- a finished model response, with all the tool results of that response
- a snapshot of the sandbox's disk, without processes or memory
- the native prefix of the agent's history, pinned by a checksum

### Storage

PostgreSQL holds the state of the jobs, and the disk holds the large files. Back up both together.
The schema is `resources/runstore/schema.sql`:

| Table | Records |
|---|---|
| `rs_workers` | each worker: its host, session, process, grade checksums, last heartbeat and stop |
| `rs_jobs` | each job: its request, affinity key, state, lease and result |
| `rs_affinity` | the host that each affinity key is bound to |
| `rs_attempts` | each attempt: its frozen payload, execution record and result |
| `rs_events` | the journal events of each attempt |
| `rs_tools`, `rs_prefix_nodes` | the tool calls and their prefix DAG |
| `rs_recoveries` | the recovery points |

PostgreSQL's text and jsonb cannot hold U+0000, which a command's binary output can carry, so the
Run Store stores each U+0000 in an event, a result, an error or an indexed call as U+FFFD. The
journal on disk keeps the original.

The disk holds the attempt directories under `artifact_root`, and the snapshots: the store's
`snapshots/` on podman, and AgentENV's layers on a microVM.

## Start

The Run Store needs a dedicated, persistent PostgreSQL database. The services start the database,
the API and a worker with `python -m sprout.services up runstore-api runstore-worker`
([services](SERVICES.md)). To start the Run Store by hand, do these steps from the repository root:

1. Install the extras: `python3 -m pip install -e ".[mini,runstore]"`.
2. Copy `resources/runstore/config.example.json` to `<config>`. Replace each `/path/to/` in it.
3. Export `SPROUT_RUNSTORE_DSN`, for example `host=127.0.0.1 dbname=sprout_runstore`, and a random
   `SPROUT_RUNSTORE_TOKEN`, for example from `openssl rand -hex 32`.
4. Create the schema with `python3 -m sprout.runstore init --config <config>`. It is idempotent.
5. Start the API with `python3 -m sprout.runstore serve --config <config>`. It listens on
   `127.0.0.1:18110`, unless you give `--host` and `--port`.
6. In a second shell with the same DSN, start a worker with
   `python3 -m sprout.worker start --config <config> --concurrency 8`.

`python -m sprout.worker start` follows these rules:

- It refuses a volatile `artifact_root`, such as a directory under `/tmp`. So do `init` and `serve`.
- It checks the profiles before the worker registers ([Checks at the start](#checks-at-the-start)).
- It runs at most `--concurrency` attempts at a time. The default is 1.
- Its worker id is `<host>/<name>`. `--worker-name` sets `<name>`, and the default is `default`.
  Each worker on one host needs its own name.
- With `--once`, it claims one job, waits for the attempts that it started, and exits.

Before you start parallel attempts, run `python ops/prepull_images.py --tasks-dir <tasks dir>`. On
podman, it extracts the base of each image into the store one time. On a microVM, it builds each
template one time, because concurrent builds of one template race.

### Config

The config file holds `artifact_root` and the named `profiles`. Each request names its profile. The
example config has a podman and a microVM profile for rollouts and for grades, and a `review`
profile. A profile can have these keys:

| Key | What it does |
|---|---|
| `python` | The interpreter of the attempt child. A wrapper script must `exec` it, because the child stops when its parent is not the worker. |
| `run_defaults`, `grade_defaults`, `review_defaults` | The defaults of a `rollout`, `grade` or `review` spec. At submission, the API merges them under the request `spec`, key by key, and stores the result. |
| `env` | The variables that the agent gets. |
| `worker_env` | The variables that only the attempt child gets, such as the key that a rollout contract's `api_key_env` names. Without that key, the rollout fails before its agent starts. |
| `snapshot_repository` | The on-disk snapshot repository of a microVM profile, for the snapshot check. |

The config and the requests must not hold a credential, a nonempty value under a key such as
`api_key`. Write `{"$env": "VAR"}`. The attempt child reads `VAR` from the worker's environment. The
agent gets `VAR` and each `worker_env` variable as empty, unless `env` sets it.

### Checks at the start

The worker checks its profiles when it starts, and the API checks each request at submission. So no
job fails on the configuration of its worker
([ADR-0006](decisions/0006-worker-heartbeat-and-rerun-from-scratch.md)). Before the worker
registers, `checked_grades()` in `src/sprout/worker/profiles.py` checks these rules for each profile:

- Each key of a `*_defaults` section is a field of `RunSpec`, `GradeSpec` or `ReviewSpec`.
- The `backend` of `run_defaults` and of `grade_defaults` is `podman` or `microvm`, or none.
- A grade profile names all four keys of its [grade files](#grade-files), or none of them.
- Each grade file has the checksum that the profile gives.

If a rule fails, `start` exits with `Profile <name>: <reason>`. At submission, the API answers 422
for a request that `JobSpec.validate()` refuses or that names an unknown profile.

### Grade files

A `swe-rebench-v2` grade reads a dataset and a parser file on the worker's host. A grade profile
names both in its `grade_defaults`, each with its checksum:

| Key | Value |
|---|---|
| `dataset_path`, `dataset_sha256` | The JSON or JSONL dataset, and its SHA-256 in hex. |
| `parser_path`, `grader_revision` | The upstream `lib/agent/log_parsers.py`, and `sha256:` with its SHA-256 in hex. |

The worker registers the checksum pair of each grade profile in `rs_workers.capabilities`.
`Store.claim()` gives a `swe-rebench-v2` grade only to a worker that registered the grade's pair. A
grade whose pair no live worker registered stays queued until its client stops it. A worker reads
its grade files again only when it starts again.

### Image USER and ENV

The runtime runs every command as root. Two options of a backend section run these commands the way
Harbor runs them:

- the agent's shell commands
- the `solution/solve.sh` of an `oracle` run
- the `tests/test.sh` of a `harbor` grade

| Option | What it does |
|---|---|
| `image_user` | The commands run as the image's `USER` through `runuser`, or as root when the image names none. |
| `image_env` | With `image_user`, the commands get the image's `ENV` again after the switch of user. On a microVM, the template also starts the runtime under the image's `ENV`. |

Both options are off by default. `eval.evaluate` and `eval.gate` turn both on, unless you pass
`--no-image-user` or `--no-image-env`. A microVM template needs `image_env` to run the image's own
tools, because AgentENV's guest agent rebuilds the runtime's `PATH`. Each of the 113 DeepSWE images
had `PATH` entries that the runtime lacked.

### Timeouts

The watchdog stops an attempt child at its limit:

| Limit | Value |
|---|---|
| rollout child | `timeout_s`, default 3600 s. With a rollout contract, at most the time left before `deadline_at`. A message-export rollout gets 120 s more. |
| grade child | `timeout_s` for the tests, plus 1200 s for the preparation and the collection. |
| review child | `timeout_s` for the whole review. |

A rollout's microVM lease is `max(sandbox_ttl, timeout_s + 600)` s, and a kept sandbox adds
`KEEP_SANDBOX_S` to `timeout_s`. A podman sandbox has no lease. One `shell` call has 300 s by
default.

## Submit and observe

A client submits a `JobSpec` under an idempotency key, then waits for the result:

```python
import os
from sprout.runstore.client import Client
from sprout.runstore.specs import JobSpec

client = Client("http://127.0.0.1:18110", os.environ["SPROUT_RUNSTORE_TOKEN"])
spec = {"prompt": "The task...", "model": "my-model", "sandbox_image": "my-image", "timeout_s": 3600}
job = JobSpec(kind="rollout", profile="mini", context={"task_id": "task-1"}, spec=spec)
job_id = client.submit(job, idempotency_key="experiment-1/task-1/parent")
print(client.wait(job_id))  # a TimeoutError after timeout_s does not cancel the job
```

Each route requires `Authorization: Bearer $SPROUT_RUNSTORE_TOKEN`, or the API answers 401. The same
key with the same request returns the same job. The API answers 409 to a request that contradicts
the stored state, such as another request under a used key. It answers 422 to an invalid request,
and 404 for an unknown job, attempt or point.

| Route | Result |
|---|---|
| `POST /v1/jobs` + `Idempotency-Key` | The accepted job (202). |
| `GET /v1/jobs?state=&limit=100` | Recent jobs, newest first. The limit is at most 1000. |
| `GET /v1/jobs/{id}` | The frozen request, the lease, the state, the result and the error. |
| `GET /v1/jobs/{id}/attempts` | The attempts, with their progress and execution records. |
| `GET /v1/jobs/{id}/events?attempt_id=&after=0&limit=1000` | The journal events after `seq`. The limit is at most 10000. |
| `GET /v1/jobs/{id}/tools?attempt_id=&after=0&limit=1000` | The tool calls and responses after `depth`. The limit is at most 10000. |
| `GET /v1/jobs/{id}/recovery-points?attempt_id=` | The recovery points, each with `available`. |
| `GET /v1/jobs/{id}/result` | The result, with `ready`: true for `succeeded`, `failed` and `cancelled`. |
| `POST /v1/jobs/{id}/branch` | A new queued branch from `point_id` ([Branch](#branch)). |
| `POST /v1/jobs/{id}/cancel` | Cancels a queued job. Any other job gets 409. |
| `POST /v1/jobs/{id}/stop` | Cancels a queued job, or stops a running one ([Stopping](#stopping)). |

The default `attempt_id` is the job's active attempt.

## Branch

A branch is a new rollout job that starts from a recovery point of a finished rollout. Call
`client.branch(job_id, point_id, idempotency_key=..., continuation=...)` with a point of
`client.recovery_points(job_id)` whose `available` is true. `available` checks the prefix checksum,
and that the backend still has the snapshot. The branch keeps the spec and the profile of its parent.
In `overrides`, it can change only the fields of `BRANCH_OVERRIDES`, such as `model`. The required
`continuation` says what the actor gets after the restored prefix
(`src/sprout/contract/continuation.py`):

| `guidance` | What the actor gets after the prefix |
|---|---|
| `assistant-turn` | The `assistant_turn`, run through the normal tool path. Its call IDs must be new to the history. With `assistant_turn_rendering: "reasoning"`, its text enters the history as `reasoning_content`. |
| `user-hint` | A user message with the `hint`. The training export strips the hint. |
| `none` | Nothing. |

A branch inherits its parent's rollout contract. `budget` sets the branch's own `max_turns`,
`deadline_at` or both over that contract. The `max_turns` of a branch counts from zero. To give a
branch only the turns that its parent had left, set `max_turns` to the parent's `max_turns` minus
the point's `message_step`. Each attempt of a branch, also a rerun, starts at its `parent_point`.

## Review

A review job runs SPROUT's analyst and reviewer (`sprout.worker.jobs.review`) over the finished
attempts of one task, and returns branch points. Its caller, the RL driver or
`eval.evaluate --method sprout`, submits the branches. `ReviewSpec` in `src/sprout/runstore/specs.py`
has its fields. These fields shape the result:

| Field | Default | Meaning |
|---|---|---|
| `parents` | | The attempts to read, each with the result of its grade job in `grade`. |
| `guidance` | `assistant-turn` | What the review writes at each point: `assistant-turn`, `user-hint` or `none`. |
| `points`, `count_mode` | 2, `adaptive` | At most (`adaptive`) or exactly (`fixed`) this many points, in total or, with `points_per_attempt`, in each failed attempt. |
| `candidates` | 4 | The independent turns written at each point. A `user-hint` or `none` review must set 1. |

The review analyses each failed parent, then picks the points. Candidates with the same commands
merge, and keep their count in `multiplicity`. Each point names its parent attempt, its tool `step`
and the reviewer's `why`. The caller maps `step` to the recovery point of that `tool_depth`. The
reviewer writes each turn flat, as `content` and `commands`, because it miscounted the braces of the
nested form. In one TerminalBench run, the brace count rejected 19 of 79 reviewer replies.

### Calls of a review

A call starts as soon as the answers that it needs are in. The figure shows which call waits for
which.

<p align="center">
  <img src="figures/runstore-review-calls.svg" width="1080" alt="Two dependency graphs of the calls of one review over two failed attempts, with two points and two candidates. Top, one point review, the default: analyst-1 and analyst-2 each send their report to points, which sends point 1 to point1-candidate1 and point1-candidate2, and point 2 to point2-candidate1 and point2-candidate2. Bottom, with points_per_attempt: analyst-1 sends its report to points-1, which sends its point to point1.1-candidate1 and point1.1-candidate2. analyst-2 sends its report to points-2, which sends its point to point2.1-candidate1 and point2.1-candidate2. The two chains do not wait for each other."/>
</p>
<p align="center"><sub>The calls of a review over two failed attempts, with two candidates at each point, named as their receipt directories <code>reviews/&lt;name&gt;/</code> name them.</sub></p>

At most `max_concurrent_requests` (8) requests are in flight at one time. Each call writes its
receipts to `reviews/<name>/`, and each call but an analysis writes `plan-<name>.json`. The result
keeps `Call` order, whatever order the calls finish in. If a call raises, no later call in
`Call` order starts, and the review raises the first failure. When the worker stops a review, the
review stops every call and raises at once.

### Leak checks

Each candidate has `leak_terms` (`src/sprout/worker/jobs/review/leak/terms.py`). These are the
hidden test names and paths of the grade's `fail_to_pass` and `test_patch_paths` that the turn
mentions and that the history before the point never shows. This check only reports. The trainer
decides whether to credit the candidate. With `leak_judge` in the spec, Jev also judges each turn
against the grade's test patch. Then a turn that either check flags goes back to the reviewer, up to
`resamples` (2) times. After that, the review takes the turn, and `leak_judge.leak` records the flag.

### Review requests

Each call sends its requests through its own `ReviewTransport`
(`src/sprout/worker/jobs/review/transport.py`). The transport writes each try to a receipt in
`reviews/<name>/`. It sends a try that timed out one more time, with half the output budget. After a
status in `TRANSIENT_STATUSES`, such as 502, or a dropped connection, it tries again after 10 s, 30 s
and 60 s. After an HTTP 400, the analyst or the reviewer halves the prompt budget, at most
`PROMPT_HALVINGS` (2) times. These limits come from measured runs:

| Limit | Value | Why |
|---|---|---|
| `prompt_chars` | 400,000 characters | About 100k tokens, so the request fits the context with 64k output tokens. The largest reviewer request known to succeed had 746k characters. |
| `RESULT_CHARS`, `ARG_CHARS`, `MESSAGE_CHARS` | 6000, 4000, 2000 | The characters of one transcript line. Tool results had a median of about 1.9k characters. A cap of 300 dropped the assertion that explains a failure. |
| `CHARS_PER_TOKEN` | 3.2 | The pessimistic end of what the journals measured. |

The analyst's transcript keeps the head and the tail of a long line, because a test's verdict is at
the end. A transcript that is still too long loses its middle, because the late steps hold the
failure. A reviewer request that is too long goes through the `STAGES` of
`src/sprout/worker/jobs/review/request_size.py`. They window the branch histories first, because a
branch repeats its parent's prefix. They never reduce the analyst reports. The plan file records the
whole exchange of a call, and the fit of its last request in `prompt_budget`.

## Training export

A rollout whose `extra.rollout_contract` sets `message_export` returns its trajectory as training
messages for the [RL driver](RL_DRIVER.md) (`src/sprout/worker/jobs/export.py`). The attempt child
marks the result `truncated` when all these conditions are true:

- `stop_reason` is `max_turns_reached`, `timeout` or `max_total_tokens`, or it is an invalid
  attempt's `format_invalid`, `response_truncated` or `repetition` (the harness in training mode).
- The run has a final snapshot, and the journal has no `execution.uncertain` event.
- The failure is not an `infrastructure` failure.

The worker settles a truncated result as `succeeded`. The export adds these fields:

| Field | Content |
|---|---|
| `training_messages` | The native history of the slot (mini's or the harness's), without hints. |
| `training_tools` | The tool schemas that the model was offered. |
| `training_provenance` | The messages that no policy sampled: `prefix_messages` restored from the branch point, and `inserted_messages`. |
| `training_origin` | For a branch: the `job_id`, `point_id` and `tool_depth` of its point. |

If the export fails, the result has `training_export_error`. A truncated result then falls back to
its last recovery point, if it has one, and records the fallback in `timeout_fallback`.

### Salvage

If a message-export rollout fails and will not run again, the worker cuts it back to its last valid
recovery point (`salvage_interrupted()`). The job then finishes `succeeded`, with the status
`truncated`, `stop_reason: "interrupted"` and the failure in `interrupted`. An attempt without a
closed turn that has an assistant message stays failed.

## Official grading

A grade job gives the verdict of a benchmark on the work of one rollout. Queue it as
`JobSpec(kind="grade", profile="grade", spec=asdict(GradeSpec(...)), affinity=<key>)`. A grade from
a `snapshot_id` needs the `affinity` of its rollout. An unresolved task is a successful job, with
`resolved: false` or a reward of 0. If a grader cannot make the grade, for example because a sandbox
does not start, it raises. The job then runs again while attempts remain.

### The `harbor` grader

The `harbor` grader (`src/sprout/worker/jobs/harbor/grade.py`) runs the verifier of the Harbor task
`task_dir` as Harbor 0.22.0 does ([ADR-0003](decisions/0003-sprout-runs-harbor-verifier.md)). It
grades a fresh sandbox from `snapshot_id`, or a [kept sandbox](#kept-sandboxes). A verifier that
does not share the agent's environment runs in a sandbox from `tests/Dockerfile`. The reward comes
from `reward.json`, or else `reward.txt`. If the verifier wrote no reward, the reward is 0, and
`reason` names the cause, such as `verifier_timed_out`.

### The `swe-rebench-v2` grader

The `swe-rebench-v2` grader (`src/sprout/worker/jobs/grade.py`) does what the upstream
`SWE-rebench-V2/scripts/eval.py` does, in two sandboxes:

1. It checks the [grade files](#grade-files) against their checksums, and reads the row `instance_id`.
2. In a sandbox from `snapshot_id`, it writes the agent's diff against `base_commit` to `model.patch`.
3. In a pristine sandbox from `image`, it applies the diff, then the row's `test_patch`.
4. It runs `install_config.test_cmd`. It parses the log with the parser of `install_config.log_parser`.

So only the agent's diff reaches the verdict. The task is resolved when every `FAIL_TO_PASS` and
`PASS_TO_PASS` test passed. A completed result has these fields:

- `resolved`, and `progress`, the share of the `FAIL_TO_PASS` tests that passed
- `model_patch`, `test_patch` and `log`, the paths of the files in the attempt directory
- `test_patch_paths`, `touched_test_paths` and `fail_to_pass`, which no analyst prompt shows
- `test_statuses` and `expected_pass_share`, after the tests finished

A failure that the agent's work can cause gives a result of 0, with a `reason`:

| `reason` | When |
|---|---|
| `model_patch_unavailable` | The snapshot's `git diff` fails, for example after the agent removed `.git`. |
| `model_patch_apply_failed` | The agent's diff does not apply. |
| `test_patch_apply_failed` | The test patch does not apply. |
| `tests_timed_out` | The tests outran `timeout_s`. |
| `test_runner_killed` | The tests exited with a code other than 0 or 1, for example after an OOM kill. |

## Claims

`Store.claim()` gives a live worker the next queued job that the worker can run on its host. If the
worker is not live, `claim()` raises `Conflict`. A claim takes a job only when all these conditions
are true:

- `ready_at` has passed, and `attempt_count` is less than `max_attempts`.
- The job's kind is one of the kinds that the worker runs.
- The job is not bound to another worker ([Kept sandboxes](#kept-sandboxes)).
- The job's affinity key is unbound, or bound to the worker's host.
- A `swe-rebench-v2` grade has a checksum pair that the worker registered
  ([Grade files](#grade-files)).

The claim takes the jobs bound to the worker first. Then it takes the jobs whose key is bound to the
worker's host, then the others. In each group, the earliest `ready_at`, then `created_at`, comes
first. The claim does these actions in one transaction:

1. It binds the job's affinity key to the worker's host, if the key is unbound.
2. It adds an attempt, and adds 1 to `attempt_count`.
3. It sets a lease of 600 s.

If two claims on two hosts bind one key at the same time, the first commit binds it. The other claim
returns None, and its job stays queued.

### Affinity keys

Each job has an affinity key, so that the jobs of one group run on the host that holds their data
([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). A rollout leaves its snapshots and
its kept sandbox on its host. `Store.submit()` stores the key in `rs_jobs.affinity`. A request
without `affinity` takes the first of these keys:

1. the key of the job that the request names:
   - for a branch, the job of its `parent_point`
   - for a grade with `in_place_job`, that rollout
   - for a review, the jobs of its `parents`, which must share one key
2. the id of the job itself

A grade from a `snapshot_id` names no job, so it must set `affinity` to the key of its rollout. A
request with `affinity` must agree with the key of each job that it names. Otherwise the API answers
422. The first claim of a job binds its key to the claiming worker's `host`, in `rs_affinity`. The
key stays bound, also after its host is lost ([evaluation](EVALUATION.md#affinity-keys)).

## Failure handling

A job whose attempt is lost runs again from the start of its spec, never from a recovery point
([ADR-0006](decisions/0006-worker-heartbeat-and-rerun-from-scratch.md)). An attempt is lost when its
worker dies or is cut off, or when its child leaves no outcome. An infrastructure failure also runs
the job again, and an actor failure does not. If the host of a job's key is lost, the job fails.

### Job states

A job has five states. The figure shows each transition and the function that makes it. No
transition leaves a terminal state.

<p align="center">
  <img src="figures/runstore-job-states.svg" width="1080" alt="Five Run Store job states and the functions that move a job between them. Store.submit() creates a job in queued. queued goes to running through claim(), when ready_at has passed and attempts are left, the job's affinity key is unbound or bound to the worker's host, the job is not bound to another worker, and the worker holds the checksums of a swe-rebench-v2 grade. A queued bound grade goes to succeeded with the result sandbox_lost through settle_dead_workers() or claim() after its keep_until, also on a lost host. queued goes to failed with the phase data_lost through settle_dead_workers() when the host of its affinity key is lost, a bound grade too before its keep_until. queued goes to cancelled through cancel_queued() or request_stop(). running goes back to queued through finish(rerun=True) after an infrastructure failure or a lost attempt, through settle_dead_workers() when the worker is dead, the lease expired and the host of the job's key is not lost, and through release() when the worker shuts down. A rerun needs an attempt left, and none of the three goes back to queued after a stop request: then release() moves running to cancelled, and the other two to failed. running goes to succeeded through finish() when the attempt completed or was truncated. A bound grade without an attempt left goes from running to succeeded with the result sandbox_lost, through finish() or settle_dead_workers(). running goes to failed through finish() when the attempt failed and does not run again, and through settle_dead_workers() when no attempt is left, or with the phase data_lost when the host of the job's key is lost. succeeded, failed and cancelled are terminal."/>
</p>
<p align="center"><sub>The five job states of <code>rs_jobs</code>, with the function that makes each transition and its condition.</sub></p>

The phase of the job is in parentheses when it differs from the state:

| From | Trigger and condition | Actor | To |
|---|---|---|---|
| none | A client submits a valid request under a new idempotency key. | API, `Store.submit()` | `queued` |
| `queued` | `ready_at` has passed and attempts are left. The worker can run the job on its host ([Claims](#claims)). | `WorkerManager`, `Store.claim()` | `running` (`claimed`) |
| `queued` | A bound grade's `keep_until` has passed. | any worker, `Store.settle_dead_workers()` or `Store.claim()` | `succeeded`, with the result `sandbox_lost` |
| `queued` | The host of the job's affinity key is lost. | any worker, `Store.settle_dead_workers()` | `failed` (`data_lost`) |
| `queued` | A cancel request or a stop request arrives. | API, `cancel_queued()` or `request_stop()` | `cancelled` |
| `running` | The attempt completed, or its result is truncated. | `Worker`, `finish()` | `succeeded` |
| `running` | The attempt failed, and the job does not run again. | `Worker`, `finish(state="failed")` | `failed` |
| `running` | The attempt is lost or had an infrastructure failure. Attempts are left, and nobody requested a stop. | `Worker`, `finish(rerun=True)` | `queued` (`retry_wait`) |
| `running` | The worker is dead, the lease expired, and the host of the job's key is not lost. Attempts are left, and nobody requested a stop. | any worker, `Store.settle_dead_workers()` | `queued` (`retry_wait`) |
| `running` | As the row before, but no attempt is left or somebody requested a stop. | any worker, `Store.settle_dead_workers()` | `failed` |
| `running` | The worker is dead, the lease expired, and the host of the job's key is lost. | any worker, `Store.settle_dead_workers()` | `failed` (`data_lost`) |
| `running` | A grade in a kept sandbox is lost or failed on the infrastructure. No attempt is left, and nobody requested a stop. | `Worker`, `finish(rerun=True)`, or any worker, `Store.settle_dead_workers()` | `succeeded`, with the result `sandbox_lost` |
| `running` | The worker shuts down, and nobody requested a stop. | `Worker`, `release()` | `queued` (`released`) |
| `running` | The worker shuts down, and somebody requested a stop. | `Worker`, `release()` | `cancelled` |

These rules also apply:

- While the attempt runs, its phase is `starting`, then `rollout`, `grading` or the phase in the
  child's `progress.json`.
- A job that runs again waits 30 s × `attempt_count`.
- A job that fails with no attempt left has `attempt_errors`: the `error` of each attempt that
  counted.
- `release()` deletes the attempt's events, tool calls and recovery points. The released attempt
  does not count against `max_attempts`, and the job is ready at once.

### Workers and heartbeats

Each worker shows that it lives with a heartbeat every 30 s. The figure shows what happens as the
time since the last acknowledged heartbeat grows.

<p align="center">
  <img src="figures/runstore-worker-liveness.svg" width="1080" alt="A timeline in four columns, from the send time of the worker's last acknowledged heartbeat: 0 s, every 30 s (HEARTBEAT_S), 540 s (CUT_OFF_S) and 600 s (WORKER_LIVE_S). Two lanes: the worker, WorkerManager, and its running jobs in the Run Store, rs_jobs. At 0 s, worker_heartbeat() is acknowledged, or register_worker() registers the worker, and the lease_until of each running job becomes the heartbeat time plus 600 s. Every 30 s, worker_heartbeat() sends the next heartbeat. If it is acknowledged, the time goes back to 0 s. If no heartbeat is acknowledged for 540 s, the worker is cut off: it claims nothing, and the watchdogs stop the children with worker_cut_off. 60 s later the worker is dead and its leases expired. Any worker's settle_dead_workers() then settles each running job. If a row of the job's host had a heartbeat in the last 600 s, the job goes to queued (retry_wait), or to failed when no attempt is left or a stop was requested. If no row of the host had one, the host is lost: the running job goes to failed with the phase data_lost, and so does each queued job of an affinity key bound to the host."/>
</p>
<p align="center"><sub>What the worker and the Run Store do as the time since the last acknowledged heartbeat grows.</sub></p>

| Time since the last acknowledged heartbeat | Actor | What happens |
|---|---|---|
| every 30 s | `Heartbeat`, a thread of `WorkerManager` | It calls `Store.worker_heartbeat()`. The Run Store sets the row's `seen_at`, and moves the end of each lease of the worker to 600 s later. |
| 540 s (`CUT_OFF_S`) | `WorkerManager` | The worker is cut off. Each `Watchdog` stops its child with `worker_cut_off`, and the manager claims nothing. The next acknowledged heartbeat ends the cut-off. |
| 600 s (`WORKER_LIVE_S`) | any worker, `settle_dead_workers()` | The worker is dead, and its leases expired. The sweep settles its running jobs ([Dead workers](#dead-workers)). |
| 600 s, for each row of the host | any worker, `settle_dead_workers()` | The host is lost. Each unfinished job of a key bound to the host fails ([Lost hosts](#lost-hosts)). |

So a cut-off worker stops its children 60 s before another worker can run their jobs. These rules
apply to the row of the worker in `rs_workers`:

- The row records the host, the boot ID, PID and start time of the worker process, and the grade
  checksum pairs.
- At its start, the manager calls `Store.register_worker()` with a new session. If the recorded
  process is gone from this host, the new session takes the row at once. If another session holds a
  live row, `start` exits with `Worker <id> is live in another session`.
- When the manager stops, `Store.worker_stopped()` marks the row stopped and records the stop as its
  last heartbeat. The worker is then dead at once, but its host is lost only 600 s later.

### Dead workers

Before each claim, `WorkerManager` calls `Store.settle_dead_workers()`. This sweep takes each
running job whose worker is dead and whose lease expired, and ends its attempt as `lost`. If the host
of the job's key is lost, the job fails ([Lost hosts](#lost-hosts)). Otherwise the job runs again
from its spec on a worker of the same host, as after `finish(rerun=True)`. A worker of another host
never runs it, because its key is bound to the dead worker's host. A worker that starts again under
the dead worker's id settles it at once ([A job that its worker left](#a-job-that-its-worker-left)).

### Lost hosts

A host is lost when no row of that host in `rs_workers` had a heartbeat in the last 600 s, whether
its workers died or stopped ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). Then each
sweep fails each unfinished job of a key bound to the host. A queued job fails at once, and a running
job fails once its lease expired. The job ends `failed`, with the phase `data_lost`. Its result has
the `reason` `data_lost` and the `host`. These rules follow:

- If a worker of the host starts within 600 s of the host's last heartbeat or stop, the host is not
  lost.
- A shutdown releases the running jobs, so a planned reboot saves no progress of a running attempt.
  A reboot removes the kept sandboxes. A reboot of more than 600 s loses the host.
- The key stays bound, so each later job of the key fails as `data_lost` too. The client runs the
  work again under a new key ([evaluation](EVALUATION.md#a-lost-host)).

### A job that its worker left

A worker that starts again under its own id settles each running job of its id that no attempt of it
runs. It does this again once a minute, for example for a job whose `finish()` could not reach the
database. If the attempt froze no payload, no child started, and `Worker.settle()` releases the job.
Otherwise it stops the recorded child process group, matched by boot ID, PID and start time. Then it
publishes the attempt's `outcome.json`, or ends the attempt as lost if there is none.

### Frozen payload

The worker freezes the payload of each attempt in `rs_attempts.payload`, with its hash. The payload
holds the effective spec, the profile and the recovery point of a branch. The worker records the
child's identity, then sends it the payload on stdin. The child never connects to PostgreSQL.
`receive_payload()` in `src/sprout/runstore/payload.py` checks the size (at most 16 MiB), the hash,
the job and the attempt. If a check fails or the payload is late, the child exits without an outcome.

### Reruns

A job runs at most `max_infra_retries` + 1 times. `max_infra_retries` is 0 to 2, and the default
is 2. The worker decides by the `failure_kind` of the result:

| `failure_kind` | Examples | What the worker does |
|---|---|---|
| `infrastructure` | A 502, 503 or 504, a closed transport, an uncertain execution, a failed final snapshot, a child without an outcome, `worker_died`, a changed grade file, a lost attempt. | Runs the job again while attempts remain. Then the job fails. |
| `actor` | The agent's own error, a timeout, a step limit or an exhausted budget. | The job fails. |
| `stopped` | A stop request. | The job fails. |

A job that the worker cannot start, for example after its profile was removed, fails before its
child starts, and does not run again. Before a job runs again, the worker deletes the data of the
lost attempt:

1. It removes the attempt's sandboxes, or records them for the clean-up.
2. It deletes the snapshots that the journal names, and then the attempt directory. The attempt's
   `error` keeps the last 4000 characters of its `worker.log`.
3. `finish(rerun=True)` deletes the attempt's events, tool calls and recovery points.

So a branch from a point of the lost attempt fails. A rerun loses the progress of the lost attempt.
If failures cut long trajectories more often, a trainer sees shorter trajectories. A grader failure
runs only the grade job again, not its rollout.

### Supervision

`WorkerManager` runs each attempt in a thread. A `Watchdog` stops the child's process group with one
of these stop reasons:

- `timeout`: the attempt's deadline passed ([Timeouts](#timeouts)).
- `stopped`: a client requested a stop ([Stopping](#stopping)).
- `worker_shutdown`: the worker shuts down.
- `worker_cut_off`: the worker is cut off ([Workers and heartbeats](#workers-and-heartbeats)).

A child also stops when its worker process dies. About once a second, the child checks that its
parent is still the worker. If not, the child sends itself SIGTERM. Its outcome then has the stop
reason `worker_died`, an infrastructure failure. If the teardown has not ended 60 s later, the child
writes the outcome and exits with the status 3.

On SIGINT or SIGTERM, the manager stops claiming, and the watchdogs stop the children. The manager
waits up to 20 s. In that time, the worker removes the sandboxes and the data of each stopped
attempt, and releases its job. Then the manager marks the worker's row stopped. An attempt that does
not end within the 20 s stays `running`, until a sweep or a restarted worker settles it.

Once a minute, `Worker.clean_up()` deletes the local data of each lost attempt of the worker on this
host. It also frees each sandbox in `<artifact_root>/leftover-sandboxes.jsonl`. A sandbox start that
no claim names is only logged. Free such a sandbox with
`python -m sprout.worker reap --include-unknown --older-than <age>`.

### Stopping

`POST /v1/jobs/{id}/stop` cancels a `queued` job, and sets `stop_requested_at` on a `running` job.
The worker reads the stop request at its next progress write, at most 5 s later. It stops the
attempt's process and removes the sandboxes. Then it finishes the job `failed`, with the
`failure_kind` and the `stop_reason` `stopped`. The worker never runs a stopped job again, but it
salvages a stopped message-export rollout that has a valid recovery point ([Salvage](#salvage)).

## Validation

Run `python3 -m pytest tests/runstore tests/worker -q`. The database tests skip unless
`SPROUT_RUNSTORE_TEST_DSN` names an isolated PostgreSQL database. Each test makes a temporary schema
in it. Do not point the variable at a production database.
