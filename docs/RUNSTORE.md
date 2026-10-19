# Run Store

The Run Store is the job queue of Sprout: every rollout, grade and review runs as one of its
jobs. It is an authenticated HTTP API over PostgreSQL. A worker on the same host claims each job
and runs each attempt of it in a child process. Each worker reports a heartbeat every 30 s. A job
whose attempt is lost, for example with its worker, runs again from the start of its spec. The jobs
of one affinity key run on one host. When no worker of that host reported a heartbeat for 600 s,
because its workers died or stopped, its unfinished jobs fail as `data_lost`
([Claims](#claims), [Lost hosts](#lost-hosts)).

A job has one of three kinds:

- `rollout`: one agent run through `sprout.worker.run.runner.Runner`, with the agent that the
  request names ([Agents](#agents))
- `grade`: the verdict of a benchmark on one rollout, in a fresh sandbox or in the sandbox that
  the rollout kept ([Official grading](#official-grading))
- `review`: hindsight over finished attempts, against the model endpoint and without a sandbox
  ([Review](#review))

`rl_driver` submits its Miles rollouts here. `eval.evaluate` and `eval.gate` submit the jobs of
their Harbor tasks here ([evaluation](EVALUATION.md)). Read [Start](#start) to run the API and a
worker, and [Submit and observe](#submit-and-observe) to queue jobs. Read
[Failure handling](#failure-handling) before you investigate a job that did not succeed.

## Job flow

A job goes from a client through PostgreSQL to one worker, and the worker writes the result back.
The client and the worker never talk to each other. Each one talks only to the API or to the
database. The figure shows the calls of one attempt in the order that they occur.

<p align="center">
  <img src="figures/runstore-job-flow.svg" width="1080" alt="A sequence diagram of one Run Store job with seven lifelines: client, Run Store API, PostgreSQL, WorkerManager, Worker, attempt child and sandbox. WorkerManager and Worker are inside one box, the sprout.worker start process. 1: the client sends POST /v1/jobs to the API. 2: the API calls Store.submit(), and the job is queued. 3: WorkerManager calls Store.claim(), and the job is running with a lease of 600 s, which the heartbeats of WorkerManager renew. 4: WorkerManager calls run_claimed() on Worker. 5: Worker calls freeze_payload(). 6: Worker starts the attempt child and sends the payload on stdin. In a loop until the child exits, 7: the child sends tool calls to the sandbox, and 8: Worker calls progress() at most every 5 s. 9: the child leaves outcome.json and the journal for Worker, a file relation drawn dashed. 10: Worker calls _cleanup() to check that each claimed sandbox is gone. 11: Worker calls finish(), and the job is settled. 12: the client calls Client.wait(), which polls the result."/>
</p>
<p align="center"><sub>One attempt of a job, from <code>POST /v1/jobs</code> to the settled state, with the calls numbered in order.</sub></p>

These steps occur for each attempt:

1. The client calls `Client.submit()`. It sends `POST /v1/jobs` with an `Idempotency-Key` header.
2. The API validates the request and merges the defaults of the request's profile into it. Then
   the API calls `Store.submit()`. The job is a row of `rs_jobs`, in the state `queued`.
3. When `WorkerManager` has a free slot, it calls `Store.claim()`. The claim takes only a job that
   the worker can run on its host ([Claims](#claims)). The job changes to `running`. The claim adds
   an attempt to `rs_attempts` and gives the attempt a lease of 600 s. The heartbeats of the
   worker renew the lease ([Workers and heartbeats](#workers-and-heartbeats)).
4. `WorkerManager` calls `Worker.run_claimed()` on a thread for the attempt.
5. The worker merges the profile defaults again and checks the backend. Then it freezes the
   attempt payload with `Store.freeze_payload()`.
6. The worker starts the attempt child, a process of the module `sprout.worker.jobs.rollout`. The
   child writes its files in the attempt directory, `<artifact_root>/<job id>/<attempt id>/`. The
   worker records the process identity of the child with `Store.progress()`. Then it sends the
   payload to the child on stdin.
7. The child runs the rollout, the grade or the review. A rollout and a grade run commands in
   sandboxes. A review sends requests to the model endpoint only.
8. While the child runs, the worker calls `Store.progress()` at most every 5 s. Each call records
   the phase and reads a stop request. The worker also copies the new journal events to
   `rs_events`, and it indexes the tool calls and the recovery points.
9. The child writes `outcome.json` in the attempt directory, beside its journal
   `trajectory.jsonl`. Then the child exits.
10. The worker reads the outcome. It checks that each sandbox that the attempt claimed is gone
    (`_cleanup()`).
11. The worker calls `Store.finish()`. The job goes to one of these states
    ([Job states](#job-states)):
    - `succeeded`
    - `failed`
    - `queued`, to run again
12. The client calls `Client.wait()`, which polls `GET /v1/jobs/{id}/result`. The poll ends when
    `ready` is true.

## Boundaries

The Run Store runs on one host, for one trusted operator. It runs each sandbox on a backend whose
cleanup it can confirm. The subsections give these limits, and what the Run Store keeps.

### One trusted operator, one host

One operator runs the API and the workers on one host. They share one database and one artifact
directory. Each client submits tasks that run code. So keep the listener on `127.0.0.1`, or put it
behind an authenticated proxy.

### Owned sandboxes: podman by default, or microVMs

A rollout request must give a `sandbox_image`. The `backend` of the effective spec selects where an
attempt runs, the same way as for `python -m sprout.worker run`. The effective spec is the request
`spec`, merged over the defaults of its profile. The backends are these:

- `podman`, when the backend names no backend or names `podman`. The sandbox is a rootless
  container. Its rootfs is a subvolume of the btrfs `store` that `ops/setup_podman.sh` makes.
- `microvm`, when the backend names it. The sandbox is an AgentENV microVM.

The worker refuses every other backend, because it can confirm cleanup only on these two. Before a
rollout or grade attempt starts, `owned_backend()` reads the backend from the effective spec. If
the backend is not `podman` or `microvm`, or a rollout has no `sandbox_image`, the worker raises
`ValueError`. The attempt child does not start, and the job fails.

After each attempt, the worker checks that each sandbox that the attempt claimed is gone. It skips
a sandbox that a rollout keeps for a grade ([Kept sandboxes](#kept-sandboxes)). The check depends
on the backend:

- On `podman`, the worker looks for these leftovers:
  - the container
  - the network
  - the rootfs subvolume

  If a leftover remains, the worker removes it the same way `python -m sprout.worker reap` does.
  Then the worker looks again.
- On `microvm`, the worker deletes the sandbox. AgentENV must then return 404 for it. If the
  `microvm` section names no `server_url`, the worker cannot confirm cleanup.

The worker also reads the resource ledger of the attempt, `resources.jsonl`. If the ledger records
a sandbox start that no claim answers, the worker cannot confirm cleanup. If the worker cannot
confirm that a sandbox is gone, it still publishes the outcome. It records the sandbox for its
clean-up, which frees it later ([Supervision](#supervision)).

The worker and its attempt children use the same podman binary. The worker passes
`SPROUT_PODMAN_BIN` and `SPROUT_PODMAN_STORE` to each child. The worker takes the binary from the
first of these sources that is set:

1. `podman_bin` in the `podman` section of the effective `backend`
2. the `SPROUT_PODMAN_BIN` environment variable. When the services start the worker, the
   services' `podman` check sets this variable.
3. the podman that `ops/setup_podman.sh` installed, if it is executable
4. `podman` on `PATH`

After you change the podman, restart the worker.

The worker sets the run ID, the journal and `cwd` of each rollout itself. When a client submits a
rollout request, the Run Store rejects it if it sets one of these fields:

- `session`
- `journal_path`
- `run_id`
- `cwd`, to a value other than `.`
- one of these native-restoration keys in `extra`:
  - `native_prefix`
  - `native_home`
  - `checkpoint_identity`

The Run Store also rejects a rollout request in these cases:

- The request has a field that `RunSpec` does not have.
- `prompt` is empty.
- `slot` is not `mini-swe-agent`.
- `agent` names none of the agents in [Agents](#agents).
- `agent` is `oracle`, and `harbor_task` is not set.
- `keep_sandbox` is not a boolean.
- `extra` is not a JSON object.

### Agents

The `agent` of a rollout says what runs in its sandbox:

- `mini`, the default: the model, in the slot `mini-swe-agent`
- `oracle`: the `solution/solve.sh` of the Harbor task directory `harbor_task`, which the request
  must name
- `nop`: nothing

An `oracle` run does what Harbor's `oracle` agent does:

1. It writes each file of the task's `solution/` to `/solution` in the sandbox.
2. It runs `/solution/solve.sh` with its output in `/logs/agent/oracle.txt`. The timeout is the
   task's agent timeout, and at most the run's `timeout_s`.

The run completes whatever exit code `solve.sh` returns. If `solve.sh` outruns the timeout, the
run times out. With `image_user` ([Image USER and ENV](#image-user-and-env)), root first makes
`solve.sh` executable and `/logs/agent` writable to all. Then the image's `USER` runs `solve.sh`.

Every agent runs through the same runner, so each run has a journal and ends with a final disk
snapshot. A rollout that names `harbor_task` starts its sandbox on the task's agent network
policy, as Harbor's agent phase does. It does not use the `allow_internet` of its profile's
backend.

### Kept sandboxes

A rollout with `keep_sandbox: true` keeps its sandbox after the run, so that a `harbor` grade can
run in it ([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)). It keeps the sandbox for at most
`KEEP_SANDBOX_S`, 1200 s (20 minutes). The worker does these steps:

1. The run takes its final disk snapshot. A run without a final snapshot keeps no sandbox for a
   grade.
2. If the attempt completes or its result is truncated ([Training export](#training-export)), the
   result carries `"kept_sandbox": {"sandbox_id": ..., "keep_until": <ISO-8601 UTC>}`. The worker
   records the sandbox in `<artifact_root>/kept-sandboxes.jsonl`. It records the sandbox before
   the cleanup check, so that the check leaves the sandbox.
3. Otherwise, the worker removes `kept_sandbox` from the result. The cleanup after the attempt
   removes the sandbox.
4. A `harbor` grade with `in_place_job` takes the sandbox over. Its attempt claims the sandbox,
   and the cleanup after that attempt removes it. If the attempt ends before its child starts, the
   worker records the sandbox again, with the same `keep_until`. The next attempt of the grade can
   then take it over.
5. Once a minute, the worker removes each recorded sandbox that no grade took and whose
   `keep_until` has passed.

To see the removal plan of step 5, run this command:

```bash
python3 -m sprout.worker reap --ledger <artifact_root>/kept-sandboxes.jsonl --dry-run
```

The command prints each sandbox that it would free. Without `--dry-run`, it also frees them. A
microVM's lease covers the run's `timeout_s` plus `KEEP_SANDBOX_S` ([Timeouts](#timeouts)).

### Recovery points are closed messages

A branch starts only from a point where the agent's history and its disk agree. So a recovery
point has these parts:

- a finished model response
- all the tool results of that response
- an exact snapshot of the sandbox
- the native prefix of the agent's history, pinned by a checksum

A response with two tool calls exposes one point, after its last result. A missing or unsafe
capture exposes no point.

If a call submits the task, mini does not run the later calls of the same response. Each later
call gets a tool result that starts with `Not executed`, and mini records a `tool.skipped` event
for it in the journal. The response still exposes one point, after its last result. The snapshot
of that point holds the disk as the submitting call left it.

Snapshots are disk-only. A snapshot does not keep these things:

- processes
- memory
- extra mounts
- side effects

### Storage

PostgreSQL holds the state of the jobs, and the disk holds the large files. Back up both together.
The schema is `resources/runstore/schema.sql`. Its tables hold these records:

| Table | Records |
|---|---|
| `rs_workers` | workers, their hosts, sessions, session processes, grade checksums, last heartbeats and stops |
| `rs_jobs` | requests, affinity keys, states, leases, results |
| `rs_affinity` | the host that each affinity key is bound to |
| `rs_attempts` | attempts, frozen payloads, execution records, the deletion of a lost attempt's local data |
| `rs_events` | the journal events of each attempt |
| `rs_tools`, `rs_prefix_nodes` | the tool calls, their responses and their prefix DAG |
| `rs_recoveries` | recovery points |

The disk holds the native logs in the attempt directories under `artifact_root`, and the
snapshots. On podman, the snapshots are in the store's `snapshots/`. On a microVM, they are
AgentENV's layers. The Run Store and the worker keep the data of each attempt, except the data of
an attempt whose job ran again or that a shutdown released ([Reruns](#reruns)). Nothing replays from
a cache.

## Start

The Run Store needs a dedicated, persistent PostgreSQL database. The services do every step of the
procedure below with one command:

```bash
python -m sprout.services up runstore-api runstore-worker
```

The services read `~/.config/sprout/runstore.json`, and they do these steps
([services](SERVICES.md)):

1. They start the PostgreSQL container.
2. They set the DSN and the token.
3. They run `init`, then `serve`.
4. They run `python -m sprout.worker start`.

To start the Run Store by hand, do these steps from the repository root:

1. Install the extras:

   ```bash
   python3 -m pip install -e ".[mini,runstore]"
   ```

2. Copy `resources/runstore/config.example.json` to `<config>`. Replace each `/path/to/` path in
   it with a path on this host.
3. Set the database and the token:

   ```bash
   export SPROUT_RUNSTORE_DSN='host=127.0.0.1 dbname=sprout_runstore user=sprout_runstore'
   export SPROUT_RUNSTORE_TOKEN="$(openssl rand -hex 32)"
   ```

4. Create the schema:

   ```bash
   python3 -m sprout.runstore init --config <config>
   ```

5. Start the API. It listens on `127.0.0.1:18110` unless you give `--host` and `--port`:

   ```bash
   python3 -m sprout.runstore serve --config <config>
   ```

6. In a second shell with the same `SPROUT_RUNSTORE_DSN`, start a worker:

   ```bash
   python3 -m sprout.worker start --config <config> --concurrency 8
   ```

These rules apply to the commands:

- `init` is idempotent.
- `sprout.worker start` and each `sprout.runstore` command refuse a volatile `artifact_root`, for
  example a directory under `/tmp`.
- `sprout.worker start` checks the profiles of the config before the worker registers
  ([Checks at the start](#checks-at-the-start)). If a check fails, `start` exits with the reason.
- One worker runs at most `--concurrency` attempts at a time. The default is 1.
- Two or more workers can share one database. The lease in the database gives each job to one
  worker at a time.
- Each worker on one host needs its own `--worker-name`. The worker id is `<host>/<name>`, and the
  default name is `default`. A worker that starts again under its id first settles the jobs that it
  left ([A job that its worker left](#a-job-that-its-worker-left)). While the earlier session of the
  id reported a heartbeat in the last 600 s, and its process still runs on this host, `start` exits
  with `Worker <id> is live in another session` ([Workers and heartbeats](#workers-and-heartbeats)).
- With `--once`, the worker claims one job. It waits for the attempts that it started, and then it
  exits.

Pre-pull the task images before you start parallel attempts:

1. Run `python ops/prepull_images.py --tasks-dir <tasks dir> --workers 4`.
2. Start the attempts after the script finishes.

On podman, the script extracts the base of each image into the store one time. Without it, each of
the first attempts extracts the base itself. On a microVM, the script builds the template of each
image. If several attempts build the same template at the same time, the builds race.

The example config has these profiles:

| Profile | Section | Backend |
|---|---|---|
| `mini` | `run_defaults` | podman |
| `grade` | `grade_defaults` | podman |
| `mini-microvm` | `run_defaults` | microvm, with `image_env` |
| `grade-microvm` | `grade_defaults` | microvm, with `image_env` |
| `review` | `worker_env` only | none |

The podman profiles name the store `/path/to/sprout-btrfs`. `ops/setup_podman.sh` makes the store
`~/sprout-btrfs` one time, with `sudo ops/btrfs_store.sh`.

### Config

The config file holds `artifact_root` and the named `profiles`. Each request names its profile in
`profile`. A profile can have these keys:

| Key | What it does |
|---|---|
| `python` | The interpreter of the attempt child. The default is the worker's own interpreter. A wrapper script must `exec` the interpreter: the child stops when its parent is not the worker process ([Supervision](#supervision)). |
| `run_defaults` | The defaults of a `rollout` spec. |
| `grade_defaults` | The defaults of a `grade` spec. |
| `review_defaults` | The defaults of a `review` spec. |
| `env` | The variables that the attempt child exports and that the agent gets. |
| `worker_env` | The variables that the attempt child exports for itself only, for example the upstream keys of the gateway. |
| `snapshot_repository` | The on-disk snapshot repository of a microVM profile. The snapshot check also reads the manifests there. |

The API merges the defaults of the job's kind under the request `spec` at submission. Nested
objects merge key by key. The worker merges them again before each attempt.

Credentials follow these rules:

- The config and the requests must not hold an inline credential. An inline credential is a
  nonempty value under a key that names a secret, such as `api_key` or `password`.
  `read_config()` and the API reject one. Write `{"$env": "VAR"}` instead.
- The attempt child reads each `$env` variable from the worker's environment.
- The agent gets each variable that a `$env` reference names, and each `worker_env` variable, as
  an empty value, unless `env` sets it.

A job carries no digest of its profile
([ADR-0006](decisions/0006-worker-heartbeat-and-rerun-from-scratch.md)). The API freezes the
defaults into the request `spec` at submission, so a later edit of the profile does not change a
value that the spec holds. The API and each worker read the config when they start. A worker that
starts with an edited profile uses these values of the profile for each job, also for a job
submitted before the edit:

- each default that the spec does not hold
- `python`, `env`, `worker_env` and `snapshot_repository`

A request that an older Run Store stored with a `profile_hash` key loses the key at the next
`python3 -m sprout.runstore init`. Its `request_hash` still covers the key, so the Run Store
refuses a resubmission of the request under its idempotency key with 409.

When a rollout's `extra.rollout_contract` is set, the runner starts a gateway. The gateway routes
the agent's model requests to the contract's `model_endpoint`. The contract's `api_key_env` names
the variable that holds the provider key. Put that variable in `worker_env`. If the variable is
unset or empty, the rollout fails before its agent starts. A rollout with
`budget_usd` and no contract also gets a gateway. That gateway has no route, and it refuses each
model request.

### Checks at the start

A worker checks its configuration when it starts, and the API checks each request when a client
submits it. So no job fails on the configuration of its worker
([ADR-0006](decisions/0006-worker-heartbeat-and-rerun-from-scratch.md)).

Before the worker registers, `python -m sprout.worker start` checks each profile of its config
(`checked_grades()` in `src/sprout/worker/profiles.py`). A profile that fails a check would fail
each of its jobs, so `start` exits with `Profile <name>: <reason>`, and the worker claims nothing.
The checks are these:

- Each key of `run_defaults`, `grade_defaults` and `review_defaults` is a field of `RunSpec`,
  `GradeSpec` or `ReviewSpec`.
- The `backend` of `run_defaults` and of `grade_defaults` names `podman`, `microvm` or no backend.
- A grade profile names the four keys of its grade files, or none of them
  ([Grade files](#grade-files)).
- Each grade file that a profile names has the checksum that the profile gives.

At submission, the API checks each request. It answers these statuses:

- 422, with the reason, for a request that `JobSpec.validate()` in `src/sprout/runstore/specs.py`
  refuses, or that names a profile that the config does not have.
- 404 for a request that names an unknown job or recovery point ([Affinity keys](#affinity-keys)).

### Grade files

A `swe-rebench-v2` grade reads a dataset and a parser file on the worker's host. A grade profile
names the two files in its `grade_defaults`, each with its checksum:

| Key | Value |
|---|---|
| `dataset_path` | The JSON or JSONL dataset. |
| `dataset_sha256` | The SHA-256 of the dataset file, in hex. |
| `parser_path` | The parser file, the upstream `lib/agent/log_parsers.py`. |
| `grader_revision` | `sha256:` and the SHA-256 of the parser file, in hex. |

This is an example of a grade profile:

```json
"rebench-grade": {
  "python": "/path/to/sprout/.venv/bin/python",
  "grade_defaults": {
    "dataset_path": "/path/to/tasks.jsonl",
    "dataset_sha256": "<sha256 of tasks.jsonl>",
    "parser_path": "/path/to/log_parsers.py",
    "grader_revision": "sha256:<sha256 of log_parsers.py>"
  }
}
```

The worker registers the pair of `dataset_sha256` and `grader_revision` of each grade profile, in
`rs_workers.capabilities`. `Store.claim()` gives a `swe-rebench-v2` grade only to a worker that
registered the grade's pair ([Claims](#claims)). These rules apply:

- A grade's own `dataset_path` and `parser_path` must name files with the same checksums on the
  worker's host.
- A grade whose pair no live worker registered stays queued until its client stops it.
- A `harbor` grade needs no pair.
- A worker reads its grade files again only when it starts again.

### Image USER and ENV

The runtime runs every command as root. Two options of a backend section, `podman` or `microvm`,
run some commands the way Harbor runs them: as the image's `USER`, with the image's `ENV`. These
commands are:

- the agent's shell commands, which are mini's bash actions
- the `solution/solve.sh` of an `oracle` run
- the `tests/test.sh` of a `harbor` grade

Both options are off by default, so a rollout of the RL driver runs every command as root.
`eval.evaluate` and `eval.gate` turn both on in each job's `backend`, unless you pass
`--no-image-user` or `--no-image-env` ([evaluation](EVALUATION.md)). The worker merges a job's
`backend` over its profile's, so a job's value overrides the profile's value.

| Option | What it does |
|---|---|
| `image_user` | These commands run as the image's `USER` through `runuser`, or as root when the image names no `USER`. A podman sandbox reads the `USER` from the config that the store keeps beside it, also for a sandbox started from a snapshot. A microVM reads it with `regctl` from the registry. The image is the Harbor task's `docker_image`, or for a separate verifier the `FROM` image of `tests/Dockerfile`. Without a Harbor task, the image is the run's `sandbox_image`. |
| `image_env` | With `image_user`, each of these commands gets the image's `ENV` again after the switch of user. A podman container always starts with the image's `ENV`. So on podman, this option changes only the variables that `runuser` resets: `HOME`, `SHELL`, `USER` and `LOGNAME`. On a microVM, `image_env` also builds templates that start the runtime under the image's `ENV`. |

Every other command runs as root. These are examples:

- mini's probes of its working directory
- a grade's preparation of `/logs/verifier`, and its `[[verifier.collect]]` commands
- the replayed `RUN` steps of a separate verifier's `tests/Dockerfile`

With `image_user`, the image must hold the same tools that Harbor's command wrapper needs:

- `/bin/bash`
- `env`
- `getent`
- `runuser`

On a microVM, a branch of a run without a Harbor task fails with `image_user`. The branch starts
from a snapshot, and no OCI reference names its image.

A microVM template needs `image_env` to run the image's own tools, because AgentENV's guest agent
starts the runtime with its own `PATH`. The guest agent passes most of the image's `ENV` to the
runtime, but it rebuilds `PATH` from its defaults. On one DeepSWE image, `/opt/venv/bin` was not on
the runtime's `PATH`. So `pytest` was not found, and `python` was the system interpreter, not the
project's. Each of the 113 DeepSWE images had `PATH` entries that the runtime did not have. These
are examples:

- `/root/go/bin`, in 34 images
- `/app/node_modules/.bin`
- `/opt/venv/bin`

With `image_env`, the template builder reads the image's OCI config with `regctl`. The template
starts the runtime as `env KEY=VALUE ... sprout-runtime`, so each tool call gets the image's `ENV`.
The variables of the guest agent stay unless the image sets them, for example its `E2B_*`
variables and `HOME`. The template name includes a hash of the `ENV`. A template built without
`image_env` keeps its name and its behavior, so the SWE-bench templates of the reported results do
not change.

### Timeouts

Each limit below applies to every attempt of its kind:

| Limit | Value | Owner |
|---|---|---|
| microVM lease | `max(microvm.sandbox_ttl, ceil(<budget>) + 600)` s. The budget is the run's `timeout_s`, plus `KEEP_SANDBOX_S` for a kept sandbox. The lease bounds the sandbox, not the run. | `with_sandbox_budget()` |
| podman sandbox | No lease. The sandbox lives until the attempt, or the cleanup after it, removes it. | |
| rollout child | `timeout_s`, default 3600 s. It covers the child's startup and the actor. With a rollout contract, it is at most the time left before `deadline_at`. A message-export rollout gets 120 s more to write its export. | `Worker.run_claimed()` |
| grade child | `timeout_s` for the tests, plus 1200 s for the preparation and the collection | `Worker.run_claimed()` |
| review child | `timeout_s` for the whole review | `Worker.run_claimed()` |
| one `shell` call | 300 s by default. The run can stop the call first. | the runtime |

## Submit and observe

A client submits a `JobSpec` under an idempotency key, then waits for the result:

```python
import os
from sprout.runstore.client import Client
from sprout.runstore.specs import JobSpec

client = Client("http://127.0.0.1:18110", os.environ["SPROUT_RUNSTORE_TOKEN"])
job_id = client.submit(
    JobSpec(
        kind="rollout",
        profile="mini",
        context={"task_id": "task-1"},
        spec={
            "prompt": "The task description...",
            "model": "my-model",
            "sandbox_image": "my-prepared-image-or-snapshot",
            "timeout_s": 3600,
        },
    ),
    idempotency_key="experiment-1/task-1/parent",
)
print(client.wait(job_id))
```

These rules apply to each request:

- The same key with the same request returns the same job. The same key with a different request
  returns 409.
- `wait()` raises `TimeoutError` after its `timeout_s`. The timeout does not cancel the job.
- Each endpoint requires `Authorization: Bearer $SPROUT_RUNSTORE_TOKEN`. Without it, the API
  answers 401.
- These endpoints take `attempt_id`, whose default is the active attempt:
  - `events`
  - `tools`
  - `recovery-points`
- A request that contradicts the stored state gets 409. An invalid request gets 422. An unknown
  job, attempt or point gets 404.

| Endpoint | Result |
|---|---|
| `POST /v1/jobs` + `Idempotency-Key` | The accepted job (202) |
| `GET /v1/jobs?state=&limit=100` | Recent jobs, newest first. The limit is at most 1000. |
| `GET /v1/jobs/{id}` | The frozen request, the lease, the state, the result and the error |
| `GET /v1/jobs/{id}/attempts` | The attempts, their progress and `execution.scope` |
| `GET /v1/jobs/{id}/events?after=0&limit=1000` | The journal events after `seq`. The limit is at most 10000. |
| `GET /v1/jobs/{id}/tools?after=0&limit=1000` | The ordered tool calls and responses after `depth`. The limit is at most 10000. |
| `GET /v1/jobs/{id}/recovery-points` | The message boundaries, each with `available` |
| `GET /v1/jobs/{id}/result` | The result, with `ready`. `ready` is true for `succeeded`, `failed` and `cancelled`. |
| `POST /v1/jobs/{id}/branch` | A new queued branch from `point_id` ([Branch](#branch)) |
| `POST /v1/jobs/{id}/cancel` | Cancels a queued job. Any other job gets 409. |
| `POST /v1/jobs/{id}/stop` | Cancels a queued job, or stops a running one ([Stopping](#stopping)) |

## Branch

A branch is a new rollout job that starts from a recovery point of a finished rollout. To branch,
call `client.branch(job_id, point_id, idempotency_key=..., continuation=...)` with a point from
`client.recovery_points(job_id)` whose `available` is true. The branch starts from these parts of
the point and its parent:

- the point's snapshot
- the point's native prefix
- the parent's profile
- the parent's slot
- the parent's prompt
- the parent's environment
- the parent's tools

In `overrides`, a branch can change only these fields of the spec:

- `model`
- `timeout_s`
- `budget_usd`

The Run Store rejects the branch in these cases:

- The point does not belong to the parent rollout. The API answers 422.
- The point is no longer available. The API answers 409.
- `overrides` changes another field of the spec. The API answers 422.

`available` checks the prefix checksum and the snapshot again:

- On podman, the store must hold the snapshot directory `snapshots/<id>`. The check lists the
  directory, as `python -m ops.report snapshots` does.
- On a microVM, AgentENV must know the snapshot. With `snapshot_repository`, the on-disk manifest
  must also be complete.

The branch's `continuation` says what the actor gets after the restored prefix. It is required. It
has the same three modes as a SPROUT branch trial (`--branch-guidance`), and the same code applies
it (`src/sprout/contract/continuation.py`):

| `continuation` | What the actor gets after the prefix |
|---|---|
| `{"guidance": "assistant-turn", "assistant_turn": {role, content, tool_calls}}` | The turn runs through the normal tool path before the actor's first request. Its real results join the history. Its call IDs must be new to the history, which the API checks at submission. Its calls must fit the tools that the run offers, which mini checks. |
| `{"guidance": "user-hint", "hint": "..."}` | A user message with the hint, worded as for a Harbor branch. In message-export mode, the attempt child wraps the hint in `<sprout_training_hint>`, and the training export strips it. |
| `{"guidance": "none"}` | Nothing. The actor gets the prefix alone. |

An assistant-turn continuation can also set `assistant_turn_rendering`. It says how the turn
enters the history:

- `content`, the default, inserts the turn as written. The default is never stored in a body, so
  request hashes and idempotency keys stay as they were before the field existed.
- `reasoning` first checks the turn in its written form. Then it inserts
  `{"role": "assistant", "content": "", "reasoning_content": <its content>, "tool_calls": <its calls>}`
  (`render_reasoning`). This is the shape of a turn that a reasoning policy samples. So the actor
  continues after the same rendering that a trainer scores. The RL driver's repairs use it.

The other guidance modes reject `reasoning`. The branch's `fork.origin` records the rendering.

A branch inherits its parent's rollout contract. The contract's `deadline_at` is the parent
group's, which can already be past. Its `max_turns` counts again from zero in the branch. `budget`
sets the branch's own limits over that contract, as `{"max_turns": N, "deadline_at": T}` with one
key or both. The API rejects `budget` for a parent without a contract. If the deadline has passed
when a worker claims the branch, the attempt fails before its child starts.

A branch that must run only the turns that its parent had left at the point sets
`max_turns = <parent's max_turns> - <the point's message_step>`. The `message_step` of a point
counts each model turn up to and including the turn of the point. A rejected response is a model
turn too.

Each attempt of a branch starts at its `parent_point`, with its `continuation`. This is also true
for an attempt that runs the branch again ([Reruns](#reruns)).

## Review

A review job runs SPROUT's analyst and reviewer (`sprout.worker.jobs.review.branching`) over the
finished attempts of one task. Its caller, the RL driver or an evaluation
(`eval.evaluate --method sprout`), then branches where hindsight says that a continuation is worth
trying. The review runs no sandbox and needs no `backend`. Its profile section is
`review_defaults`, and the worker adds `artifact_root` to its spec. Queue a review as
`JobSpec(kind="review", spec=asdict(ReviewSpec(...)))`. A `ReviewSpec` has these fields:

| Field | Default | Meaning |
|---|---|---|
| `problem` | | The task's problem statement. |
| `model`, `model_endpoint` | | The model of the analyst and the reviewer, and its Chat Completions endpoint. The RL driver names the policy itself, under the review prompts. |
| `api_key_env` | | The variable of the worker's environment that holds the endpoint's bearer key, as `eval.evaluate --review-endpoint` and `--review-key-env` name them. A review without it fails. |
| `parents` | | The attempts to read. Each one names `name`, `job_id`, `attempt_id`, its grade job's result in `grade`, and optionally `hint`. |
| `guidance` | `assistant-turn` | What the review writes at each point: `assistant-turn`, `user-hint` or `none`. |
| `points` | 2 | The number of branch points to pick. |
| `points_per_attempt` | false | Pick `points` in each failed attempt, not `points` in total. |
| `count_mode` | `adaptive` | `adaptive` picks at most `points` points. `fixed` picks exactly `points` points. |
| `candidates` | 4 | The independent reviews at each point. A `user-hint` or `none` review must set 1. |
| `temperature` | 0.8 | The temperature of each request, from 0 to 2. |
| `max_output_tokens` | 16384 | The output budget of one request. |
| `prompt_chars` | 400000 | The size limit of one request ([Review requests](#review-requests)). |
| `transcript_tokens` | 100000 | The token budget of the analyst's transcript. |
| `reasoning_effort` | none | The `reasoning_effort` of each request. |
| `max_attempts` | 3 | The replies that the analyst and the reviewer get to give a valid report or plan. |
| `round_no` | 1 | The round, which the call IDs of the inserted turns name. |
| `timeout_s` | 3600 | The time limit of the whole review. |
| `request_timeout_s` | `timeout_s` | The time limit of one model request. |
| `max_concurrent_requests` | 8 | The most model requests in flight at once. The endpoint also serves the rollouts. |
| `structured` | false | The analyst, and an `assistant-turn` reviewer, ask for schema-constrained JSON (`response_format: json_schema`). |
| `prompts` | the packaged file | A prompts file on the worker host, as `{"path", "sha256"}`. |
| `leak_judge` | none | The Jev check of each reviewer-written turn ([Leak checks](#leak-checks)). |

The grade result of each parent must have a boolean `resolved`. The analyst reads these fields of
it:

- `resolved`
- `progress`
- `test_statuses`
- `reason`
- `error_output`
- the `model_patch` path
- the `log` path

The leak checks ([Leak checks](#leak-checks)) read these fields:

- `fail_to_pass`
- `test_patch_paths`
- the `test_patch` path

The review analyses each failed parent one time. Then it picks the points in one of two ways:

- Without `points_per_attempt`, one point review over all the analyses picks `points` branch
  points in total. A point that two attempts failed at can be picked one time.
- With `points_per_attempt`, one point review for each failed attempt picks up to `points` in that
  attempt.

If a plan has a count that `count_mode` does not allow, the review job sends the plan back to the
reviewer. The `guidance` says what the review writes at each point:

- `assistant-turn`: `candidates` independent reviews at each point, at `temperature`, each limited
  to that point, write the turn to insert there. Proposals with the same commands merge, and the
  merged candidate keeps their `multiplicity`.
- `user-hint`: the point review writes a hint at each point, in `hint`.
- `none`: nothing. The review picks the points alone.

If no parent failed, the review sends no request. Its result has no points and the `reason`
`every attempt resolved the task; nothing to repair`.

The result has these keys:

- `status`, which is `completed`, and `guidance`
- `points`: the points, each with the fields of the next table
- `reports`: each analysis, by parent name
- `analysed`: the names of the failed parents
- `review_requests`: the number of requests, retries included

Each point has these fields:

| Field | Content |
|---|---|
| `base` | The name of the parent that the point is in. |
| `job_id`, `attempt_id` | The job and the attempt of that parent. |
| `run_name` | The name of the branch run. |
| `step` | The tool step of the point. |
| `snapshot_id` | The snapshot of the point. |
| `conversation_cut` | The cut of the native conversation at the point. |
| `why` | The reviewer's reason. |
| `candidates` | The candidates at the point, for `assistant-turn` guidance ([Leak checks](#leak-checks)). |
| `samples` | The number of reviews at the point, for `assistant-turn` guidance. |
| `hint` | The hint at the point, for `user-hint` guidance. |

The caller maps `step` to the attempt's recovery point of that `tool_depth`, and submits the
branches.

The analyst and the reviewer read the packaged `resources/review/prompts.yaml`, or the file that
`prompts` names on the worker host. If that file no longer has the `sha256` of `prompts`, the
review fails. With `structured: true`, the analyst asks for schema-constrained JSON instead of a
fenced plain-text report. The reviewer asks for it only under `assistant-turn` guidance. A
`user-hint` reviewer answers in JSON without a schema, and a `none` reviewer in a labelled plan
fence. One model request can take `request_timeout_s`, and the whole review `timeout_s`.

### Calls of a review

A review is a set of calls, and a call starts as soon as the answers that it needs are in. The
figure shows which call waits for which.

<p align="center">
  <img src="figures/runstore-review-calls.svg" width="1080" alt="Two dependency graphs of the calls of one review over two failed attempts, with two points and two candidates. Top, one point review, the default: analyst-1 and analyst-2 each send their report to points, which sends point 1 to point1-candidate1 and point1-candidate2, and point 2 to point2-candidate1 and point2-candidate2. Bottom, with points_per_attempt: analyst-1 sends its report to points-1, which sends its point to point1.1-candidate1 and point1.1-candidate2. analyst-2 sends its report to points-2, which sends its point to point2.1-candidate1 and point2.1-candidate2. The two chains do not wait for each other."/>
</p>
<p align="center"><sub>The calls of a review over two failed attempts, with two candidates at each point, named as their receipt directories <code>reviews/&lt;name&gt;/</code> name them.</sub></p>

Each call has a stage and a name (`Call`). `<attempt>` is the position of the failed attempt,
from 1. The stages and names are these:

| Stage | What the call does | Name without `points_per_attempt` | Name with `points_per_attempt` |
|---|---|---|---|
| `ANALYSIS` | analyses one failed attempt | `analyst-<attempt>` | `analyst-<attempt>` |
| `POINTS` | picks the points | `points` | `points-<attempt>` |
| `CANDIDATE` | writes one candidate turn at a point | `point<point>-candidate<sample>` | `point<attempt>.<point>-candidate<sample>` |

The calls run side by side wherever no call needs another's answer. At most
`max_concurrent_requests` requests are in flight at a time. `Call` order sorts the calls by these
keys, in this order:

1. the stage
2. the failed attempt
3. the point
4. the sample

These rules give the order in which the calls start:

1. The analyses of the failed attempts run side by side.
2. Without `points_per_attempt`, the one point review waits for every analysis. Then the candidate
   samples at all its points run together.
3. With `points_per_attempt`, each failed attempt is a pipeline of its own. Its analysis feeds its
   point review, which feeds the candidate samples at its points. The pipelines run side by side.
4. With one slot, the calls run one after another in `Call` order.

The result has the points and samples in `Call` order, whatever the order in which the calls
finish. Each call has its own receipts directory, `reviews/<name>/`. Each point review and each
candidate call also has its own plan file, `plan-<name>.json`. An analysis has no plan file.

A call that raises fails the review. The review then fails with the error that a review with one
slot fails with, whatever the timing:

1. No call after the first failed call, in `Call` order, starts.
2. The running calls after it send no further request.
3. When every running call has returned, the review raises the first failure. The review notes
   the other failures on it.

When the worker stops a review (SIGTERM), the review stops every call and raises the interrupt at
once. Every model request is a receipt under the attempt directory, as for a SPROUT review.

An `assistant-turn` reviewer writes each turn flat: `content`, the assistant's text, and
`commands`, its bash commands in execution order. The review job then builds the tool calls, with
the IDs `reviewer-r<round>-b<branch>-c<call>`. So the actor receives an ordinary assistant message
(`sprout.worker.jobs.review.structured_review`). The flat form exists because the nested wire form
made the reviewer close `}}}` after each command, and it miscounted after long commands. In one
TerminalBench run, 23 of 79 reviewer replies were rejected, 19 of them for the brace count at the
end of a tool call. The model repeated the miscount on every retry. The review job still accepts a
turn in the nested `assistant_turn` form, so that a recorded plan replays.

### Leak checks

Each candidate is `{assistant_turn, why, multiplicity, leak_terms}`. `leak_terms`
(`src/sprout/worker/jobs/review/leak/terms.py`) lists the hidden test names and test file paths
that the candidate's turn mentions and that h_s never shows. The checked turn is the first of the
merged proposals, the one that the branch executes. The check reads these parts of that turn:

- its content
- its reasoning
- its commands

h_s is the history that a continuation from that point keeps. The check removes the hints from it,
as the training export does, and keeps the tool outputs.

The hidden vocabulary comes from two facts of the parent's grade:

- `test_patch_paths`: each path of 6 or more characters. Also the path's basename, when the
  basename is that long and is not a name that every repository has, such as `conftest.py`.
- `fail_to_pass`: each name of 8 or more characters. Also each `[A-Za-z0-9_]+` token of the name
  that is that long and is not all digits.

PASS_TO_PASS names are not terms. No analyst or reviewer prompt shows either fact (`grade_of`).
But the analyst reads the test log and the statuses, and the reviewer reads the analyst's report.
That is how a name reaches a turn. `leak_terms` is `[]` for a clean turn, and `null` when the
grade has neither fact. The check is lexical, and it only reports. The review returns every
candidate, and the caller executes it. The trainer decides whether to credit a candidate.

With `leak_judge` in the spec, Jev also judges every turn that a candidate review writes, for
content copied from the hidden tests. The judge reads the test patch
that the grade names. `leak_judge` has these optional keys:

- `threshold`, default 0.75
- `model`, default `jev-1.13.0`
- `resamples`, default 2
- `api_key_env`, default `TYPESAFE_API_KEY`: the worker's variable that holds the key

The judge runs before the review takes the candidate's plan. If either check flags a turn, the
turn goes back to the reviewer, up to `resamples` times. The reviewer learns only that the turn
states what the actor could not know. After the last return, the review takes the turn as it is.
Each candidate then also carries `leak_judge`: `{leak, leak_terms, jev}`. `leak` has one of these
values:

- true, when a check flagged the turn
- false, when both checks ran and neither flagged the turn
- `null`, when neither check flagged the turn but a check could not run

### Review requests

Each call of a review sends its analyst or reviewer requests through its own `ReviewTransport`
(`src/sprout/worker/jobs/review/transport.py`). The transport writes each try to a receipt
`review-NNN.json` in the call's `reviews/<name>/` directory. It numbers the receipts in the order
that the tries start. It writes the receipt before it sends the request, and again when the try
ends. A receipt holds these fields:

| Field | Content |
|---|---|
| `request` | The `model`, `prompt`, `max_tokens`, `reasoning_effort` and `temperature` of the try, and its `response_format` when it has one. |
| `started_at`, `timeout_s` | When the try started, and its time limit. |
| `finished_at`, `seconds` | When the try ended, and how long it took. |
| `response`, `text` | The reply, and its text. |
| `error` | The error of a failed try, instead of `response` and `text`. |
| `retry_of`, `retry_reason` | For a retry: the receipt of the try that it repeats, and why it repeats it. |

The transport numbers the tries under a lock. A retry names the earlier try of its own call, also
when several threads share one transport. The transport sends a request again in these cases:

1. The try timed out. The transport sends it one more time, with half the output budget. A plan is
   a few kilobytes, so a try that runs to the timeout spent its budget on reasoning. 64k output
   tokens took about 30 minutes at the rates seen. A smaller budget ends sooner, and the round
   still gets its review.
2. The endpoint answered with a status in `TRANSIENT_STATUSES`, or the connection dropped. The
   transport waits 10 s and tries again. Before the second retry it waits 30 s, and before the
   third 60 s. Each retry has the same budget. Without these retries, one 502 on a reviewer
   request fails the round.

`TRANSIENT_STATUSES` holds these statuses:

- 408 Request Timeout
- 429 Too Many Requests
- 500 Internal Server Error
- 502 Bad Gateway
- 503 Service Unavailable
- 504 Gateway Timeout

An HTTP 400 raises `PromptRejected`, because the endpoint refused the request itself. In practice,
the prompt was longer than the model's context. A reply with `finish_reason: length` raises
`ReviewResponseTruncated`. Its receipt keeps the reply in full, and the caller asks for a complete
reply without echoing it. After the review stops a call, each further try of that call raises
`RequestStopped` and sends nothing. A wait before a retry then ends at once. A request in flight
runs to its end.

**The size of a request.** The analyst and the reviewer hold each request to `prompt_chars`
characters (`DEFAULT_PROMPT_CHARS`, 400,000). At four characters per token, that is about 100k
prompt tokens. With 64k output tokens, it stays inside the model's context, also where JSON
tokenizes worse than prose. It is about half of the largest reviewer request known to succeed: 746k
characters, for a 136-step parent alone. It is a tenth of one that the endpoint refused with HTTP
400: 3.7M characters, for the same parent and four graded branches, each with the parent's prefix up
to its cut. With `structured`, the analyst and the reviewer subtract the size of the schema from the
budget.

When the endpoint refuses a request with HTTP 400, the analyst or the reviewer sends it again. The
new budget is half the old one, and less than the size of the refused prompt. The analyst or the
reviewer does this `PROMPT_HALVINGS` (2) times at most. The reviewer also stops at the last stage of
the reduction below.

**The analyst's transcript** (`render_transcript`) has one line for each tool step. The steps are
numbered as the checkpoints number them, one for each tool call in order. So each step that the
analyst names has a checkpoint. The renderer clips each line first, and only then the whole
transcript:

1. It clips each line to the cap of its kind:
   - a tool result to `RESULT_CHARS` (6000)
   - a call's arguments to `ARG_CHARS` (4000)
   - an agent message to `MESSAGE_CHARS` (2000)

   A clipped line keeps its head and its tail, because the verdict of a test run is at the end of
   its output.
2. If the transcript is still over budget, it loses its middle. The head keeps a third of the
   budget and the tail two thirds, because the late steps hold the failure under diagnosis. A
   token budget converts to characters at `CHARS_PER_TOKEN` (3.2), the pessimistic end of what
   these journals measured.

The caps for each line come from a real run. Tool results had a median of about 1.9k characters
and a maximum of about 17k. A cap of 300 characters kept the first two lines of each test run, and
it dropped the assertion that explains the failure. Arguments are small, with a median of about
112 characters. The argument that matters, a `str_replace` payload, is worth showing whole. The
middle of a transcript matters less: a 66-step run rendered to 34k characters.

The transcript gets the budget that remains after these parts of the prompt:

- the template
- the problem
- the verdict
- room for one feedback

When the remainder is less than a quarter of the budget, the renderer clips the verdict first.

**A reviewer request over budget** goes through `fit_review_prompt`, which applies one stage after
another (`STAGES`) until the request fits. It never reduces the analyst reports, because the
reviewer chooses by them. A window keeps these messages of a history:

- the messages around the cut of each analyst candidate
- the last messages
- the first two messages, which are the system and task prompts

| Stage | Parent history | Branch histories | Clip |
|---|---|---|---|
| `branch histories windowed` | whole | 12 before a cut, 4 after, the last 8 | none |
| `parent history windowed` | as the branches | 12 before, 4 after, the last 8 | none |
| `tool results, reasoning and arguments clipped to 6000` | as the branches | 12 before, 4 after, the last 8 | 6000 |
| `clipped to 2000` | as the branches | 12 before, 4 after, the last 8 | 2000 |
| `windows narrowed, clipped to 1000` | as the branches | 6 before, 2 after, the last 4 | 1000 |
| `reports alone` | no message | no message | none |

The branch histories go first, because a branch repeats the parent's prefix up to its cut. At the
floor, `reports alone`, each history is one marker. The prompt already has the task text, and the
actor's system prompt does not help to choose a point. Each omitted run of messages becomes one
marker that says how many messages it stands for and where. So `prefix_message_counts` stay exact.
The reviewer sends the last stage even when it does not fit.

**The plan file** `plan-<name>.json` records the whole exchange of a point review or a candidate
call. It holds these fields:

| Field | Content |
|---|---|
| `reports`, `available_steps` | The analyst reports that the reviewer read, and the eligible steps of each attempt. |
| `branch_limit`, `branch_count_mode`, `branch_guidance`, `reviewer_max_attempts` | The settings of the review. |
| `prompts` | The path and sha256 of the prompts file. |
| `structured_review` | Whether the request was schema-constrained. It is true for an assistant-turn review through a transport with `supports_json_schema`. |
| `structured_output` | With `structured_review`: the schema and its size. |
| `prompt_budget` | How the last request was fitted: its `stage`, `within_budget`, and the sizes and counts for each attempt. Also its `halvings`, and under `refused` the fits that the endpoint refused before it, oldest first. |
| `review_attempts` | Each request to the reviewer: its prompt and size, the reply, and its status or validation error. |
| `turn_resamples`, `turn_rejections`, `taken_flagged` | With a turn check: the limit of returns, the returns made, and the flagged plan that the review took. |
| `selected_branches` | The resolved points, each with its `turn_check`. |

A candidate call's review can choose only its point's step. So the candidates at one point are
independent proposals for that point. A plan with a turn that the check flags goes back to the
reviewer (`TURN_REJECTED`). This costs no reviewer attempt. After `resamples` such returns, the
review takes the plan as it is. The review also takes the last flagged plan when no later attempt
gives a plan (`taken_flagged`).

## Training export

A rollout whose `extra.rollout_contract` sets `message_export` returns its trajectory as training
messages. The RL driver reads them ([RL driver](RL_DRIVER.md)). When the attempt ends, the attempt
child adds the export to the result (`complete_message_result()`). The export marks every message
that no policy sampled.

First, the child marks the result truncated when all these conditions are true:

- `stop_reason` is `max_turns_reached` or `timeout`.
- The journal has no `execution.uncertain` event.
- The run has a final snapshot.
- The failure is not an `infrastructure` failure.

A truncated result has the status `truncated`, and it keeps the agent's own status in
`agent_status`. The worker settles a truncated result as `succeeded`. Then the child adds these
fields:

| Field | Content |
|---|---|
| `native_session_id` | The mini session that the messages come from. |
| `training_messages` | The native mini history, cleaned. The export removes the hints from the system and user messages, and changes `developer` to `system`. For a truncated result, it drops an unfinished last record. |
| `training_tools` | The Chat Completions tool schemas that the model was offered. |
| `training_origin` | For a branch: the `job_id`, `point_id` and `tool_depth` of the branch point. |
| `rollout_usage` | The `model_calls` and `tool_calls` of the run. |
| `training_provenance` | The messages that no policy sampled in this job: `prefix_messages`, the count of leading messages restored from the branch point, and `inserted_messages`, the indices of every inserted assistant turn of this job or of an ancestor. |

If the export fails, the result gets `training_export_error`. A truncated mini result without an
uncertain execution then falls back to its last recovery point, if it has one. The fallback has
these effects:

- `training_messages` end at that point.
- `final_snapshot_id` becomes the point's snapshot.
- `timeout_fallback` records these values:
  - the point's `tool_depth`
  - the point's `message_step`
  - the point's `native_prefix`
  - the `discarded_final_snapshot_id`
- The child removes `training_export_error` from the result.

If the export cannot tell which messages a policy sampled, `training_export_error` starts with
`training provenance:`.

### Salvage

When a message-export rollout fails and will not run again, the worker cuts it back to its last
valid recovery point (`salvage_interrupted()`). These are examples of such failures:

- the attempt crashed
- the attempt lost its sandbox connection
- the attempt ran an uncertain command
- someone stopped the attempt

The salvaged result has the messages up to the point, and the point's snapshot. Both come from
before the first call of the next turn ran. The job finishes `succeeded`, with the status
`truncated` and `stop_reason: "interrupted"`. Its `interrupted` field says what went wrong, and at
which point. The RL driver grades it as it grades a timeout. The worker does not salvage an attempt
that has no closed turn with an assistant message. Such an attempt stays failed.

## Official grading

A grade job gives the verdict of a benchmark on the work of one rollout. Queue it as
`JobSpec(kind="grade", profile="grade", spec=asdict(GradeSpec(...)), affinity=<key>)`. A grade from a
`snapshot_id` needs `<key>`, the affinity key of its rollout ([Affinity keys](#affinity-keys)). The
grade profile's `grade_defaults` supply the sandbox settings in `backend`. The backend is podman unless it names
microvm. An unresolved grade is a successful job, with `resolved: false`. `GradeSpec`
(`src/sprout/runstore/specs.py`) has these fields:

| Field | Default | Meaning |
|---|---|---|
| `benchmark` | | `harbor` or `swe-rebench-v2` |
| `instance_id` | | The task row in the dataset (`swe-rebench-v2`). |
| `snapshot_id` | | The submission: the final snapshot of the rollout. |
| `dataset_path`, `dataset_sha256` | | The JSON or JSONL dataset, pinned by SHA256 (`swe-rebench-v2`). |
| `grader_revision` | | `sha256:<hex>` of the parser file (`swe-rebench-v2`). |
| `parser_path` | | The parser file, the upstream `lib/agent/log_parsers.py` (`swe-rebench-v2`). |
| `image` | | The task's pristine image (`swe-rebench-v2`). |
| `task_dir` | | The Harbor task directory (`harbor`). |
| `in_place_job` | | The rollout job whose kept sandbox the grade runs in (`harbor`). |
| `backend` | `{}` | The sandbox settings. |
| `resources` | `{"cpu": 2, "memory_mb": 12288}` | The shape of each grading sandbox. |
| `timeout_s` | 3600 | The time limit of the tests. |
| `verifier_network` | `deny` | `deny` or `allow`. It becomes the backend's `allow_internet` for a `swe-rebench-v2` grade. |

A `harbor` grade runs the verifier of the Harbor task directory `task_dir`. It names exactly one
of these sources:

- `snapshot_id`: the grade runs in a fresh sandbox from that snapshot.
- `in_place_job`: the grade runs in the sandbox that this rollout job kept.

The Run Store accepts an `in_place_job` grade only when the rollout job has finished and its result
carries `kept_sandbox` with a UTC `keep_until`. Then these rules apply to the grade:

- Only the worker whose attempt kept the sandbox (`bound_worker`) claims it. The binding names the
  stable worker id, `<host>/<name>`.
- That worker claims it only before `keep_until`, and before any job that is not bound to it
  ([Claims](#claims)).
- If `keep_until` passes first, the next claim or sweep of any worker finishes the queued grade
  `succeeded`, with the result `{"reward": 0.0, "reason": "sandbox_lost"}`. This also applies when
  the host of the grade's affinity key is lost.
- If the bound worker dies or stops, the queued grade waits. A worker that starts again under the
  same id before `keep_until` claims it, and grades in the sandbox if the sandbox still exists.
- If a sweep finds the host of the grade's affinity key lost before `keep_until` passes, the grade
  fails as `data_lost`, as the other jobs of its key do ([Lost hosts](#lost-hosts)).
- If an attempt of the grade is lost, or fails for the infrastructure, and the grade has no attempt
  left, the grade finishes `succeeded`, with the result `sandbox_lost`. The grade has no other
  sandbox to run in. A grade of manual evaluation has no attempt after its first.
- The worker hands the grade's attempt the sandbox only if it still keeps it and `keep_until` has
  not passed. Otherwise, or when the sandbox does not answer, the grade gives the reward 0 with
  the reason `sandbox_lost`.

A `swe-rebench-v2` grade needs these fields:

- `instance_id`
- `snapshot_id`
- `dataset_path` and `dataset_sha256`
- `grader_revision` and `parser_path`
- `image`

It does not take `task_dir` or `in_place_job`.

### The `harbor` grader

The `harbor` grader (`src/sprout/worker/jobs/harbor/grade.py`) runs the task's verifier the way
Harbor 0.22.0 runs it ([ADR-0003](decisions/0003-sprout-runs-harbor-verifier.md)). It reads the
task with `sprout.contract.harbor_task.load_task`, and it does these steps:

1. It starts the agent's sandbox from `snapshot_id`, with the task's CPUs and memory over the
   grade's `resources`. With `in_place_job`, it attaches to the sandbox that the rollout kept
   instead. If the verifier shares the agent's environment, it then switches the sandbox's egress
   to the verifier's policy. A microVM cannot switch, so such a grade's attempt fails. The cleanup
   removes the sandbox, and a rerun gives 0 as `sandbox_lost`.
2. It runs the task's `[[verifier.collect]]` commands in that sandbox. Then it copies
   `/logs/artifacts` and the task's `artifacts` to the attempt's `artifacts/`. Both steps are best
   effort, as in Harbor.
3. If the verifier shares the agent's environment, it uploads `tests/` to `/tests` in that
   sandbox, and the verifier runs there. Otherwise, it destroys the agent's sandbox and starts a
   new sandbox from the `FROM` image of `tests/Dockerfile`. It replays the `COPY` and `RUN` steps
   of that file (`src/sprout/worker/jobs/harbor/overlay.py`, which `eval.harbor` uses too). Then
   it copies the artifacts into the new sandbox, at their own paths.
4. It empties `/logs/verifier`. It runs
   `(/tests/test.sh) > /logs/verifier/test-stdout.txt 2>&1` within the task's verifier timeout.
   Then it copies `/logs/verifier` to the attempt's `verifier/`.
5. The reward is the `reward` of `reward.json`, or else the number in `reward.txt`.

The task's network policies decide egress, not `verifier_network`. A sandbox that runs the verifier
starts on the verifier's policy. A separate verifier's agent sandbox starts on the agent's policy.

The result is `{"status": "completed", "reward", "reason", "verifier": {...}}`. `reason` is `null`
when the verifier wrote the reward. Otherwise the reward is 0, and `reason` is one of these:

- `sandbox_lost`
- `verifier_timed_out`
- `reward_missing`
- `reward_unreadable`, for one of these reward files:
  - an empty file
  - a file that holds no number, or NaN
  - a `reward.json` without a numeric `reward`

If the grader cannot make the grade, it raises an error. These are examples of the causes:

- a sandbox that does not start
- a file that does not move
- a `tests/Dockerfile` that a sandbox cannot replay

The attempt is then an infrastructure failure. The worker runs the grade again while attempts
remain, and then the job fails.

`tests/test.sh` runs as the image's `USER` when the grade's backend turns
`image_user` on ([Image USER and ENV](#image-user-and-env)), as in Harbor. Otherwise it runs as
root, the runtime's user, also on an image whose `USER` is not root.

### The `swe-rebench-v2` grader

The `swe-rebench-v2` grader (`src/sprout/worker/jobs/grade.py`) does what the official
`SWE-rebench-V2/scripts/eval.py` does, in two sandboxes. Nothing that the agent changed outside its
diff can reach the verdict. The grader does these steps:

1. It checks the SHA256 of the dataset against `dataset_sha256`. Then it selects the row of
   `instance_id`. If the dataset does not have exactly one such row, the grader raises an error.
2. It checks the parser file against `grader_revision`.
3. It starts a sandbox from the final snapshot, `snapshot_id`. There it writes the agent's diff
   against `base_commit` to `model.patch`, with `git add -A` and then `git diff --cached --binary`.
   The diff turns these settings off and fixes the `a/` and `b/` prefixes, so that the agent's
   repository config cannot change the diff:
   - external diff
   - textconv
   - colour
   - fsmonitor

   Then the grader destroys that sandbox.
4. It starts a pristine sandbox from `image`, and resets the repository in `/<repo name>` to
   `HEAD`.
5. It applies the agent's diff, if the diff is not empty, and then the row's `test_patch`. Each
   one goes through `git apply --3way`, as upstream. If git cannot apply one, a merge conflict
   included, the grade is 0 with the reason `model_patch_apply_failed` or `test_patch_apply_failed`.
6. It runs `install_config.test_cmd` in `/<repo name>` through the test script `eval.sh`. The
   script runs each command, and it exits 1 if one fails.
7. It parses the test log with the function that `install_config.log_parser` names in
   `parser_path`.

The task is resolved when every `FAIL_TO_PASS` and every `PASS_TO_PASS` test passed. A task
without expected tests is not resolved.

Every git command of the grader runs with `-c safe.directory='*'`. Without it, git refuses a
repository that another uid owns, and the refusal would score as the agent's fault.

No file travels inside a command. The runtime runs a command as one `sh -c` argument, which Linux
caps at `MAX_ARG_STRLEN` (128 KiB). The runtime reports the `E2BIG` of a longer one as a bare exit
1. When the patches were staged as base64 in the command, every grade of a task whose diff or test
patch passed about 96 KiB failed. Each sandbox gets a private directory,
`/tmp/sprout-grade-<uuid4 hex>`, made with `mkdir -m 700`. That command refuses a path that the
snapshot already holds. Each file of the grade moves through `upload_file` and `download_file`,
which use `podman cp` or AgentENV's file service. The attempt directory keeps these files:

- `model.patch`, the agent's diff
- `test.patch`, the row's test patch
- `eval.sh`, the test script
- `swerebench-tests.log`, the test log

A completed result carries these fields:

- `resolved`
- `progress`: the share of the `FAIL_TO_PASS` tests that passed. It is 0 when a `PASS_TO_PASS`
  test fails. A task without `FAIL_TO_PASS` tests gives its verdict, 1 or 0.
- `grader_revision`
- the `model_patch` and `test_patch` paths
- three facts about the task's hidden tests:
  - `test_patch_paths`: every path that the test patch names, old and new, sorted, without
    `/dev/null`
  - `touched_test_paths`: the paths of `test_patch_paths` that the agent's diff names too
  - `fail_to_pass`: the row's list

A grade whose tests finished, with exit 0 or 1, adds `expected_pass_share`, the share of all the
expected tests that passed. It also adds `test_statuses` and the `log` path.

A failure that the agent's work can cause is also a completed result. It has these fields:

- `resolved: false`
- `progress: 0.0`
- a `reason`
- an `error_output`

The reasons are these:

| `reason` | When |
|---|---|
| `model_patch_unavailable` | The snapshot's `git diff` exits non-zero, for example after the agent removed `.git` or the base commit. |
| `model_patch_apply_failed` | The agent's diff does not apply. |
| `test_patch_apply_failed` | The test patch does not apply. |
| `tests_timed_out` | The tests outran `timeout_s`. The result has the `log` so far, when the grader could fetch it. |
| `test_runner_killed` | The tests exited with a code other than 0 or 1, for example after an OOM kill. The result has the `log`. |

If the grader cannot make the grade, it raises an error. The attempt is then an infrastructure
failure, and the worker runs the grade again while attempts remain. Then the job fails. The grader
cannot make the grade in these cases:

- a sandbox that does not restore or start
- a file that does not move
- a parser that raises
- a dataset or a parser file whose checksum differs from the grade's

A worker claims a `swe-rebench-v2` grade only when one of its grade profiles names files with the
grade's checksums ([Grade files](#grade-files)). So a checksum differs only when a file changed
after the worker started, or when the grade's own path names another file. The grader checks both
checksums before it starts a sandbox.

## Claims

`Store.claim()` gives a worker the next queued job that the worker can run on its host. Only a live
worker claims: the worker needs a row in `rs_workers` with a heartbeat in the last 600 s
([Workers and heartbeats](#workers-and-heartbeats)). Otherwise `claim()` raises `Conflict`. A claim
takes a job only when all of these conditions hold:

- `ready_at` has passed, and `attempt_count` is less than `max_attempts`.
- The job's kind is one of the kinds that the worker runs.
- The job is not bound to another worker. A grade in a kept sandbox is bound to the worker that
  keeps the sandbox, until `keep_until` ([Official grading](#official-grading)).
- The affinity key of the job is unbound, or bound to the host of the worker
  ([Affinity keys](#affinity-keys)).
- A `swe-rebench-v2` grade has the `dataset_sha256` and the `grader_revision` of a pair that the
  worker registered ([Grade files](#grade-files)). A `harbor` grade needs no pair.

The claim takes these jobs in this order:

1. the jobs bound to the worker
2. the jobs whose affinity key is bound to the worker's host
3. the other jobs

In each of these groups, the job with the earliest `ready_at` comes first, then the job with the
earliest `created_at`. Before it selects a job, each claim finishes each queued bound grade past its
`keep_until` as `sandbox_lost`. Then the claim does these actions in one transaction:

1. It binds the affinity key of the job to the worker's host, if the key is unbound.
2. It adds a new attempt.
3. It adds 1 to `attempt_count`.
4. It sets a lease of 600 s.

Two claims on two hosts can select two jobs of one unbound key at the same time. Then the claim
that commits first binds the key. The other claim waits for that commit, takes no job and returns
None. Its job stays queued, and the next claim of the worker selects again.

### Affinity keys

Each job has an affinity key, so that the jobs of one group run on one host
([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). A rollout leaves its data on the host
that ran it: its snapshots, its attempt directory and its kept sandbox. A branch, a review and a
grade of the group read that data. The search coordinator sets the `affinity` field of each job of
a group to the group's key.

The Run Store stores the key of each job in `rs_jobs.affinity` when a client submits the job
(`Store.submit()`). A request with `affinity` keeps it, if the jobs that it names have that key or
none. A request without `affinity` takes a key in this order:

1. The key of the job that the request names:
   - for a branch, the job of its `parent_point`
   - for a grade with `in_place_job`, that rollout
   - for a review, the jobs of its `parents`, which must share one key
2. Else the id of the job itself.

A grade from a `snapshot_id` names no job, but its snapshot is on the host of its rollout. So it
must have `affinity`, the key of that rollout. The search coordinator sets it. The API refuses
these requests:

- a request that names an unknown job or point, with 404
- a grade from a `snapshot_id` without `affinity`, with 422
- a review whose parents have two or more keys, with 422, with or without `affinity`
- an `affinity` that differs from the key of a job that the request names, with 422
- an `affinity` that is not text of 1 to 256 characters, with 422

The first claim of a job of a key binds the key to the host of the claiming worker, in
`rs_affinity`. The host is the `host` of the worker's row, the node name of the worker's host. The
key stays bound after its jobs finish, and after its host is lost ([Lost hosts](#lost-hosts)). A
job stored before the column existed has its own id as its key.

## Failure handling

A job whose attempt is lost runs again from the start of its spec, never from a recovery point
([ADR-0006](decisions/0006-worker-heartbeat-and-rerun-from-scratch.md)). An attempt is lost when
its worker dies, is cut off or fails after the child started, or when the child leaves no outcome.
An infrastructure failure also runs the job again, and an actor failure does not. Before a rerun,
the worker deletes the data of the lost attempt. A heartbeat of each worker tells the Run Store
which workers live. When no worker of a host reported a heartbeat for 600 s, the unfinished jobs of
each affinity key bound to that host fail as `data_lost` ([Lost hosts](#lost-hosts)).

### Job states

A job has five states. The figure shows each transition and the function that makes it. No
transition leaves the three terminal states:

- `succeeded`
- `failed`
- `cancelled`

<p align="center">
  <img src="figures/runstore-job-states.svg" width="1080" alt="Five Run Store job states and the functions that move a job between them. Store.submit() creates a job in queued. queued goes to running through claim(), when ready_at has passed and attempts are left, the job's affinity key is unbound or bound to the worker's host, the job is not bound to another worker, and the worker holds the checksums of a swe-rebench-v2 grade. A queued bound grade goes to succeeded with the result sandbox_lost through settle_dead_workers() or claim() after its keep_until, also on a lost host. queued goes to failed with the phase data_lost through settle_dead_workers() when the host of its affinity key is lost, a bound grade too before its keep_until. queued goes to cancelled through cancel_queued() or request_stop(). running goes back to queued through finish(rerun=True) after an infrastructure failure or a lost attempt, through settle_dead_workers() when the worker is dead, the lease expired and the host of the job's key is not lost, and through release() when the worker shuts down. A rerun needs an attempt left, and none of the three goes back to queued after a stop request: then release() moves running to cancelled, and the other two to failed. running goes to succeeded through finish() when the attempt completed or was truncated. A bound grade without an attempt left goes from running to succeeded with the result sandbox_lost, through finish() or settle_dead_workers(). running goes to failed through finish() when the attempt failed and does not run again, and through settle_dead_workers() when no attempt is left, or with the phase data_lost when the host of the job's key is lost. succeeded, failed and cancelled are terminal."/>
</p>
<p align="center"><sub>The five job states of <code>rs_jobs</code>, with the function that makes each transition and its condition.</sub></p>

The phase of the job is in parentheses:

| From | Trigger and condition | Actor | To |
|---|---|---|---|
| none | A client submits a valid request under a new idempotency key. | API, `Store.submit()` | `queued` (`queued`) |
| `queued` | `ready_at` has passed and `attempt_count` is less than `max_attempts`. The worker can run the job on its host ([Claims](#claims)). | `WorkerManager`, `Store.claim()` | `running` (`claimed`) |
| `queued` | A bound grade's `keep_until` has passed. | any worker's next `Store.settle_dead_workers()` or `Store.claim()` | `succeeded` (`succeeded`), with the result `sandbox_lost` |
| `queued` | The host of the job's affinity key is lost. A grade in a kept sandbox also fails before its `keep_until`. | any worker's `Store.settle_dead_workers()` | `failed` (`data_lost`) |
| `queued` | A cancel request or a stop request arrives. | API, `cancel_queued()` or `request_stop()` | `cancelled` (`cancelled`) |
| `running` | The attempt completed or its result is truncated. | `Worker`, `finish()` | `succeeded` (`succeeded`) |
| `running` | The attempt failed, and the job does not run again. | `Worker`, `finish(state="failed")` | `failed` (`failed`) |
| `running` | The attempt is lost or had an infrastructure failure. Attempts are left, and nobody requested a stop. | `Worker`, `finish(rerun=True)` | `queued` (`retry_wait`) |
| `running` | The worker is dead, the lease expired, and the host of the job's key is not lost. Attempts are left, and nobody requested a stop. | any worker's `Store.settle_dead_workers()` | `queued` (`retry_wait`) |
| `running` | The worker is dead, the lease expired, and the host of the job's key is not lost. No attempt is left, or somebody requested a stop. | any worker's `Store.settle_dead_workers()` | `failed` (`failed`) |
| `running` | The worker is dead, the lease expired, and the host of the job's key is lost. | any worker's `Store.settle_dead_workers()` | `failed` (`data_lost`) |
| `running` | The worker shuts down, and nobody requested a stop. | `Worker`, `release()` | `queued` (`released`) |
| `running` | The worker shuts down, and somebody requested a stop. | `Worker`, `release()` | `cancelled` (`cancelled`) |

While the attempt runs, its progress writes set the phase in this order:

1. `starting`, when the child has started
2. `rollout` or `grading`, or the phase that the child reports in `progress.json`

These rules also apply:

- `claim()` selects and orders the jobs as [Claims](#claims) gives.
- A job that runs again waits 30 s × `attempt_count`. `finish()` and `settle_dead_workers()` set
  `ready_at` to that time.
- `finish(rerun=True)` is valid only for an `infrastructure` failure. Otherwise `finish()` raises
  `ValueError`.
- An attempt that runs again ends `lost`. The attempt of a job that fails because no attempt is
  left ends `failed`, or `lost` when its worker died. The attempt of a job that fails as
  `data_lost` ends `lost`.
- A job that fails because no attempt is left has `attempt_errors` in its result: the `error` of
  each attempt that counted, in the order of the attempts.
- `release()` does these actions in one transaction:
  - sets the state of the attempt to `released`
  - deletes the events, the tool calls and the recovery points of the attempt
  - subtracts 1 from `attempt_count`, so the attempt does not count against `max_attempts`
  - sets `ready_at` to the current time, so the job waits for no backoff
- A released attempt keeps its number. The next claim gives its attempt the next number.

### Workers and heartbeats

Each worker has a stable id and a row in `rs_workers`. The worker is live while its row is not
stopped and has a heartbeat that is less than 600 s old. Otherwise the worker is dead.
`WorkerManager` keeps the row and the leases of the worker's jobs in these steps:

1. Before its first claim, the manager calls `Store.register_worker()` with a new session token.
   The session takes the row. The row records these values:
   - the host of the worker
   - the identity of the worker process: its boot ID, PID and start time
   - the pairs of checksums of its grade profiles ([Grade files](#grade-files))

   The lease of each running job of the worker then ends 600 s later.
2. Every 30 s, a thread of the manager calls `Store.worker_heartbeat()`. The Run Store records the
   heartbeat, and moves the end of the lease of each running job of the worker to 600 s after it.
   Then it acknowledges the heartbeat. It does not record or acknowledge a heartbeat of a stopped
   row.
3. When the manager stops, it calls `Store.worker_stopped()`. The Run Store marks the row stopped,
   and records the time of the stop as the row's last heartbeat. The row stays.

The worker id is `<host>/<name>`. `<host>` is the node name of the host. `--worker-name` of
`python -m sprout.worker start` sets `<name>`, and the default is `default`. These rules apply to
the registration:

- Before it registers, the manager reads the row of its id (`Store.worker()`). If the row is of
  this host, and the process that the row records is gone from this host, the manager names the
  session of the row in `register_worker(replaces=...)`. The new session then takes the row at once.
  The process is gone in these cases:
  - The process exited.
  - Its PID names a process with another start time.
  - The host booted again, so the boot ID is different.
- If another session holds the row, the row is live, and the manager did not name that session,
  `register_worker()` raises `Conflict`. The Run Store also refuses a `replaces` from another
  host. `start` then exits with `Worker <id> is live in another session`.
- So a worker that crashed can register again at once on its own host. A row of another host, or
  a row that records no process, is free 600 s after its last heartbeat. A stopped row is free at
  once, so a worker that stopped can register again at once.
- A grade in a kept sandbox is bound to the worker id, so a worker that starts again under its
  id still claims it.

The manager measures each wait from the send time of the last acknowledged heartbeat, or of the
registration. These rules apply:

- Without an acknowledgment for 540 s, the worker is cut off. Each watchdog stops its child with
  the stop reason `worker_cut_off`, and the manager claims nothing. The next acknowledged
  heartbeat ends the cut-off.
- If an answer says that another session holds the row, the worker is cut off for good, and the
  manager stops.

A cut-off worker stops its children 60 s before another worker can run their jobs.

### Dead workers

Before each claim, `WorkerManager` calls `Store.settle_dead_workers()`. This sweep does these steps
in one transaction:

1. It finishes each queued bound grade whose `keep_until` has passed. The grade ends `succeeded`,
   with the result `sandbox_lost` ([Official grading](#official-grading)).
2. It fails each queued job whose affinity key is bound to a lost host as `data_lost`
   ([Lost hosts](#lost-hosts)).
3. It takes each running job whose worker is dead and whose lease has expired.

Step 1 comes before step 2. So a queued grade whose `keep_until` has passed scores 0, also on a
lost host. It does not fail as `data_lost`.

The sweep ends the active attempt of each running job of step 3 as `lost`, with the error
`Worker <id> stopped reporting heartbeats`. Then it settles the job:

- A job whose affinity key is bound to a lost host fails as `data_lost` ([Lost hosts](#lost-hosts)).
- Another job runs again from its spec, as after `finish(rerun=True)`. It fails if no attempt is
  left, or if somebody requested a stop. A grade in a kept sandbox stays bound to its worker.

The sweep does not settle a queued grade that is bound to a dead worker while its `keep_until` has
not passed and its host is not lost. The grade waits for a worker under the same id
([Official grading](#official-grading)).

The sweep deletes the events, the tool calls and the recovery points of each attempt that runs
again. The local data of the attempt stays on the host of the dead worker, until the worker lives
again ([Supervision](#supervision)). The data of a host that never returns stays on that host.

The jobs of a crashed worker wait up to 600 s after its last heartbeat before another worker of its
host runs them. A worker of another host never runs them, because their affinity keys are bound to
the host of the crashed worker. A worker that starts again on that host under the id of the
crashed worker settles them at once ([A job that its worker left](#a-job-that-its-worker-left)).

### Lost hosts

A host is lost when no row of that host in `rs_workers` has a heartbeat in the last 600 s, whether
its workers died or stopped ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). The last
heartbeat of a stopped row is the time of its stop. These rules follow:

- If a worker of the host starts again within 600 s of the last stop or heartbeat on the host, the
  host is not lost, and its jobs wait for that worker.
- A shutdown releases the running jobs of the worker, and they run again from the start
  ([Supervision](#supervision)). A planned reboot saves no progress of a running attempt.
- A host that reboots for more than 600 s is lost. A reboot of any length removes the kept
  sandboxes of the host.

At each sweep, `settle_dead_workers()` fails each unfinished job whose affinity key is bound to a
lost host:

- A queued job fails at once.
- A running job fails once its lease has expired. Its attempt ends `lost`.

Each of these jobs ends `failed`, with the phase `data_lost` and this result:

```json
{
  "status": "error",
  "failure_kind": "infrastructure",
  "reason": "data_lost",
  "host": "<host>",
  "error": "Host <host> is lost: none of its workers reported a heartbeat in 600 s"
}
```

These rules also apply:

- A queued grade in a kept sandbox of the lost host fails as `data_lost` too, if its `keep_until`
  has not passed. Its search runs again. If its `keep_until` has passed, the sweep finishes it as
  `sandbox_lost` first ([Dead workers](#dead-workers)).
- The Run Store keeps every job, attempt and result of the key.
- The key stays bound to the lost host. A worker of another host claims no later job of the key,
  and the next sweep fails such a job as `data_lost` too.
- The client runs the search again under a new key. The search coordinator runs an RL group again
  from its roots. The evaluation runs a task again from a new parent, and the gate runs the lost
  check again ([EVALUATION.md](EVALUATION.md#a-lost-host)).
- A worker that starts again on the lost host deletes the local data of its lost attempts
  ([Supervision](#supervision)).

### A job that its worker left

A worker that starts again under its own id first settles the jobs that its earlier session left
running. Once a minute, it also settles each running job of its id that no attempt of it runs, for
example a job whose `finish()` could not reach the database. `Worker.settle()` does these steps for
each such job:

1. If the attempt froze no payload, no child started. The worker releases the job, and stops.
2. It stops the child process group that the attempt recorded. It sends SIGTERM, and SIGKILL
   after 10 s. It identifies the process by these values, so that it never stops a process that
   reuses the PID:
   - the boot ID
   - the PID
   - the start time
3. If the attempt left `outcome.json`, the worker publishes it as at the end of an attempt.
4. Otherwise the attempt is lost, with the error `Attempt left no outcome`, and the job runs again
   ([Reruns](#reruns)).

### Frozen payload

The worker freezes the payload of each attempt in `rs_attempts.payload`, with its hash. The
payload holds these parts:

- the effective spec
- the profile
- the recovery point that the attempt starts from

The worker sends the payload to the child over stdin, after it has recorded the child's identity.
The payload is at most 16 MiB. The child never connects to PostgreSQL. The child rejects a payload
in these cases, and then it exits without an outcome:

- The payload is missing.
- The payload is truncated or larger than 16 MiB.
- The payload does not match its hash, its job or its attempt.
- The payload does not arrive in time.

### Reruns

A job runs at most `max_infra_retries` + 1 times. `max_infra_retries` is 0 to 2, and the default
is 2. The backoff is 30 s before the second attempt and 60 s before the third. The worker decides
by the `failure_kind` of the result:

| `failure_kind` | Examples | What the worker does |
|---|---|---|
| `infrastructure` | A 502, 503 or 504, a closed transport, an uncertain execution, a failed final snapshot, a child that exited without an outcome, a child whose worker process died (`worker_died`), a grade file whose checksum changed, or a lost attempt. | Runs the job again while attempts remain. Then the job fails. |
| `actor` | The agent's own error, a timeout, a step limit or an exhausted budget. | Does not run the job again. The job fails. |
| `stopped` | A stop request. | Does not run the job again. The job fails. |

No failure kind is for the configuration. The worker checks its profiles when it starts, and the
API checks each request at submission ([Checks at the start](#checks-at-the-start)). A job that the
worker cannot start, for example after its profile was removed from the config, fails before its
child starts. It does not run again.

An attempt is lost in these cases:

- The worker failed after the child started, for example when it could not record the child.
- The child exited without an outcome.
- The worker was cut off while the child ran.
- The worker died ([Dead workers](#dead-workers)).

Each attempt starts where its spec starts. A branch starts at its `parent_point`, with its
`continuation`. Any other job starts from the beginning. Each attempt gets the full `timeout_s` and
`budget_usd` of its spec, and the group deadline `deadline_at` still bounds its time. An attempt
never starts from a recovery point of an earlier attempt.

Before the job runs again, the worker deletes the data of the lost attempt in this order:

1. It removes the sandboxes of the attempt, or records them for the clean-up
   ([Supervision](#supervision)).
2. It deletes the snapshots that the `checkpoint.captured` records of the attempt's journal name,
   as `python -m sprout.worker sweep` does.
3. It deletes the attempt directory.
4. `finish(rerun=True)` deletes the events, the tool calls and the recovery points of the attempt
   in the Run Store, and each prefix node that no other attempt reaches.

The attempt row stays. Its `error` ends with the last 4000 characters of the attempt's
`worker.log`. A branch from a point of the lost attempt then finds no point, and fails.

A rollout that runs again loses the progress of its lost attempt. If failures cut long
trajectories more often than short ones, a trainer sees shorter trajectories than it would. A
grader failure runs only the grade job again, not its rollout. The worker salvages a
message-export rollout that will not run again ([Salvage](#salvage)).

### Supervision

`WorkerManager` runs each attempt in a thread. A `Watchdog` stops the child's process group in
these cases:

- the attempt's deadline passes (`timeout`)
- someone requests a stop (`stopped`)
- the worker shuts down (`worker_shutdown`)
- the worker is cut off (`worker_cut_off`)

The attempt then ends with the stop reason of the watchdog. A thread that blocks keeps its slot.

A child also stops when its worker process dies, for example after SIGKILL or a crash. The worker
gives the child its own PID as the last argument of the command line. From its first check, after
it read the payload, the child checks about once a second that its parent process has this PID.
When the worker process exits, the kernel gives the child another parent. Within about 1 s, the
child sends itself SIGTERM. Then the child stops as after a SIGTERM from the watchdog:

1. A rollout removes its sandbox, unless it has `keep_sandbox: true`
   ([Kept sandboxes](#kept-sandboxes)).
2. The child writes `outcome.json`. Whatever the run reported, the outcome has these values:

   ```json
   {
     "status": "error",
     "failure_kind": "infrastructure",
     "stop_reason": "worker_died",
     "error": "Worker process exited"
   }
   ```
3. The child exits.

A worker that starts again under the id of the dead worker settles the job from this outcome. The job
runs again while attempts remain ([A job that its worker left](#a-job-that-its-worker-left)). These
rules also apply:

- If the teardown has not ended 60 s after the child sent itself SIGTERM, the child writes the
  outcome, if it has not written one yet. Then it exits with the status 3. When the worker of the id
  starts again, it removes the sandboxes of the attempt, or records them for the clean-up.
- A SIGTERM that reaches the child after its worker process exited also gives the outcome
  `worker_died`. For example, a worker that starts again under the same id can settle the job and
  send the SIGTERM before the next check of the child.
- The SIGTERM handler of the child raises the interrupt at the first SIGTERM only. A later SIGTERM
  does not stop the teardown. The watchdog of a live worker sends SIGKILL 10 s after its SIGTERM.

On SIGINT or SIGTERM, the manager stops claiming, and the watchdog of each attempt stops its child.
The manager waits up to 20 s for the attempts. Then it stops its heartbeats and marks the worker's
row stopped. The worker releases each job that the shutdown stopped, and the attempt does not
count against `max_attempts`. The worker does these steps for each one:

1. If the child has not started and a grade in place took its kept sandbox over, the worker
   records the sandbox again ([Kept sandboxes](#kept-sandboxes)).
2. It removes the attempt's sandboxes, or records them for the clean-up.
3. It deletes the attempt's data, as before a rerun.
4. It calls `Store.release()`. The job changes to `queued`, and a claim can take it at once. If
   somebody requested a stop, `release()` cancels the job instead.

A cut-off before the child starts also releases the job. An attempt that does not end within the
20 s stays `running`. Another worker's sweep runs its job again once the lease expires, or the
worker settles it when it starts again.

Once a minute, the manager also does these steps:

1. It settles each running job of its worker that no attempt of it runs
   ([A job that its worker left](#a-job-that-its-worker-left)).
2. `Worker.clean_up()` deletes the local data of each lost attempt of the worker that is still on
   this host. It stops the attempt's child, removes its sandboxes, and deletes its snapshots and
   its directory. Then it adds the tail of the attempt's `worker.log` to the attempt's `error`, and
   sets the attempt's `data_deleted_at`. It also sets `data_deleted_at` on a lost attempt that has
   no directory, for example because the worker deleted it before a rerun. `Store.lost_attempts()`
   returns only the lost attempts without `data_deleted_at`. So each clean-up reads only the
   attempts that were lost since the clean-up before, and the attempts that it could not finish.
3. `Worker.clean_up()` frees each sandbox in `<artifact_root>/leftover-sandboxes.jsonl` that still
   exists. A sandbox that it cannot free stays recorded for the next minute.
4. It removes each kept sandbox whose `keep_until` has passed.

A sandbox whose removal the worker cannot confirm does not hold the job. The worker publishes the
outcome, and records in `leftover-sandboxes.jsonl` each sandbox that the attempt's ledger still
claims. The worker cannot record a sandbox start that no claim names, so it logs the attempt
directory. Free such a sandbox with `python -m sprout.worker reap --include-unknown --older-than
<age>`.

### Stopping

`POST /v1/jobs/{id}/stop` acts on the job's state:

- A `queued` job changes to `cancelled`.
- A `running` job gets `stop_requested_at`. A second request keeps the first time.
- A finished job stays as it is.

The worker reads the stop request at its next progress write, at most 5 s later. Then it does these
steps:

1. It stops the attempt's process the way a deadline does.
2. It removes the attempt's sandboxes.
3. It finishes the job `failed`, with `failure_kind` and `stop_reason` `stopped`.

The worker never runs a stopped job again. But the worker salvages a stopped message-export
rollout that has a valid recovery point, and that job finishes `succeeded` ([Salvage](#salvage)).

## Validation

Run the tests of the Run Store and the worker:

```bash
python3 -m pytest tests/runstore tests/worker -q
```

The database tests skip unless `SPROUT_RUNSTORE_TEST_DSN` names an isolated PostgreSQL database.
Each test makes a temporary schema in it and drops the schema afterwards. Do not point the
variable at a production database.
