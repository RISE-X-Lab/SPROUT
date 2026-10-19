# RL driver

The RL driver runs a trainer's rollout groups as Run Store jobs, and it returns their trajectories
and rewards. It serves HTTP on `127.0.0.1:11001` by default. It talks HTTP to the Run Store at
`127.0.0.1:18110`. The driver only schedules the work, and the Run Store's workers do it. The
search coordinator in `src/sprout/search/` submits and follows the jobs, and it runs the search that
branches the failed roots of a task.

Read this document if you do one of these tasks:

- connect Miles, or another trainer, to Sprout
- start and configure the driver
- operate the driver, and recover a group after a lost reply
- change the search, or predict which branches it makes

[RUNSTORE.md](RUNSTORE.md) describes the jobs that the driver submits, and the leak checks of a review.

## Architecture: the driver schedules, the workers execute

The driver runs no part of a job itself. It keeps a record of each group, and it reaches the jobs
only through the Run Store API. A worker claims each job from the Run Store and runs it. The figure
shows the components and the requests between them.

<p align="center">
  <img src="figures/rl-driver-flow.svg" width="1000" alt="Twelve components in three boundaries, and the requests between them. Miles sends POST, GET and DELETE requests on /rollout-groups to the HTTP API of the RL driver process, create_app() on port 11001. Inside the driver, the HTTP API hands the Miles routes to the Miles adapter, MessageAdapter, and the execution routes and the poller's tick() to the execution groups, Driver. MessageAdapter calls Driver with submit() and release(). Driver calls tick() on the search coordinator, Coordinator with Stages and Search. The coordinator calls active(), get() and save() on the group documents, Ledger, one SQLite file. The coordinator sends HTTP requests on /v1/jobs to the Run Store API, sprout.runstore serve on port 18110, to submit, branch, poll and stop jobs. The Run Store API reads and writes PostgreSQL through Store. PostgreSQL holds jobs, attempts, events and recovery points. The worker, python -m sprout.worker start, claims and finishes jobs in PostgreSQL directly. It runs three kinds of job: a rollout job (Runner, mini-swe-agent and Sprout tools in a podman or microvm sandbox), a grade job (the SWE-rebench grader, in two sandboxes) and a review job (SPROUT's analyst and reviewer). The rollout job sends Chat Completions requests through the inference gateway to the model server at the request's model_endpoint, and the review job sends its model calls to the same server."/>
</p>
<p align="center"><sub>Figure 1: Miles calls the driver, the driver calls the Run Store API, and the workers claim the jobs from PostgreSQL.</sub></p>

Each component has one role:

| Component | Code | Role |
| --- | --- | --- |
| Miles | `miles.rollout.sprout.rollout_fn.SproutRolloutFn`, in the Miles branch `sprout-grpo` | Sends one request for each prompt group, and trains on the result. |
| HTTP API and poller | `create_app()` in `src/sprout/rl_driver/server.py` | Serves the routes. Its background thread calls `Driver.tick()`. |
| Miles adapter | `MessageAdapter` in `src/sprout/rl_driver/messages.py` | Checks a Miles request, translates it to an execution group, and exports the Miles result. |
| Execution groups | `Driver` in `src/sprout/rl_driver/driver.py` | Checks and records a group, gives its view, and releases it. |
| Search coordinator | `Coordinator` in `src/sprout/search/coordinator.py`, with `Stages` and `Search` | Submits each job, polls it, and runs the search. |
| Group documents | `Ledger` in `src/sprout/rl_driver/ledger.py` | Keeps the request and the JSON document of each group in one SQLite file. The coordinator reads and saves them with `active()`, `get()` and `save()`. |
| Run Store API | `python -m sprout.runstore serve` | Accepts the jobs, and keeps jobs, attempts, events and recovery points in PostgreSQL. |
| Worker | `python -m sprout.worker start` | Claims each job in PostgreSQL (`Store.claim()`), runs it, and records its end (`Store.finish()`). A job is a rollout, a grade or a review. |

These steps take one Miles group from the request to the result:

1. Miles sends `POST /rollout-groups`. `MessageAdapter` checks the request and makes one rollout
   sample for each root slot.
2. `Driver.submit()` records the group in the ledger.
3. The poller calls `Driver.tick()`, which calls `Coordinator.tick()`. The coordinator submits each
   rollout job to the Run Store under a fixed idempotency key.
4. A worker claims each rollout job. Its Runner runs mini-swe-agent with the Sprout tools in a
   sandbox. Each model call goes through the worker's inference gateway to the request's
   `model_endpoint`.
5. When a rollout succeeds, the coordinator submits its grade job. A worker grades the final
   snapshot of the rollout with `tasks[task_id].grade`.
6. If the request has a `search`, the coordinator submits a review job after every root settles.
   Then it submits the branches that the review proposes.
7. When every job of the group has settled, the group document is ready. Miles reads
   `GET /rollout-groups/{id}`. The first read exports the hint-free messages and the rewards.
8. Miles sends `DELETE /rollout-groups/{id}` to acknowledge the result.

The driver does not start any of these:

- the Run Store
- a worker
- the model server
- the training

When the driver process stops, the jobs that it submitted keep running. A restarted driver follows
them again ([Idempotency, restarts and cancellation](#idempotency-restarts-and-cancellation)).

The worker profile selects the sandbox. By default, a sandbox is a rootless podman container on this
host, on a btrfs store. A profile whose `backend` is `microvm` uses an AgentENV microVM instead.

The evaluation uses the same search coordinator. It keeps its group documents in JSON files instead
of the ledger ([EVALUATION.md](EVALUATION.md)).

## Start and configuration

Before you start the driver, the Run Store API and at least one worker must run. Then do these
steps:

1. Go to the root of the Sprout checkout.
2. Install the dependencies of the driver: `python3.12 -m pip install -e ".[rl]"`.
3. Export the Run Store token: `export SPROUT_RUNSTORE_TOKEN='<existing-runstore-token>'`.
4. Start the driver: `python3.12 -m sprout.rl_driver --config resources/rl_driver/config.example.json`.
   The driver serves on `127.0.0.1:11001`. `GET /health` returns
   `{"role": "driver", "miles_configured": true}`.

To serve on another address, add `--host <host>`, `--port <port>` or both. The example config sets
`driver_token_env` to `null`, so the driver accepts requests without a token. Use that only for
local access.

`python -m sprout.services up rl-driver` also starts the driver ([SERVICES.md](SERVICES.md)). The
command does these steps:

1. It starts the database and the Run Store API, if their health probes fail.
2. It starts the driver with `~/.config/sprout/rl_driver.json`. The driver gets
   `SPROUT_RL_DRIVER_TOKEN` from the generated `rl-driver-token` secret.

The worker is not a dependency of `rl-driver`. Start the worker with
`python -m sprout.services up runstore-worker`, or start every enabled service with
`python -m sprout.services up`.

`python -m sprout.services init` writes `rl_driver.json` from the example config, if the file does
not exist. The written file differs from the example in two fields:

- `ledger` is `~/.local/state/sprout/services/rl-driver/groups.sqlite3`, under the state directory.
- `driver_token_env` is `SPROUT_RL_DRIVER_TOKEN`.

The command `python3.12 -m sprout.rl_driver` stops with a usage error when one of these is true:

- The config has a field that [The config file](#the-config-file) does not list.
- `runstore_url` is not an HTTP(S) URL with a host.
- `runstore_url` has one of these parts:
  - credentials
  - a query
  - a fragment
- A token variable that the config names is not set, or it is empty.
- The config has no `ledger`.
- `poll_interval_s` is not a finite positive number.
- The ledger is bound to another Run Store URL.
- The `miles` section is not valid.

If another driver process holds the ledger, the app does not start. `Ledger.owner()` raises
`Conflict`.

### The config file

The config file has these top-level fields (`resources/rl_driver/config.example.json`):

| Field | Default | Meaning |
| --- | --- | --- |
| `runstore_url` | `http://127.0.0.1:18110` | The address of the Run Store. It must not have credentials, a query or a fragment. |
| `runstore_token_env` | `SPROUT_RUNSTORE_TOKEN` | The environment variable that holds the Run Store token. The variable must be set. |
| `driver_token_env` | `SPROUT_RL_DRIVER_TOKEN` | The environment variable that holds the bearer token of the callers. `null` turns authentication off, for local use. |
| `ledger` | required | The path of the SQLite file. `~` is the home directory. A relative path starts from the directory of the config file. |
| `poll_interval_s` | `1` | The seconds between two ticks of the poller. |
| `miles` | none | The Miles adapter config. Without it, the Miles routes return 503. |

The `miles` section has these fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `profile` | `mini` | The worker profile of the root rollouts. A branch keeps the profile of its parent. |
| `run_defaults` | `{}` | A JSON object of RunSpec fields that the adapter merges into each root rollout. `slot` must be `mini-swe-agent`. `extra` must be a JSON object. `extra.rollout_contract` is refused, because each request gives it. `model` is used when the request names no model. |
| `image_resources` | required | `cpu` and `memory_mb`, positive integers: the resources of each sandbox. Without them, the adapter refuses each request. |
| `tasks` | `{}` | For each `task_id`, a `grade`: a grade JobSpec template without `snapshot_id`. A request for a task without one is refused. |
| `api_key_env` | `null` | The variable in the worker's environment that holds the key of the inference endpoint. |
| `max_samples` | `1000` | The largest `max_samples` that a request can ask for. |
| `grade_wall_seconds` | `600` | The longest time that one grade can run. It must be at least 240. See [Time limits](#time-limits). |
| `review` | none | The review config. A search request needs it. |

The `review` section names the worker profile of the reviews and the reviewer's model-call settings.
The adapter copies each setting into the `ReviewSpec` of each review job. The defaults are those of
`ReviewSpec`:

| Field | Default | Meaning |
| --- | --- | --- |
| `profile` | required | The worker profile that runs the review jobs. |
| `max_output_tokens` | `16384` | The output limit of one model call of the review. |
| `prompt_chars` | `400000` | The character budget of one prompt of the analyst or the reviewer. |
| `transcript_tokens` | `100000` | The token budget of the transcript that the analyst reads. |
| `temperature` | `0.8` | The sampling temperature of the model calls, from 0 to 2. |
| `reasoning_effort` | `null` | The reasoning effort of the model calls. |
| `max_attempts` | `3` | The tries of the analyst or the reviewer to give a valid reply. |
| `max_concurrent_requests` | `8` | The most model requests that one review has in flight at the same time. |
| `leak_judge` | `null` | The leak check of each repair turn: `threshold`, `model`, `resamples` and `api_key_env`, each optional. See [Leak checks](#leak-checks). |

### Tokens, the ledger and the model key

The driver keeps every token out of the ledger, and one driver process holds a ledger at a time.
These rules apply:

- The driver reads each token from the environment only. It writes no token to the ledger.
- The driver refuses a group request that holds an inline credential, such as an `api_key` value.
  Use a `{"$env": "<NAME>"}` reference instead (`no_credentials` in `src/sprout/runstore/specs.py`).
- On its first use, the ledger binds to the `runstore_url` of the config. A later start with another
  URL fails.
- One driver process holds the ledger at a time, through a lock on the file `<ledger>.lock`.
- The workers decide how many jobs run at the same time. The driver sets no limit.
- If the inference endpoint needs a key, `miles.api_key_env` names a variable in the worker's
  environment. The worker profile's `worker_env` puts the variable there.
- The model address comes from the request. The worker's gateway gives that route no other
  credential.

## The Miles protocol

Miles sends one request for each prompt group to `/rollout-groups`, and it reads one result for each
group. The adapter checks each request before it records the group, and it refuses a request that
fails a check. Then no job exists for that request.

### Requests

`POST /rollout-groups` takes one request for each prompt group. The request has no version field.
The adapter refuses each unknown field. `MessageRequest` in `src/sprout/rl_driver/messages.py`
defines the fields. `examples/rl_driver/request.example.json` is a complete request.

| Field | Rule |
| --- | --- |
| `rollout_job_id` | A non-empty string. It is the identity of the group. |
| `rollout_id` | An integer, 0 or more. |
| `prompt_group_id` | A non-empty string. |
| `task_id` | A non-empty string. `miles.tasks` must have a `grade` for it. |
| `image` | A non-empty string: the task image. See [Image, resources and jobs](#image-resources-and-jobs). |
| `prompt` | Text, or a list of one message `{"role": "user", "content": "<text>"}`. The adapter refuses any other form before it queues a job. |
| `model_endpoint` | An HTTP(S) URL with a host, and without credentials, a query or a fragment. It must serve `/v1/chat/completions`. It does not have to return token records. |
| `model` | Optional. Without it, the adapter uses `run_defaults.model`. |
| `sample_slots` | A non-empty list of `{"sample_slot_id": ..., "sample_index": ...}`. Each id and each index is unique. |
| `max_samples` | 1 ≤ `minimum_returned_samples` ≤ `max_samples` ≤ the number of slots. `max_samples` is at most `miles.max_samples`. |
| `minimum_returned_samples` | See `max_samples`. |
| `max_turns` | A positive integer. It limits each trajectory on its own. |
| `sampling_params` | Optional. See [Sampling parameters](#sampling-parameters). |
| `budgets` | `{"max_wall_time_seconds": <seconds>}`, finite and positive. No other key is allowed. |
| `finalization_timeout_seconds` | Optional, finite and positive. The default is 1800. |
| `search` | Optional. See [Search](#search-hindsight-branches-after-the-root-rollouts). |

One turn is one model call. The tool calls that a response triggers take no turn of their own. The
worker counts the tool calls but does not limit them. If `budgets` names `max_model_calls` or
`max_tool_calls`, the adapter refuses the request with this error:
`budgets contains only max_wall_time_seconds; use per-trajectory max_turns`.

### Sampling parameters

`sampling_params` accepts these keys (`validate` in `src/sprout/contract/sampling.py`):

| Key | Value |
| --- | --- |
| `temperature` | 0 or more |
| `top_p` | more than 0, and at most 1 |
| `top_k` | -1, or an integer of 0 or more |
| `stop` | text, a list of texts, or `null` |
| `reasoning_effort` | `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or `ultra` |
| `max_tokens`, `max_new_tokens`, `max_output_tokens` | a positive integer. If the request gives more than one, they must be equal. |

The adapter refuses any other key. The worker applies the keys to each Chat Completions request. It
always sends the output length as `max_tokens`.

### Image, resources and jobs

The adapter makes these jobs for each root slot:

- one rollout job, in the worker profile `miles.profile`
- one grade job, from the template `tasks[task_id].grade`

Without a search, the first `max_samples` slots get a root rollout. With a search, the first
`search.roots` slots get one.

If `image` starts with `prime/primeintellect/`, the adapter replaces that prefix with
`docker.io/swerebenchv2/`. Prime's dataset documents this rewrite. The rollout and the grade both
use the rewritten image. `metadata.source_image` in the result keeps the image of the request. Each
sandbox gets the `cpu` and the `memory_mb` of `miles.image_resources`.

The rollout job gets `extra.rollout_contract` with these fields:

- `message_export`: `true`
- `model_endpoint` and `model`
- `sampling_params`
- `deadline_at`: the execution deadline
- `max_turns`
- `api_key_env`: `miles.api_key_env`

Each job of a Miles group has `max_infra_retries` 0. The Run Store therefore gives each Miles job
one counted attempt. If a worker shuts down, it releases the job, and that attempt does not count
([Job states](RUNSTORE.md#job-states)).

### Time limits

A Miles group has these time limits:

| Limit | Default | What it bounds | What happens at the limit |
| --- | --- | --- | --- |
| `budgets.max_wall_time_seconds` | required | The execution phase of the roots, from the submission. Time in the queue counts. | The worker stops the agent with `stop_reason` `timeout`. The rollout is graded. |
| `finalization_timeout_seconds` | 1800 | The final snapshot, the message export and the grades, after the execution phase. | It is part of the group deadline. |
| `miles.grade_wall_seconds` | 600 | One grade, from when its job leaves the queue. | The driver records the grade as `expired` and sends a stop request for its job. The sample is missing. |
| `search.review_seconds` | required | One review job. | It is the `timeout_s` of the review job. |
| `search.wall_time_seconds` | required | One branch, from when the driver plans it. | It is the `deadline_at` of the branch. |
| group `deadline_at` | computed | The whole group, from the submission. | The coordinator requests the cancellation of the group. The result is `cancelled`, with no trajectories. |

The adapter computes the group deadline when it accepts the request:

```text
deadline_at = submission time + max_wall_time_seconds + finalization_timeout_seconds
              + SearchSpec.search_seconds(grade_wall_seconds)
search_seconds = R × (review_seconds + wall_time_seconds) + (R − 1) × grade_wall_seconds
```

Without a search, the last term is 0. R is `repair_rounds` when `candidates` is more than 0, and 1
otherwise. The grades of the last round use the finalization time.

The execution deadline does not cancel the group. The rollouts stop at it, and their grades run in
the finalization time. The timeout of each root rollout is the smaller of `run_defaults.timeout_s`
and `max_wall_time_seconds`. The `timeout_s` of each grade is the smaller of the template's
`timeout_s` and `grade_wall_seconds` minus 120 s. The 120 s are for the start of the sandboxes and
the diff.

### Grading

Each grade is a Run Store grade job, from the template `tasks[task_id].grade`. A `swe-rebench-v2`
template has these `spec` fields:

| Field | Meaning |
| --- | --- |
| `benchmark` | `swe-rebench-v2` |
| `instance_id` | The task row. |
| `dataset_path`, `dataset_sha256` | The frozen task rows, pinned by their SHA-256. The path must be valid on the worker, and a grade profile of the worker must name a file with this SHA-256. |
| `parser_path` | The SWE-rebench log-parser file on the worker. |
| `grader_revision` | `sha256:<digest>` of the parser file. |

`grade_swerebench` in `src/sprout/worker/jobs/grade.py` grades in two sandboxes:

1. The worker restores the final snapshot in a sandbox. That sandbox only gives the agent's diff
   against `base_commit`.
2. The worker starts a fresh sandbox of the task image.
3. The worker applies the diff. It does not apply an empty diff.
4. The worker applies the task's test_patch.
5. The worker runs the test command, and it parses the log with the pinned parser.

The worker moves these files into and out of a sandbox through `upload_file` and `download_file`,
never inside a command:

- the patches
- the test script
- the test log

The section "Official grading" of [RUNSTORE.md](RUNSTORE.md) gives the result files.

The result has these values:

- `resolved` is `true` only when every FAIL_TO_PASS and every PASS_TO_PASS test passes. A task
  without tests is not resolved.
- `progress` is the share of the FAIL_TO_PASS tests that passed. If a PASS_TO_PASS test fails,
  `progress` is 0. If the task has no FAIL_TO_PASS tests, `progress` is 1.0 when the task is
  resolved and 0.0 when it is not.
- `expected_pass_share` is the share of all expected tests that passed, PASS_TO_PASS included. No
  weight uses it.

Each completed grade also has three facts about the hidden tests of the task:

- `test_patch_paths`: each path that the test_patch names, old and new, sorted and unique
- `touched_test_paths`: the paths of `test_patch_paths` that the agent's diff also changes
- `fail_to_pass`: the FAIL_TO_PASS tests of the task row

No prompt of the analyst or the reviewer shows these three facts.

A failure that the agent can cause is a completed grade of 0. Such a grade has these values:

- `resolved: false`
- `progress: 0.0`
- a `reason` from the next table
- an `error_output` that describes the failure

| `reason` | Cause |
| --- | --- |
| `model_patch_unavailable` | `git diff` failed in the snapshot, for example because the agent deleted `.git` or the base commit. |
| `model_patch_apply_failed` | The agent's diff does not apply. |
| `test_patch_apply_failed` | The test_patch does not apply after the diff. |
| `tests_timed_out` | The tests did not finish within the `timeout_s` of the grade. |
| `test_runner_killed` | The test script exited with a code other than 0 or 1, for example after the OOM killer stopped it. |

A grade that the worker cannot make does not complete:

- The grade job fails in these cases:
  - A sandbox does not restore or start.
  - A file does not move in or out of a sandbox.
  - The parser raises an error.
  - The checksum of the dataset or of the parser on the worker differs from the spec.
- No worker claims the grade while no live worker has the dataset and the parser of the spec. A
  worker has them when one of its grade profiles names them with these checksums
  ([RUNSTORE.md](RUNSTORE.md#grade-files)). The grade job stays queued. When the group's
  `deadline_at` passes, the coordinator cancels the group.

In the first case, the sample has no verdict, so the sample is missing
([Missing trajectories](#missing-trajectories)). The Run Store does not retry the grade of a Miles
group. A grade with `resolved: false` is a valid result. The driver does not run the actor again.

### Stopping and truncation

When a rollout reaches `max_turns` or its deadline, the worker does these steps:

1. It stops the agent.
2. It waits for the tools of the agent to finish.
3. It saves the final snapshot.
4. It ends the job as `succeeded`. The result has `status: "truncated"` and the `stop_reason`
   `max_turns_reached` or `timeout`.

Then the driver submits the grade. The reward of a truncated trajectory comes from the real grade.

Some rollouts fail midway and will not run again, for example after one of these events:

- a crash
- a lost connection to the sandbox
- an uncertain command result
- a stop request

The worker cuts such a rollout at its last valid recovery point. It ends the job as `succeeded`,
with a truncated result and `stop_reason: "interrupted"`. The driver grades the cut rollout and
returns it like any truncated trajectory. If the rollout closed no turn, its actor fails, and its
slot is in `failed_samples`.

These conditions make an attempt fail explicitly:

- an uncertain result of an execution
- a safe snapshot that the worker cannot save
- a native record that does not make a valid trajectory

The worker gives an attempt that exports training messages 120 s more than its timeout, to write the
export (`TERMINATION_GRACE_SECONDS`). The cut-off and the lost-contact checks of the worker still
apply in that time. A tool operation that has started finishes under the drain and cleanup
rules of the worker. A stop does not end every side effect at once.

### Results

Get the result of a Miles group with these steps:

1. Send `POST /rollout-groups`. The driver answers 202 with `rollout_job_id` and `status`.
2. Poll `GET /rollout-groups/{rollout_job_id}` until `status` is not `queued` or `running`.
3. Store or consume the result.
4. Send `DELETE /rollout-groups/{rollout_job_id}`. The driver records the acknowledgement.

The first read of a ready group exports the result and stores it in the ledger. Each later read
returns the stored result, also after a restart of the driver.

A group that failed after it ran again 2 times exports no sample ([A lost host](#a-lost-host)).
Its result has the `status` `failed` and the `error` of the group as `stop_reason`. The samples
that settled before the last loss are only a part of the group, so the result does not contain
them. Miles leaves the group out of the step ([The Miles side](#the-miles-side)).

The result has these fields:

| Field | Meaning |
| --- | --- |
| `rollout_job_id`, `prompt_group_id`, `max_samples` | The values of the request. |
| `actual_samples` | The number of trajectories. |
| `trajectories` | One entry for each sample that exported a trajectory, in slot order. |
| `failed_samples` | `{"sample_slot_id": ..., "reason": ...}` for each sample that exported no trajectory. Empty when the group exports no sample: before it is ready, after a cancellation, or after its runs again. |
| `search_branches` | The number of trajectories that have a parent branch. |
| `consumed_budget` | The sums of `model_calls` and `tool_calls` over the trajectories. |
| `status` | The status of the group. See the next table. |
| `stop_reason` | The reasons of the failed samples, one after the other, or `null`. For a group that failed after its runs again, the `error` of the group. |

The result has one of these statuses:

| `status` | Condition |
| --- | --- |
| `queued` | The poller has not advanced the group yet. |
| `running` | The group is not ready. |
| `completed` | Each sample of the group exported a trajectory. |
| `early_stopped` | At least `minimum_returned_samples` trajectories, and at least one entry in `failed_samples`. |
| `failed` | Fewer than `minimum_returned_samples` trajectories, or the group failed after its runs again. The second kind has no trajectory. |
| `cancelled` | A `DELETE` or the group deadline requested the cancellation before the first read of the ready group. The result has no trajectories. |

Miles sends `minimum_returned_samples` 1 with a search request. A root whose actor failed is then in
`failed_samples`, and the group does not fail. When every root fails, the group is `failed` with no
trajectories. Miles treats it as an empty group.

Each trajectory has these fields:

| Field | Meaning |
| --- | --- |
| `sample_slot_id` | The slot of the request. |
| `branch_id` | The Run Store job id of the actor. |
| `parent_branch_id` | For a branch, the job id of its parent. For a root, `null`. |
| `group` | `root`, `student:<point_id>` or `repair:<point_id>:<n>` in a search. `null` without a search. Miles computes z-scores within a group. |
| `messages` | The hint-free messages. |
| `tools` | The Chat Completions tool schemas that the model was offered. |
| `reward` | 1.0 when the grade resolved the task, and 0.0 otherwise. A missing trajectory also has 0.0. |
| `status` | `completed` or `truncated`. |
| `stop_reason` | For a truncated trajectory, `max_turns_reached`, `timeout` or `interrupted`. Otherwise `null`. |
| `hints_removed` | Always `true`. |
| `metadata` | See the next table. |

The `metadata` of a trajectory has these fields, in this order:

| Field | Meaning |
| --- | --- |
| `job_id`, `attempt_id` | The job and the attempt of the actor. |
| `source_image` | The `image` of the request. |
| `graded_snapshot_id` | The snapshot that the grade read. |
| `progress` | The `progress` of the grade, or `null` for a missing trajectory. |
| `origin` | For a branch, the `job_id` and the `point_id` that it continues. For a root, `{}`. |
| `provenance` | See [Provenance and hint removal](#provenance-and-hint-removal). |
| `search` | `null` without a search. See [`metadata.search`](#metadatasearch). |
| `logprob_context` | Always `hint_free_messages`. |
| `zero_reason` | Why the trajectory is missing, or `null` when it has a verdict. |
| `touched_test_paths` | The `touched_test_paths` of the grade: `[]` for none. `null` for a missing trajectory, or for a grade result without the field. |

### Missing trajectories

A trajectory is missing, not a 0, when its grade gave no verdict. The driver still returns the
trajectory, with these values:

- `reward` is 0.0, because the wire needs a number.
- `metadata.progress` and `metadata.touched_test_paths` are `null`.
- `metadata.zero_reason` names the cause.

`zero_reason` in `src/sprout/search/outcome.py` gives the cause, one of these:

| `zero_reason` | Cause |
| --- | --- |
| `grade_failed` | The grade job failed. |
| `grade_expired` | The grade ran past `grade_wall_seconds`. |
| `grade_cancelled` | The grade job ended `cancelled`. |
| `grade_skipped` | The driver did not run the grade, because the actor did not succeed. |
| `grade_no_verdict` | The grade succeeded without a boolean `resolved`. |

In a group that exports its samples, a sample that exports no trajectory is in `failed_samples`
with its reason, and it is missing too.
`export_problem` in `src/sprout/search/outcome.py` checks these conditions, in this order:

1. The actor succeeded.
2. A truncated actor left a verified cutoff.
3. The worker recorded no `training_export_error`.
4. The worker exported the native messages.
5. The worker recorded which messages the policy sampled (`training_provenance`).
6. The messages are in their cleaned training form.
7. The sample has a grade.

In a search group, Miles makes no sample from a missing trajectory, and it counts the trajectory in
no statistic. `rollout/sprout/reward_mean` covers only the trajectories with a verdict. The missing
ones count toward `rollout/sprout/ungradable`. Miles does not treat groups without a search this
way yet.

### Shared examples

Sprout and Miles test against the same three example files in `examples/rl_driver/`:

| File | Content | Sprout test |
| --- | --- | --- |
| `request.example.json` | A request without a search. | `tests/rl_driver/test_miles_rollouts.py` |
| `result.example.json` | The result that the adapter returns for that request. | `tests/rl_driver/test_miles_rollouts.py` |
| `search_result.example.json` | One search whose two roots both failed (`FF`). Each root has one point. Each point has 2 students and 2 repair turns with 2 repairs each: 14 trajectories in all. | `tests/search/test_miles_examples.py` |

Each Sprout test checks that the driver returns exactly the result file. Miles's tests use the
first two files as fixtures. Miles's GSML tests read a byte-for-byte copy of the third file,
`tests/fast/rollout/sprout/fixtures/sprout_search_result.example.json`.

Miles's `test_gsml_wiring.py` checks that the three files are byte-for-byte equal to Miles's copies.
It looks for a Sprout checkout in `$SPROUT_REPO`, or in a `sprout` or `SPROUT` directory beside
Miles. If it finds none, it skips. If you change an example file here, copy it into the fixtures of
Miles.

## Search: hindsight branches after the root rollouts

A search spends its extra samples only on the tasks whose roots failed. The roots run first. Their
label then decides whether a review runs and which branches follow. A request asks for a search with
the optional field `search` (`SearchSpec` in `src/sprout/search/spec.py`).

### The search fields

`search` must have all of these fields:

| Field | Value | Role |
| --- | --- | --- |
| `roots` | 1 or more | The rollouts from scratch. They take the first `roots` slots. |
| `points` | 1 or more | The most branch points that the review picks in each failed root. |
| `candidates` | 0 or more | Under `full`: the repair turns that the review writes at each point. 0 means no repairs. |
| `student_continuations` | 1 or more | Under `full`: the students at each point. |
| `candidate_continuations` | 0 or more. 0 exactly when `candidates` is 0. | Under `full`: the repairs from each repair turn. |
| `tf_student_continuations` | 0 or more | Under `points_only`: the students at each point. |
| `repair_rounds` | 1 or more. More than 1 only when `candidates` and `candidate_continuations` are both 1. | Under `full`: the rounds of repairs. |
| `review_seconds` | finite, more than 0 | The `timeout_s` of each review job. |
| `wall_time_seconds` | finite, more than 0 | The time of each branch, from when the driver plans it. |

A search request must also meet these conditions:

- `search.roots` is at most `max_samples`.
- `minimum_returned_samples` is at most `search.roots`, because only the roots are sure to run.
  Miles sends 1.
- `search.roots` + `SearchSpec.branches()` is at most `max_samples`.
- The adapter config has a `review` section. The adapter builds the review job templates when it
  accepts the request, so a review config that is not valid fails the request at once.

### The label of the roots decides the shape

When every root has settled, the search counts the roots:

- `planned`: `search.roots`
- `graded`: the roots that exported a trajectory and got a verdict
- `resolved`: the graded roots whose verdict resolved the task

A root counts toward neither `graded` nor `resolved` in these cases:

- Its actor failed.
- Its export is unavailable.
- Its grade is missing.

`search_outcome` in `src/sprout/search/outcome.py` makes the label from the three counts. Miles's
`miles.rollout.sprout.gsml.search_outcome` is the same function. Miles recomputes the label from the
roots that it receives. A change of the label rules here needs the same change in Miles.

| Label | Roots | Shape | What runs next |
| --- | --- | --- | --- |
| `TT` | All are graded, and all resolved. | `none` | No review and no branches. |
| `TT_partial` | Every graded root resolved, and some are missing. | `none` | No review and no branches. |
| `ungraded` | None is graded. | `none` | No review and no branches. |
| `TF` | Some resolved, and some failed. | `points_only` | A review that only picks points. Then `tf_student_continuations` students at each point. |
| `FF_partial` | None resolved, some failed, and some are missing. | `points_only` | The same as `TF`. |
| `FF` | All are graded, and none resolved. | `full` | The full review. Then students and repairs at each point. |

If `tf_student_continuations` is 0, the shape of `TF` and `FF_partial` is `none` (`search_shape`).
Such a search runs no review and no branch. Its `search.error` is
`tf_student_continuations is 0: a task whose roots split is not branched`.

### Search states

A search moves through five states. Each review and each round of branches waits until everything
before it has settled. The figure shows the states and the functions of `Search` in
`src/sprout/search/rounds.py` that move a search between them.

<p align="center">
  <img src="figures/rl-driver-search-states.svg" width="1000" alt="The five states of a group's search and the functions of Search that move it between them. roots goes to reviewing through _review_roots() when every root has settled, the shape of the roots' label is not none and the group is not cancelled. roots goes to done when the shape is none, when the group is cancelled, or for a stored search document from before outcome-keyed review. reviewing goes to branching through _plan_branches() when the review succeeded and at least one branch was planned. reviewing goes to done when the review did not succeed or no branch could be made. branching goes to chaining through _next_round() when every branch has settled, the shape is full, a repair round is left, the group is not cancelled and a point qualifies. branching goes to done when there is no next round. chaining goes back to branching through _plan_chain() when the review of the round succeeded and at least one branch was planned. chaining goes to done when that review did not succeed or no branch could be made. done is terminal. In done, search.error, search.chain_error or search.chain_end, when it is set, gives the reason."/>
</p>
<p align="center"><sub>Figure 2: A search runs the roots, a review, the branches, and then one review and one set of branches for each later repair round.</sub></p>

The search state is `search.state` in the group document. `Search.advance()` moves it at each tick,
after the samples advance:

| From | Trigger and condition | Actor | To |
| --- | --- | --- | --- |
| `roots` | Every root's actor and grade have settled. The shape is not `none`, and the group is not cancelled. | `_review_roots()` plans the review | `reviewing` |
| `roots` | The shape is `none`. `search.error` gets the reason, for example `every root resolved the task`. | `_review_roots()` | `done` |
| `roots` | The group is cancelled, and the shape is not `none`. `search.error` is `null`. | `_review_roots()` | `done` |
| `roots` | The stored document predates outcome-keyed review. See [Stored search groups](#stored-search-groups). | `_review_roots()` | `done` |
| `reviewing` | The review job succeeded, and at least one branch was planned. | `_plan_branches()` | `branching` |
| `reviewing` | The review job succeeded, but the search can make no branch from it. `search.error` is `the review proposed no branch that could be made`. | `_plan_branches()` | `done` |
| `reviewing` | The review job ended in another done state. `search.error` gets its error, or `review <state>`. | `Search.advance()` | `done` |
| `branching` | Every branch and its grade have settled. The shape is `full`, a repair round is left, the group is not cancelled, and a point qualifies. | `_next_round()` plans the next review | `chaining` |
| `branching` | Every branch has settled, and the condition of the previous row is false. | `Search.advance()` | `done` |
| `chaining` | The review of the round succeeded, and at least one branch was planned. | `_plan_chain()` | `branching` |
| `chaining` | The review of the round succeeded, but the search can make no branch from it. `search.chain_error` says so. | `_plan_chain()` | `done` |
| `chaining` | The review of the round ended in another done state. `search.chain_error` gets its error. | `Search.advance()` | `done` |

These rules also apply:

- A search in `branching` waits for every branch of the group, not only for the branches of the
  last round.
- If no point qualifies for a next round, `search.chain_end` says so. After the last round, the
  search sets no reason.
- When a search ends early, the group returns the samples that it has. Unfilled slots are not
  failures, so `actual_samples` can be less than `max_samples`.
- If a Run Store call fails or a review plan is malformed, the search keeps its state and records
  the error in `search.last_error`. The next tick tries the same step again.
- The search records the points and the skipped entries of a review only after it read every
  point. This applies to the review after the roots and to the review of each later round. If a
  read fails, the next tick plans every point again. No point gets two sets of branches, and
  `search.skipped` gets the entries of each review once.
- A group is ready only when its search is `done` and every stage has settled.
- If a job of the group loses its data with its host, the group runs again from its roots. Its new
  search starts again in `roots`, and `earlier_runs` keeps the old one
  ([A lost host](#a-lost-host)).

### The review

The review is one Run Store `review` job: SPROUT's analyst and reviewer, run by the policy model
under the hindsight prompts. The section "Review" of [RUNSTORE.md](RUNSTORE.md) describes the job.
The search submits one review after the roots, and one more for each later repair round:

| Review | Sample id for its idempotency key | Parents |
| --- | --- | --- |
| After the roots | `search` | Each root whose verdict is `false`. |
| Repair round r, for r of 2 or more | `search-r<r>` | Each repair of round r − 1 that qualifies ([Repair rounds](#repair-rounds)). |

The review reads each parent with these keys of its grade result (`REVIEW_GRADE_KEYS` in
`src/sprout/search/rounds.py`):

- `resolved`
- `progress`
- `test_statuses`
- `model_patch`
- `log`
- `reason`
- `error_output`
- `fail_to_pass`
- `test_patch_paths`
- `test_patch`

Only the leak checks read the last three keys. The review does not get `touched_test_paths`.

The `context.rl_driver` of the review job records these values:

- `rollout_job_id`
- `phase`: `review`
- `outcome`: the label of the roots
- `round`: only for the review of a later round

Each review reviews each parent on its own (`points_per_attempt: true`). It picks at most `points`
points in each parent. Each review shape makes a different review job:

| Review shape | `guidance` | `candidates` | What the review writes |
| --- | --- | --- | --- |
| `full` | `assistant-turn`, or `none` when `candidates` is 0 | `candidates`, or 1 when `candidates` is 0 | At each point, at most `candidates` repair turns, each with its `leak_terms`. The review draws `candidates` independent hindsight samples, and it merges the samples that propose the same commands into one turn. |
| `points_only` | `none` | 1 | The points only. |
| `chain`, for a later round | the same as `full` | 1, with `points` 1 | One point in each parent repair, and one repair turn at it. |

The `leak_terms` of a turn are hidden test names or test file paths that the turn mentions. They
are only the terms that the history before the point never showed. `[]` is clean. `null` means that
the grade had no facts to compare against.

### Branches and slots

Each branch continues a root, or a repair, from a recovery point that a review picked. The figure
shows the branches of one failed root under the settings of the Miles launcher.

<p align="center">
  <img src="figures/rl-driver-search-tree.svg" width="1000" alt="The trajectories that a search makes from one failed root of an FF task, under the launcher's search settings: one point, one student, one repair turn with one repair, and three repair rounds. Root A, in the first slot and in the group root, failed. The review in slot search picks point P, a recovery point of root A. From P, one student in the group student:P continues with nothing added, and one repair in the group repair:P:1 continues after the reviewer's turn 1, inserted at P. If repair 1 has a verdict, a student at P was graded and nothing at P resolved, the review in slot search-r2 picks point Q1 in the history of repair 1, and repair 2 in the group repair:P:2 continues after turn 2, inserted at Q1. In the same way, the review in slot search-r3 picks point Q2 in repair 2, and repair 3 in the group repair:P:3 continues after turn 3, inserted at Q2. Root B, in the second slot and also in the group root, failed too and gets the same four branches at its own point. The task uses 2 roots plus 2 times 4 branches, 10 slots."/>
</p>
<p align="center"><sub>Figure 3: Under the launcher's settings, each failed root of an FF task gets one student and a chain of three repairs at one point.</sub></p>

For each point of the review after the roots, in the order of the review, `_plan_branches()` makes
these branches:

1. The students: `student_continuations` under `full`, or `tf_student_continuations` under
   `points_only`. A student is a branch with the guidance `none`. Nothing is added after the
   restored history.
2. Under `full` only, the repairs: `candidate_continuations` repairs for each repair turn, in the
   order of the turns. A repair is an `assistant-turn` branch that inserts the repair turn.

A repair has the `assistant_turn_rendering` `reasoning`. Before the repair runs, the worker moves
the `content` of the reviewer's turn into `reasoning_content`, and it leaves `content` empty
(`render_reasoning` in `src/sprout/contract/continuation.py`). The inserted turn then has the shape
of a turn that the reasoning policy samples itself.

Branches fill the free slots in the order that the search plans them. The search skips a branch or a
point in these cases, and it records each case in `search.skipped`:

| Case | Reason in `search.skipped` |
| --- | --- |
| No free slot is left for the branch. | `no free slot` |
| The job of the point is not a root of this group. | `not a root of this group` |
| The job of a later round's point is not a repair of this group. | `not a repair of this group` |
| The parent has no available recovery point at the tool step of the point. | `no available recovery point at that step` |

Each branch gets its own budget:

- `max_turns`: what its parent had left at the point. That is the parent's `max_turns` minus the
  `message_step` of the point, and at least 1.
- `deadline_at`: `wall_time_seconds` after the driver plans the branch.

The grade of a branch uses the grade template of its root.

`SearchSpec.branches()` is the most branches that any label can make:

```text
points × max(roots × (student_continuations + candidates × candidate_continuations × repair_rounds),
             (roots − 1) × tf_student_continuations)
```

Under `FF`, every root failed. Under `TF` and `FF_partial`, at most `roots − 1` roots failed.
`max_samples` must be at least `roots` + `branches()`. Miles allocates the slots by the same formula.
These examples apply the formula. The second column is the search of
`examples/rl_driver/search_result.example.json`. The third column is the search of
[the launcher](#the-launcher):

| Field | The shared example | The launcher |
| --- | --- | --- |
| `roots` | 2 | 2 |
| `points` | 1 | 1 |
| `candidates` | 2 | 1 |
| `student_continuations` | 2 | 1 |
| `candidate_continuations` | 2 | 1 |
| `tf_student_continuations` | 2 | 2 |
| `repair_rounds` | 1 | 3 |
| `branches()` | 1 × max(2 × (2 + 2 × 2 × 1), 1 × 2) = 12 | 1 × max(2 × (1 + 1 × 1 × 3), 1 × 2) = 8 |
| Slots: `roots` + `branches()` | 14 | 10 |

### Repair rounds

A later repair round gives a repair that did not resolve the task one more inserted turn, at a point
in its own history. Only the `full` shape has repair rounds.

When every branch of round r and its grade have settled, `_next_round()` looks for the points that
qualify. A point qualifies when both of these are true:

- A student at the point exported a trajectory and got a verdict.
- No branch at the point resolved the task. A repair counts as resolved only when its
  `touched_test_paths` is `[]`. A repair that changed a hidden test file does not count.

At each point that qualifies, the round-r repairs that have a verdict become the parents of the
review of round r + 1. That review has `round_no` r + 1. It has `points` 1 and `candidates` 1. It writes
one repair turn at one point in the history of each parent. Each new repair inserts that turn, and
it continues from the recovery point of its parent repair.

A repair of a later round follows these rules:

- With more than 1 round, `candidates` and `candidate_continuations` must both be 1.
- Students run in round 1 only.
- The repair belongs to the point where its chain started. It has that `point_id`, and its `group`
  is `repair:<point_id>:<n>`. n is (round − 1) × `candidates` + the number of the candidate. With
  `candidates` 1, n is the round.
- The repair also has these fields:
  - `round`
  - `chain_parent_job_id`: the parent repair
  - `chain_point_id` and `chain_step`: the point in the parent repair
- Its turn limit is what its parent repair had left at the new point.
- Its `leak_terms` is the union of the `leak_terms` of every turn on the chain. If one of them is
  `null`, it is `null`.

For a resolved chain, Miles makes one â sample from the inserted turn of each round. It splits the
share of that candidate equally among them. If a turn on the chain is not in the canonical
rendering, the whole chain is ineligible.

### Leak checks

With `review.leak_judge` in the adapter config, the review job checks each repair turn before it
takes the turn:

1. The review checks the turn with `leak_terms` and with Jev. Jev reads the test patch that the
   grade applied. The grade result gives its path as `test_patch`, and the driver passes it with
   `REVIEW_GRADE_KEYS`.
2. If either check flags the turn, the review sends the turn back to the reviewer to rewrite it. The
   reviewer learns only which branch stated specifics that the actor could not know, not which part.
3. The review repeats step 2 up to `resamples` times. The default is 2.
4. If the turn is still flagged after the last try, the review takes it. The metadata records the
   result.

The TypeSafe key is in the environment of the review worker, in `TYPESAFE_API_KEY` by default.
`leak_judge.api_key_env` names another variable.

### `metadata.search`

`metadata.search` describes the place of a trajectory in its search. It is `null` without a search.
Otherwise, it has these fields:

| Trajectory | Fields of `metadata.search` |
| --- | --- |
| Root | `kind: "root"`, `outcome` (the label), `root_counts` (`planned`, `graded`, `resolved`) |
| Student | `kind`, `point`, `point_id`, `step`, `message_step`, `parent_job_id`, `outcome`, `root_counts` |
| Repair | The fields of a student, and `candidate`, `multiplicity`, `samples`, `round`, `leak_terms`, `assistant_turn_rendering: "reasoning"`, and `leak_judge` when the review ran the judge |
| Repair of a later round | The fields of a repair, and `chain_parent_job_id`, `chain_point_id`, `chain_step` |

Some of these fields need a definition:

- `point` is the number of the point in the review, from 1. `step` is its tool step, and
  `message_step` is its message step.
- `multiplicity` is how many of the independent hindsight samples gave the candidate.
- `samples` is the number of hindsight samples at the point.
- `leak_judge` is the check of each inserted turn on the chain, in round order. Each check is
  `{leak, leak_terms, jev}`. A turn is clean only when `leak` is `false`. `true` means flagged, and
  `null` means that one of the checks failed to run. Without a judge, the key is absent.

Miles recomputes the label and the counts from the roots that it receives. It rejects a group whose
`outcome` or `root_counts` disagree.

`GET /miles-executions/{id}` gives the execution view of a Miles group. Its `search` has these keys,
among others:

| Key | Meaning |
| --- | --- |
| `state` | The search state. |
| `outcome`, `shape`, `root_counts` | The label, the shape and the counts of the roots. |
| `points` | The points of the review after the roots. |
| `skipped` | The skipped branches and points, with their reasons. |
| `error`, `chain_error`, `chain_end` | Why the search ended early, when it did. |
| `last_error` | The last failed Run Store call or malformed plan. |

### Stored search groups

`tf_student_continuations` is a newer field. Sprout and Miles both check the full key set of
`search`, so deploy them together. A search request that the driver stored before the upgrade lacks
the field. After the upgrade, the driver treats such a group in these ways:

- `GET` and `DELETE /rollout-groups/{id}` return 400.
- A search in its root phase ends after the roots.
- A search whose label calls for a review ends with the `search.error`
  `search document predates outcome-keyed review`.

Before you deploy, let every search group finish and be consumed.

## Provenance and hint removal

`metadata.provenance` says which `messages` the policy did not sample in this trajectory:

- `prefix_messages`: the number of leading messages that the branch restored from its point
- `inserted_messages`: the index of each assistant turn that a reviewer inserted, in this job or in
  an ancestor branch

A rollout from scratch has `0` and `[]`. The tool results of the inserted turns are real. Miles
does not train these turns as policy output in this trajectory. Miles's GSML trains only the
eligible inserted turns, separately, as â samples ([The Miles side](#the-miles-side)). A trajectory
without provenance fails. A trajectory whose messages are not in their cleaned form fails too,
because a second cleaning would shift the indices.

The worker exports the messages from its native history, with the tool calls and their results. In
a message-export rollout, the worker wraps the hint of a user-hint branch in
`<sprout_training_hint>` and `</sprout_training_hint>` before the branch runs. The export removes
these spans from system and user messages only. It treats a `developer` message as a system message.
It does not change the text of assistant or tool messages. It drops a message that held only a hint.
The export cannot reliably remove a hint without the markers.

The export accepts these roles:

- `system`
- `developer`
- `user`
- `assistant`
- `tool`

The export fails the trajectory in these cases (`clean_messages` in
`src/sprout/contract/messages.py`):

- a tool call without its result, or a result without its call
- content that is not text
- a role that the export does not accept
- an incomplete hint marker
- no assistant message

The driver treats the samples independently. It selects no branch and weights no sample.

Miles re-tokenizes the hint-free messages and computes the logprobs. These logprobs are not the
behavior probabilities of the sampling with the hint. For this path, do not turn on any of these:

- use-rollout-logprobs
- skipping the actor forward
- TIS
- rollout token replay

## The Miles side

This section describes the Miles branch `sprout-grpo`. The tests of this repository do not check it.

The Miles side is `miles.rollout.sprout.rollout_fn.SproutRolloutFn`. `--rollout-function-path`
selects it, and it brings its own `--sprout-rollout-*` arguments. Its example is in the Miles
directory `examples/swe-rebench-sprout/`. There, `prepare_data.py` writes Miles's `miles.jsonl` and
the config of this driver. The config pins the grade data and the parser by SHA-256.

Miles sends each prompt group as one request with `n_samples_per_prompt` slots. A search request
also has the slots of its branches. Miles treats the groups in these ways:

- A group without a search is normalized within the group as in GRPO. It must return in full.
- Miles leaves a group out of the step, and counts it in `rollout/sprout/groups_failed`, in these
  cases:
  - Sprout rejects the group.
  - The group fails in Sprout.
  - The group returns a shape that Miles cannot train.
- The step fails when every group fails or no group has a trainable sample.

Miles always sends `top_k` (`--rollout-top-k`, -1 when it is off), and Sprout forwards it unchanged.
Without it, the engine falls back to the `generation_config` of the checkpoint (`top_k` 20 for
Qwen3). The engine then truncates the sampling distribution while the trainer scores with the full
softmax. With `--rollout-top-p` below 1 or a positive `--rollout-top-k`, Miles derives
sampling-support replay. That replay cannot work with re-tokenized trajectories, so Miles refuses it
at construction.

### GSML credit

With `--sprout-rollout-search-points` above 0, each group is one search. The number of roots is
`n_samples_per_prompt`, which must be 2. GSML credits the samples in two steps:

1. `gsml.assemble_search_group` gives each sample a credit.
2. `rewards.post_process_gsml` (`--custom-reward-post-process-path`) turns the credit into the
   advantage of each trainable token of the sample: `A = β·z + λ·lambda_share`.

β is (1 − λ)/√2 for a root, and β_br for a branch. The advantage of an â sample is
`λ·ψ·lambda_share`. Miles computes z within the conditional group, over the members with a verdict
only.

Only `FF` tasks get λ. Miles splits λ by point:

- At a point where a student resolved, λ goes to the resolving students.
- At another point, λ goes to the repairs of eligible repair turns and to the â samples. An â sample
  is the inserted turn itself, as a single-turn target.

A repair whose `touched_test_paths` is not empty, or is not a list, counts as unresolved (R′ = 0). A
candidate whose `leak_terms` is not `[]` is ineligible. This includes `null`.

Miles checks these conditions at construction:

- λ is in [0, 1]. β_br and ψ are 0 or more.
- `--sprout-rollout-gsml-c-pre` and `--sprout-rollout-gsml-kappa-plus` are not implemented, and
  they accept only 0.
- The arguments include these three:
  - `--advantage-estimator grpo`
  - `--calculate-per-token-loss`
  - `--use-dynamic-global-batch-size`
- The arguments do not include `--normalize-advantages`.

Before Miles submits the first step, it checks that the chat template trains the `reasoning_content`
of the inserted turns (`gsml.check_reasoning_is_trained`).

With `--calculate-per-token-loss`, the loss is the sum over every trainable token of the step. Each
token is multiplied by the A of its sample, and each token has the same weight. Miles divides the sum
by the token count of the step. That count changes from step to step, and it includes the samples
with A = 0. The update is therefore consistent, not exact.

### The launcher

`run_qwen3_8_27b.py` trains on one machine with 8 GPUs. A TP 2 × CP 4 trainer shares the machine
with SGLang engines of 2 GPUs each. The launcher runs one search for each task, with the search and
GSML arguments below. A task then has at most 2 + max(2 × (1 + 3), 2) = 10 slots.

```text
--n-samples-per-prompt 2                            # the two roots of each task
--sprout-rollout-search-points 1                    # one point per failed root
--sprout-rollout-search-candidates 1                # FF: repair turns per point per round
--sprout-rollout-search-student-continuations 1     # FF: students per point
--sprout-rollout-search-candidate-continuations 1   # FF: repairs per repair turn
--sprout-rollout-search-tf-student-continuations 2  # TF, FF_partial: students per point
--sprout-rollout-search-repair-rounds 3             # FF: repair rounds
--sprout-rollout-search-review-seconds 900          # time limit of each round's review
--sprout-rollout-search-wall-time-seconds 900       # time limit of each round's branch phase
--custom-reward-post-process-path miles.rollout.sprout.rewards.post_process_gsml
--sprout-rollout-gsml-lambda 0.4 --sprout-rollout-gsml-beta-branch 1.0 --sprout-rollout-gsml-psi 1.0
--sprout-rollout-gsml-c-pre 0.0 --sprout-rollout-gsml-kappa-plus 0.0
--advantage-estimator grpo --calculate-per-token-loss --use-dynamic-global-batch-size
--rollout-top-p 1.0 --rollout-top-k -1
--loss-mask-type qwen3
```

Miles has its own defaults for some of these arguments:

- `--sprout-rollout-search-points` is 0, which means no search.
- `--sprout-rollout-search-candidates` is 4.
- Both time limits are 3600 seconds.
- The other `--sprout-rollout-search-*` and `--sprout-rollout-gsml-*` defaults are those of the
  launcher.

### A rollout without GPUs

`rollout_only.py` runs one complete rollout on a machine without GPUs. It does these steps:

1. It submits the request and waits for the result.
2. It re-tokenizes the messages with the chat template of the policy.
3. It computes the advantage that `train.py` would compute. Without a search, that is the GRPO
   normalization within the group. With a search, it is the GSML credit.
4. It saves the result in a format that `--load-debug-rollout-data` reads.

You cannot yet give such a file to `run_qwen3_8_27b.py --load-rollout-data` for training. The
launcher turns on `--use-dynamic-global-batch-size`, so Miles requires the rollout to carry
`dynamic_global_batch_size`. Miles records that value only when it trims a step to the data-parallel
size of the trainer. `rollout_only.py` has no trainer and does not trim
(`disable_rollout_trim_samples`). `convert_samples_to_train_data` then rejects the file with an
assertion.

## The internal execution interface: /execution-groups

`/execution-groups` is an entry point for debugging, not the Miles wire API. The Miles adapter
translates each Miles request into such a group too. Call the routes over HTTP.
`examples/rl_driver/execution-group.example.json` is a template of a group.

### Group requests

A group request has these fields (`validate_request` in `src/sprout/search/group.py`):

| Field | Rule |
| --- | --- |
| `rollout_job_id` | Required. A URL-safe id: 1 to 200 characters of `[A-Za-z0-9_.:-]`, starting with a letter or a digit. It is the identity of the group. |
| `prompt_group_id` | Required. A URL-safe id. |
| `context` | Optional object. Each run job and each review job gets it in `context.rl_driver.context`. |
| `samples` | 1 to 1000 samples. |

Each sample has a unique `sample_slot_id`, a URL-safe id. It has exactly one actor:

| Actor | Body | Rules |
| --- | --- | --- |
| `run` | A `JobSpec` of kind `rollout` (`RunSpec`) | It starts from scratch. Use `branch` to continue a job. It has no `affinity`. |
| `review` | A `JobSpec` of kind `review` (`ReviewSpec`) | It has no `grade` and no `affinity`. Its result is the actor result of the sample. |
| `branch` | `job_id`, `point_id` and `continuation`. Optional: `budget` and `overrides`. | `overrides` can set only `model`, `timeout_s` and `budget_usd`. |

The `context.rl_driver` of each run job and each review job records these values:

- `rollout_job_id`
- `prompt_group_id`
- `sample_slot_id`
- `context`: the `context` of the group

A branch's `continuation` says what the actor gets after the restored prefix. Its three modes are
those of the evaluation's `--branch-guidance` option, run by the same code. The section "Branch" of
[RUNSTORE.md](RUNSTORE.md) describes them:

- `assistant-turn`: the branch inserts an assistant turn and runs it.
- `user-hint`: the branch gets one user message that carries the hint.
- `none`: the branch gets nothing added.

The optional `budget` of a branch sets its own `max_turns`, `deadline_at` or both. It replaces the
values of the rollout contract that the branch inherits. The Run Store refuses a `budget` when the
parent job has no rollout contract.

### Grades

A `run` or a `branch` sample can have a `grade`: a grade JobSpec template. Its `spec` holds the
`GradeSpec` fields without `snapshot_id` and `in_place_job`, because the actor decides them. The
template has no `affinity`. For `swe-rebench-v2`, the dataset path and the parser path must be valid
on the worker.

After the actor succeeds, the driver finds the source of the grade:

1. If the template has `"in_place": true` beside its `spec`, the driver sets `in_place_job` to the
   job of the actor. The `harbor` grade then runs in the sandbox that the actor kept. The driver
   reads no snapshot. The actor needs `keep_sandbox: true` (the section "Kept sandboxes" of
   [RUNSTORE.md](RUNSTORE.md)).
2. Otherwise, if the result of the actor has a `final_snapshot_id`, the driver grades that snapshot.
3. Otherwise, the driver reads every tool call of the attempt. Each call must have a response, and
   their depths must be 1 to n without a gap.
4. The driver then takes the available recovery point at tool depth n with the largest
   `message_step`.

If the driver cannot prove the final tool boundary, the grade fails, and the driver submits no grade
job. The driver never grades an older snapshot instead. A grade with `resolved: false` is a valid
result, and it does not rerun the actor.

The `context.rl_driver` of the grade job records these values:

- `rollout_job_id` and `sample_slot_id`
- `actor_job_id` and `actor_attempt_id`
- `point_id`, for a grade from a snapshot. It is `null` for the final snapshot of the actor.

### Branches

To branch a job of a group, do these steps:

1. Read the group with `GET /execution-groups/{id}`. The actor record of each sample has its
   `job_id` and `attempt_id`.
2. Read the recovery points of the job from the Run Store: `GET /v1/jobs/{job_id}/recovery-points`.
3. Choose a point whose `available` is `true`.
4. Submit a new group, under a new group id, with a `branch` sample that names the `job_id` and the
   `point_id`.

The Run Store checks the native prefix and the snapshot of the point twice: when it accepts the
branch, and when a worker runs it. The driver never restores a sandbox or copies a session itself.

### Group status

The view of a group (`GET /execution-groups/{id}`) has one of these statuses:

| `status` | Condition |
| --- | --- |
| `queued` | The poller has not advanced the group yet. |
| `running` | The group is not ready. |
| `cancelling` | A cancellation was requested, and the group is not ready yet. |
| `completed` | The group is ready, and every actor and grade succeeded. |
| `failed` | The group is ready, and at least one actor or grade did not succeed. The other results stay. A group whose third run lost a job is also `failed`, with its `error` ([A lost host](#a-lost-host)). Its Miles result has no trajectory. |
| `cancelled` | The group is ready after a cancellation request. |

Only a group with `ready: true` is terminal. Each job of a group ends in the Run Store, also when
its worker dies ([RUNSTORE.md](RUNSTORE.md)).

The view records these values for each actor and each grade:

- `job_id` and `attempt_id`
- `state` and `phase`
- `result` and `error`
- `last_error`, after a failed Run Store call

The view also records the `affinity` of the group's current run. After a run again, it records
`earlier_runs` ([A lost host](#a-lost-host)).

The view leaves out the submission bodies and the templates, which the ledger keeps. The artifact
paths in a result are paths on the worker host. Read the events and the recovery points from the Run
Store.

### Stage states

The coordinator moves each stage of a group through the states in the figure. A stage is one of
these:

- the actor of a sample
- the grade of a sample
- a review of a search

A stage reaches the Run Store only under its idempotency key, so a retry never makes a second job.

<p align="center">
  <img src="figures/rl-driver-stage-states.svg" width="1000" alt="Nine states of one stage of a group (an actor, a grade or a review) and what moves a stage between them. Solid arrows are states that the driver sets. Dashed arrows are states that a poll copies from the Run Store job. The dotted arrow is the next run of the group, which the coordinator plans. planned goes to submitting when _dispatch() saves the intent before the POST. submitting goes to queued when the Run Store accepts the POST, or when reconcile-submission attaches the job. planned goes to skipped for the grade of an actor that did not succeed. planned goes to cancelled when the group is cancelled before the POST. planned goes to failed when _prepare_grade() cannot prove the source of a grade. submitting goes to failed when the Run Store refuses the POST with 400, 404, 409 or 422. queued goes to running when a worker claims the job. queued goes to failed with the reason data_lost when the host of the job's affinity key is lost. running goes back to queued when the attempt is lost or failed on the infrastructure and max_infra_retries leaves an attempt, or through release() when the worker shuts down. queued goes to cancelled after a stop request. running goes to succeeded when finish() records a completed or truncated result. running goes to failed when the attempt failed with no attempt left, or with the reason data_lost when the host of the job's key is lost. A grade in running goes to expired when it ran past grade_wall_seconds, and the driver stops its job. After a stage of the current run fails with the reason data_lost, run_again() plans the roots of the group again in a new run, and their stages start in planned. It does so at most 2 times, and not for a cancelled group. succeeded, failed, cancelled, skipped and expired are done."/>
</p>
<p align="center"><sub>Figure 4: The driver sets the states around a submission, each poll copies the state of the Run Store job, and a lost host starts a new run.</sub></p>

`Stages` in `src/sprout/search/stages.py` makes these transitions:

| From | Trigger and condition | Actor | To |
| --- | --- | --- | --- |
| `planned` | The group is not cancelled. A grade also needs a succeeded actor and a proved source. | `_dispatch()` saves the intent before the POST | `submitting` |
| `planned` | The group is cancelled. | `_hold()` or `_dispatch()` | `cancelled` |
| `planned` | The actor of a grade ended in a done state other than `succeeded`. | `Coordinator._advance_samples()` | `skipped` |
| `planned` | The driver cannot prove the source of a grade. | `_prepare_grade()` raises, and `advance()` records it | `failed` |
| `submitting` | The Run Store accepts the POST and returns the job id. | `_dispatch()` | `queued` |
| `submitting` | The Run Store refuses the POST with 400, 404, 409 or 422, and no earlier try was uncertain. | `_dispatch()` | `failed` |
| `submitting` | The POST fails without an answer, or with another status. | `_dispatch()` sets `submission_uncertain` | `submitting`. The next tick sends the POST again under the same key. |
| `submitting` | The group is cancelled, and the reply of the POST never arrived. | `_hold()` records `idempotency_key` and `last_error` | `submitting`, until `reconcile-submission` |
| `submitting` | An operator calls `reconcile-submission` with the job id. | `attach()` | the state of the job |
| `queued`, `running` | Each tick polls the job and copies its state. | `_poll()` | the state of the job |
| `running` | A grade of a Miles group has run for `grade_wall_seconds` since it left the queue. | `_expire()` records it and sends a stop request | `expired` |

The done states (`DONE`) are these:

- `succeeded`
- `failed`
- `cancelled`
- `skipped`
- `expired`

These rules also apply:

- A stage in a done state does not change.
- If a refusal comes after an uncertain try, the stage stays `submitting`, because the first try can
  have made the job.
- If the group is cancelled and the job is `queued` or `running`, `_poll()` sends a stop request
  before it reads the job again.
- A failed poll records `last_error` on the stage. The next tick polls again. A failed poll never
  makes a new job.
- A Miles job has `max_infra_retries` 0. It goes back from `running` to `queued` only when a worker
  shuts down and releases it.

The dashed transitions in the figure belong to the Run Store and its workers. A poll copies their
result into the stage. [RUNSTORE.md](RUNSTORE.md) describes them in full:

| From | Trigger and condition | Actor | To |
| --- | --- | --- | --- |
| `queued` | A worker that can run the job on its host has a free slot. | worker, `Store.claim()` | `running` |
| `queued` | No worker of the host of the job's affinity key reported a heartbeat for 600 s. | any worker, `Store.settle_dead_workers()` | `failed`, with the reason `data_lost` |
| `queued` | A stop request arrives. | `Store.request_stop()` | `cancelled` |
| `running` | The attempt ends with a completed or truncated result. | worker, `Store.finish()` | `succeeded` |
| `running` | The attempt failed, and no attempt is left. | worker, `Store.finish()` | `failed` |
| `running` | The attempt is lost or had an infrastructure failure, and `max_infra_retries` leaves an attempt. | worker, `Store.finish(rerun=True)` | `queued` |
| `running` | The worker died, and the lease of the job expired. | any worker, `Store.settle_dead_workers()` | `queued`, or `failed` when no attempt is left, or `failed` with the reason `data_lost` when the host of the job's key is lost. The group runs again ([A lost host](#a-lost-host)). |
| `running` | The worker shuts down. The attempt does not count. | worker, `Store.release()` | `queued`, or `cancelled` after a stop request |

The dotted arrow in the figure is the next run of the group. When a stage of the current run ends
`failed` with the reason `data_lost`, the coordinator does these steps at the end of the tick
(`run_again()`):

1. It keeps the stage, still `failed`, in `earlier_runs`.
2. It plans the stages of the roots again in a new run. Their state is `planned`.

The coordinator does not do this for a cancelled group. After 2 runs again, the group fails instead
([A lost host](#a-lost-host)).

## HTTP interface

With a driver token configured, each route, `/health` included, requires
`Authorization: Bearer <token>`. A request without the right token gets 401.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Returns `role` (`driver`) and `miles_configured`. |
| `POST /rollout-groups` | Submits a Miles request. Returns 202 with `rollout_job_id` and `status`. |
| `GET /rollout-groups/{id}` | Returns the Miles result: the trajectories and the rewards, once the group is ready. |
| `DELETE /rollout-groups/{id}` | For an active group, requests the cancellation and returns `status` `cancelled`. For a terminal group, records the acknowledgement. |
| `GET /miles-executions/{id}` | Returns the execution view behind a Miles group. |
| `POST /execution-groups` | Submits an execution group. Returns 202 with `rollout_job_id` and `status`. |
| `GET /execution-groups/{id}` or `GET /execution-groups/{id}/result` | Returns the view of the group: its status and the records of its actors and grades. |
| `DELETE /execution-groups/{id}` | For an active group, requests the cancellation and returns 202 with `status` `cancelling`. For a terminal group, records `acknowledged_at` once and returns 200. |
| `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` | Takes the body `{"phase": ..., "job_id": ...}`. See [Idempotency, restarts and cancellation](#idempotency-restarts-and-cancellation). |

The Miles routes take the Miles id as a path, so the id can hold `/`. The driver answers an error
with one of these statuses:

| Status | Condition |
| --- | --- |
| 400 | A request on a `/rollout-` route is not valid. |
| 401 | The bearer token is missing or wrong. |
| 404 | The group, the sample or the execution reference is unknown. |
| 409 | The id already names a different request. |
| 422 | A request on another route is not valid. |
| 502 | A Run Store query failed. Retry the query. The driver submits nothing twice. |
| 503 | A Miles route was called, and the config has no `miles` section. |

## Idempotency, restarts and cancellation

The driver submits each job once, even after a lost reply or a restart:

- `rollout_job_id` is the identity of a group. A repeat of the same request returns the original
  group. A different request under the same id gets 409.
- The ledger id of a Miles group is `miles-` and the SHA-256 of its Miles id. The ledger id of each
  slot is made in the same way. Each driver computes the same ids (`internal_id`).
- Before each POST, the coordinator saves the stage as `submitting` in the ledger.
- Each POST carries a fixed idempotency key (`Submitter.key()`). The key is `rl-driver/` and a
  SHA-256 hash of these values:
  - the affinity key of the group's current run ([Host affinity](#host-affinity)): the group id, or
    `<group id>-r<n>` after the n-th run again
  - the sample id
  - the phase
- A retry after a lost reply therefore makes no second job.
- The driver never makes a new attempt after an execution failure. `JobSpec.max_infra_retries` and
  the worker decide the infrastructure retries. The Miles adapter sets 0.
- After a restart, the driver follows the same jobs. The ledger keeps each group that is not
  terminal as active.

The poller ticks at these times:

- every `poll_interval_s` seconds
- at once after `Driver.submit()`
- at once after `Driver.release()`
- at once after `Driver.reconcile_submission()`

A tick advances the active groups, oldest first. A tick returns at once
while another tick runs. A failed Run Store call for one group does not stop the other groups. When
the app stops, it waits up to 45 s for the poller.

A `DELETE` on an active group cancels the group. The group deadline cancels a Miles group too.
After a cancellation, the driver does these steps:

1. It submits no stage that is still `planned`. The state of such a stage becomes `cancelled`.
2. At the next poll of each `queued` or `running` job, it sends `POST /v1/jobs/{id}/stop`. The Run
   Store cancels a queued job. The worker of a running job stops it and reclaims its sandboxes.
3. It follows each job to a done state.
4. It plans no new review or repair round.

The reply of a submission can still be unknown at the cancellation. The driver then does not send
the POST again, because that POST can make a new job. The view of the stage shows its
`idempotency_key`.

Recover such a stage with these steps:

1. Find the job with that idempotency key in the Run Store.
2. If the group is a Miles group, read its ledger id and the ledger id of the sample with
   `GET /miles-executions/{id}`.
3. Send `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` with the ledger ids and
   the body `{"phase": "actor", "job_id": "<job id>"}`. For a grade, use the phase `grade`. For a
   review, use the phase `review` and the sample `search` or `search-r<r>`.

The driver checks that the job has the idempotency key of the stage. Then it attaches the job and
follows it. It makes no job, and it does not change the state of the job. If the stage is not an
unacknowledged submission, or the job has another key, the call fails with 422.

A `DELETE` on a terminal group records a receipt of the consumption only. It deletes none of
these:

- a ledger record
- execution data
- a snapshot

The driver has no TTL and no garbage collection. The trainer
should deduplicate by the identity of the group and the sample itself.

## Host affinity

The jobs of one group run on one host, because the later jobs of a group read the data that the
earlier jobs left on their host ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)):

- A branch starts from a snapshot of its parent.
- A review reads the attempt directories of the failed roots.
- A grade reads the final snapshot or the kept sandbox of its actor.

The group document keeps the key of the group's current run in `affinity`. Its value is the group
id, and `<group id>-r<n>` after the n-th run again ([A lost host](#a-lost-host)). The coordinator
applies the key in these ways:

- Each job that the coordinator submits with `POST /v1/jobs` carries the key as its `affinity`. These
  jobs are the runs, the grades and the reviews.
- A branch carries no key of its own. It takes the key of the job that it continues.
- The driver refuses a group request whose run, review or grade template sets `affinity`.

The Run Store binds a key to the host of the first worker that claims a job of the key. After that,
only a worker on that host claims a job of the key.

### A lost host

A host is lost when no worker on it reported a heartbeat for 600 s, whether its workers died or
stopped ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). The Run Store then fails
each unfinished job of each key that is bound to that host. The state of such a job is `failed`,
and its `result.reason` is `data_lost`. The later jobs of the group cannot read the data of the lost
host, so the coordinator runs the group again from its roots.

When a poll copies such a job into a stage of the current run, the coordinator does these steps at
the end of the tick (`run_again()` in `src/sprout/search/rerun.py`):

1. It appends the current run to `earlier_runs` in the group document: its `affinity`, its `samples`
   and its `search`.
2. It sets `affinity` to `<group id>-r<n>` for the n-th run again.
3. It keeps only the roots: the samples of the group request. Their actors and grades are `planned`.
4. It starts the search again as the group started it: `state` `roots`, no review and no point, and
   every branch slot free, in the same order.

The next tick submits the roots again. Each stage gets a new idempotency key from the new
`affinity`, so the Run Store makes a new job for each stage.

These rules also apply:

- The coordinator stops no job of an earlier run. The Run Store keeps every job, attempt and
  result.
- If an actor or a grade lost its job, the search does not advance in that tick. It reviews no root
  of the lost run, and it plans no branch.
- A job that the coordinator submitted in the tick that saw the lost job belongs to the earlier run.
  The coordinator does not poll it again.
- A cancelled group does not run again. Its lost job is a failed stage.
- The group keeps its `deadline_at`, and each root keeps the `deadline_at` of its rollout contract.
  A root of a later run gets only the time that is left.
- A group runs again at most 2 times (`RUNS_AGAIN`). If a job of the third run loses its data, the
  group fails: `ready` is `true`, `status` is `failed`, and `error` names the keys of the three
  runs. The Miles adapter exports no sample of such a group ([Results](#results)).

The view of a group leaves out the submission bodies of the earlier runs too.

## Verification

Run the tests of the driver and the search from the Sprout root:

```bash
python3.12 -m pytest tests/rl_driver tests/search tests/worker/run/test_rollout_runs.py -q
```

The tests use these Run Stores:

- Most tests of `tests/rl_driver` use the fake Run Store of `tests/rl_driver/fake_runstore.py`.
- Most tests of `tests/search` use the fake Run Store of `tests/search/fake_runstore.py`.
- `tests/rl_driver/test_runstore_fidelity.py` checks that the driver does the same on the fake as on
  the real Run Store. `tests/search/test_fake_runstore.py` checks the answers of the search fake.
- If `SPROUT_RUNSTORE_TEST_DSN` names a separate test database, each fidelity test also runs on the
  real side. It reaches the Run Store API over TCP, on PostgreSQL, in a schema of its own.

These tests need no model server. They start no sandbox, and they run no verifier. Without
`SPROUT_RUNSTORE_TEST_DSN`, the real side skips. A skip does not mean that the wiring passed.

The full chain runs without GPUs. It has these parts:

- PostgreSQL
- the Run Store: `serve` and a worker with a podman profile
- this driver
- Miles's `rollout_only.py`, on CPU, with only the tokenizer of the policy

Any endpoint that serves Chat Completions can be the model. The chain runs the real agent with its
snapshots, and the real SWE-rebench grading. It stops before the optimizer.
