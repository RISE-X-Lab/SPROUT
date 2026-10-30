# RL driver

The RL driver runs a trainer's rollout groups as Run Store jobs, and it returns their trajectories
and rewards. It serves HTTP on `127.0.0.1:11001`, and it calls the Run Store API at
`127.0.0.1:18110`. The driver only schedules the work, and the Run Store's workers run every job.
The search coordinator in `src/sprout/search/` branches the failed roots of a task.

Read this document to connect Miles to Sprout, to start and configure the driver, or to change the
search. [RUNSTORE.md](RUNSTORE.md) describes the jobs themselves and their states.

## Architecture: the driver schedules, the workers execute

The driver keeps a record of each group, and it reaches the jobs only through the Run Store API. A
worker claims each job from the Run Store and runs it. Figure 1 shows the components and the
requests between them.

<p align="center">
  <img src="figures/rl-driver-flow.svg" width="1000" alt="Twelve components in three boundaries, and the requests between them. Miles sends POST, GET and DELETE requests on /rollout-groups to the HTTP API of the RL driver process, create_app() on port 11001. Inside the driver, the HTTP API hands the Miles routes to the Miles adapter, MessageAdapter, and the execution routes and the poller's tick() to the execution groups, Driver. MessageAdapter calls Driver with submit() and release(). Driver calls tick() on the search coordinator, Coordinator with Stages and Search. The coordinator calls active(), get() and save() on the group documents, Ledger, one SQLite file. The coordinator sends HTTP requests on /v1/jobs to the Run Store API, sprout.runstore serve on port 18110, to submit, branch, poll and stop jobs. The Run Store API reads and writes PostgreSQL through Store. PostgreSQL holds jobs, attempts, events and recovery points. The worker, python -m sprout.worker start, claims and finishes jobs in PostgreSQL directly. It runs three kinds of job: a rollout job (Runner, mini-swe-agent and Sprout tools in a podman or microvm sandbox), a grade job (the SWE-rebench grader, in two sandboxes) and a review job (SPROUT's analyst and reviewer). The rollout job sends Chat Completions requests through the inference gateway to the model server at the request's model_endpoint, and the review job sends its model calls to the same server."/>
</p>
<p align="center"><sub>Figure 1: Miles calls the driver, the driver calls the Run Store API, and the workers claim the jobs from PostgreSQL.</sub></p>

These steps take one Miles group from the request to the result:

1. Miles sends `POST /rollout-groups`. `MessageAdapter` in `src/sprout/rl_driver/messages.py`
   checks the request, and `Driver.submit()` in `src/sprout/rl_driver/driver.py` records the group
   in the ledger, one SQLite file (`Ledger` in `src/sprout/rl_driver/ledger.py`).
2. At each tick of the poller thread of `create_app()` in `src/sprout/rl_driver/server.py`,
   `Coordinator` in `src/sprout/search/coordinator.py` submits and polls the jobs of the group.
3. A worker runs a rollout for each root slot, and then a grade for each rollout. With a `search`,
   a review and the branches that it proposes follow. Each model call goes through the worker's
   inference gateway to the request's `model_endpoint`.
4. When every job has settled, Miles reads the result and acknowledges it.

The driver starts no other process. When the driver stops, its jobs keep running, and a restarted
driver follows them again. The worker profile selects the sandbox: a rootless podman container by
default, or an AgentENV microVM. The evaluation uses the same coordinator, with JSON files instead
of the ledger ([EVALUATION.md](EVALUATION.md)).

## Start and configuration

Before you start the driver, start the Run Store API and at least one worker. Then do these steps
from the root of the Sprout checkout:

1. Install the dependencies of the driver: `python3.12 -m pip install -e ".[rl]"`.
2. Export the Run Store token: `export SPROUT_RUNSTORE_TOKEN='<existing-runstore-token>'`.
3. Start the driver: `python3.12 -m sprout.rl_driver --config resources/rl_driver/config.example.json`.
   `GET /health` then returns `{"role": "driver", "miles_configured": true}`.

`--host` and `--port` change the address. The command checks the config before it serves. If a
check fails, the command exits with a usage error. `python -m sprout.services up rl-driver` starts
the driver as a service, with the database and the Run Store API, but without a worker. The service
reads `~/.config/sprout/rl_driver.json`, which `python -m sprout.services init` writes
([SERVICES.md](SERVICES.md)).

### The config file

`resources/rl_driver/config.example.json` shows each field. The top level configures the driver
itself, and the `miles` section configures the Miles adapter. Without `miles`, the Miles routes
return 503. These fields need care:

| Field | Rule |
| --- | --- |
| `driver_token_env` | The variable of the callers' bearer token. `null` turns authentication off. The example config sets `null`, so use it only for local access. |
| `ledger` | Required. A relative path starts from the directory of the config file. |
| `miles.run_defaults` | RunSpec fields for each root rollout. `slot` must be `mini-swe-agent`. `model` applies when the request names none. |
| `miles.image_resources` | Required: the `cpu` and the `memory_mb` of each sandbox. |
| `miles.tasks` | For each `task_id`, a `grade` job template without `snapshot_id`. The adapter refuses a request for a task without one. |
| `miles.api_key_env` | The variable in the worker's environment that holds the key of the model endpoint. The worker profile's `worker_env` must set it. |
| `miles.grade_wall_seconds` | The longest time of one grade: 600 by default, and at least 240. |
| `miles.review` | The worker `profile` of the reviews, and the reviewer's model-call settings (`REVIEW_CONFIG_FIELDS` in `src/sprout/rl_driver/messages.py`) with the defaults of `ReviewSpec`. A search request needs it. |

The driver applies these rules to the tokens and the ledger:

- The driver reads each token from the environment only, and it writes no token to the ledger. It
  refuses a group with an inline credential, such as an `api_key` value. Use `{"$env": "<NAME>"}`.
- On its first use, the ledger binds to `runstore_url`. A later start with another URL fails.
- One driver process holds the ledger at a time, through a lock on `<ledger>.lock`.

## The Miles protocol

Miles sends one request for each prompt group to `/rollout-groups`, and it reads one result for each
group. The adapter checks each request before it records the group. If a check fails, the driver
refuses the request, and no job exists for it.

### Requests

`MessageRequest` in `src/sprout/rl_driver/messages.py` defines the fields of a request, and
`examples/rl_driver/request.example.json` is a complete one. The adapter refuses an unknown field. A
request gives these things:

- the group: its identity `rollout_job_id`, and `rollout_id` and `prompt_group_id`
- the task: a `task_id` with a `grade` in `miles.tasks`, the `image`, and the `prompt`
- the policy: `model_endpoint`, an HTTP(S) URL that serves `/v1/chat/completions`, an optional
  `model`, and optional `sampling_params` (`validate` in `src/sprout/contract/sampling.py`)
- the samples: `sample_slots`, and `minimum_returned_samples` ≤ `max_samples` ≤ the number of slots
- the limits: `max_turns`, `budgets.max_wall_time_seconds` and `finalization_timeout_seconds`
- an optional `search` ([Search](#search-hindsight-branches-after-the-root-rollouts))

`max_turns` limits the model calls of each trajectory. The tool calls of a response take no turn,
and the worker does not limit them.

The adapter makes one rollout job and one grade job for each root slot. Without a search, the roots
are the first `max_samples` slots. The rollout runs in the worker profile `miles.profile`, with
`miles.image_resources`. Its `extra.rollout_contract` asks the worker to export the messages. If
`image` starts with `prime/primeintellect/`, the adapter replaces that prefix with
`docker.io/swerebenchv2/`, as Prime's dataset documents. Each job has `max_infra_retries` 0, so it
gets one counted attempt ([Job states](RUNSTORE.md#job-states)).

### Time limits

The adapter computes the group deadline when it accepts the request:

```text
deadline_at = submission time + max_wall_time_seconds + finalization_timeout_seconds + search_seconds
search_seconds = R × (review_seconds + wall_time_seconds) + (R − 1) × grade_wall_seconds
```

R is `repair_rounds` when `candidates` is more than 0, and 1 otherwise. Without a search,
`search_seconds` is 0. The finalization time, 1800 s by default, covers the work after the
execution, such as the grades of the last round. These limits have effects of their own:

| Limit | Counted from | At the limit |
| --- | --- | --- |
| `max_wall_time_seconds` | The submission. Time in the queue counts. | The worker stops each root with the `stop_reason` `timeout`. The driver grades it. |
| `grade_wall_seconds` | When the grade job leaves the queue. | The driver records the grade as `expired` and stops its job. The sample is missing. |
| `deadline_at` | The submission. | The coordinator cancels the group. The result is `cancelled`, with no trajectories. |

### Grading

Each grade is a Run Store grade job. A `swe-rebench-v2` template pins the dataset and the log parser
on the worker by SHA-256. Only a worker with these files claims the grade
([Grade files](RUNSTORE.md#grade-files)). The worker takes the agent's diff from the final snapshot,
and it runs the tests in a fresh sandbox of the task image (the section "Official grading" of
[RUNSTORE.md](RUNSTORE.md)). The driver uses these values of a completed grade:

- `resolved`: `true` only when every FAIL_TO_PASS and PASS_TO_PASS test passes. It gives the reward.
- `progress`: the share of the FAIL_TO_PASS tests that passed, or 0 if a PASS_TO_PASS test fails.
- `touched_test_paths`: the files of the hidden test patch that the agent's diff also changes.

A failure that the agent causes, such as a diff that does not apply, is a completed grade with
`resolved: false`. A grade that the worker cannot make fails, and the Run Store does not retry it.
Its sample is then missing.

### Results

Miles polls `GET /rollout-groups/{id}` until `status` is not `queued` or `running`. Then it stores
the result and sends `DELETE /rollout-groups/{id}`, which records the acknowledgement. The first
read of a ready group exports the result and stores it in the ledger. Each later read returns the
stored result, also after a restart. `examples/rl_driver/result.example.json` shows each field. The
result has one of these statuses:

| `status` | Condition |
| --- | --- |
| `queued` | The poller has not advanced the group yet. |
| `running` | The group is not ready. |
| `completed` | Each sample of the group exported a trajectory. |
| `early_stopped` | At least `minimum_returned_samples` trajectories, and at least one entry in `failed_samples`. |
| `failed` | Fewer than `minimum_returned_samples` trajectories. Or the group lost a host in each of its runs ([A lost host](#a-lost-host)): then it has no trajectory, and `stop_reason` is the `error` of the group. |
| `cancelled` | A `DELETE` or the group deadline cancelled the group before the first read of the ready group. The result has no trajectories. |

`failed_samples` gives the slot and the reason of each sample that exported no trajectory
(`export_problem` in `src/sprout/search/outcome.py`). With a search, Miles sends
`minimum_returned_samples` 1, so a failed root does not fail the group. These fields of a trajectory
need a definition:

| Field | Meaning |
| --- | --- |
| `reward` | 1.0 when the grade resolved the task, and 0.0 otherwise. An invalid attempt, which stopped with `format_invalid`, `response_truncated` or `repetition`, is not graded and gets -1.0 (Arctic-SWE's Table 18). A harness rollout in training mode that a budget ended before `submit` is not graded either, and gets 0.0 (its `require_submit: false` grades it). |
| `group` | `root`, `student:<point_id>` or `repair:<point_id>:<n>` in a search, and `null` without one. Miles computes z-scores within a group. |
| `status`, `stop_reason` | `completed`, or `truncated` with `max_turns_reached`, `timeout`, `max_total_tokens`, `interrupted`, or an invalid attempt's reason. The worker cuts an interrupted rollout, which failed midway, at its last valid recovery point. A truncated rollout gets a real grade, unless it is an invalid attempt or an unfinished training-mode harness rollout. |
| `metadata.zero_reason` | `null` when the grade gave a verdict. Otherwise the trajectory is missing, and the value says why: `grade_failed`, `grade_expired`, `grade_cancelled` or `grade_no_verdict` (`zero_reason` in `src/sprout/search/outcome.py`). A missing trajectory has the reward 0.0, because the wire needs a number, but Miles leaves it out of a search group. |
| `metadata.provenance` | See [Provenance and hint removal](#provenance-and-hint-removal). |
| `metadata.search` | See [`metadata.search`](#metadatasearch). |

Sprout and Miles test against the same three files in `examples/rl_driver/`.
`tests/rl_driver/test_miles_rollouts.py` and `tests/search/test_miles_examples.py` check that the
driver returns exactly `result.example.json` and `search_result.example.json`. Miles's
`test_gsml_wiring.py` checks that its copies are byte-for-byte equal. If you change an example file
here, copy it into the fixtures of Miles.

## Search: hindsight branches after the root rollouts

A search spends its extra samples only on the tasks whose roots failed. The roots run first. Their
label then decides whether a review runs and which branches follow. A request asks for a search with
the field `search` (`SearchSpec` in `src/sprout/search/spec.py`), which has all of these fields:

| Field | Value | Role |
| --- | --- | --- |
| `roots` | 1 or more | The rollouts from scratch. They take the first `roots` slots. |
| `points` | 1 or more | The most branch points that a review picks in each parent. |
| `candidates` | 0 or more | Under `full`: the repair turns at each point. 0 means no repairs. |
| `student_continuations` | 1 or more | Under `full`: the students at each point. |
| `candidate_continuations` | 0 or more, and 0 exactly when `candidates` is 0 | Under `full`: the repairs from each repair turn. |
| `tf_student_continuations` | 0 or more | Under `points_only`: the students at each point. |
| `repair_rounds` | 1 or more. More than 1 only when `candidates` and `candidate_continuations` are 1. | Under `full`: the rounds of repairs. |
| `review_seconds` | finite, more than 0 | The `timeout_s` of each review job. |
| `wall_time_seconds` | finite, more than 0 | The time of each branch, from when the driver plans it. |

A search request must also meet these conditions:

- `minimum_returned_samples` ≤ `search.roots`, because only the roots are sure to run.
- `search.roots` + `SearchSpec.branches()` ≤ `max_samples` ([Branches and slots](#branches-and-slots)).
- The adapter config has a valid `review` section. The adapter builds the review job templates when
  it accepts the request, so a bad review config fails the request at once.

Sprout and Miles both check the full key set of `search`. A search group that a driver stored before
`tf_student_continuations` existed ends after its roots, and its `GET` returns 400. So deploy a
change of these fields to both together, after every stored search group has finished and was
consumed.

### The label of the roots decides the shape

When every root has settled, `search_outcome` in `src/sprout/search/outcome.py` labels the roots by
three counts:

- `planned`: `search.roots`
- `graded`: the roots that exported a trajectory and got a verdict
- `resolved`: the graded roots whose verdict resolved the task

| Label | Roots | Shape | What runs next |
| --- | --- | --- | --- |
| `TT` | All are graded, and all resolved. | `none` | No review and no branches. |
| `TT_partial` | Every graded root resolved, and some are missing. | `none` | No review and no branches. |
| `ungraded` | None is graded. | `none` | No review and no branches. |
| `TF` | Some resolved, and some failed. | `points_only` | A review that only picks points, then `tf_student_continuations` students at each point. |
| `FF_partial` | None resolved, some failed, and some are missing. | `points_only` | The same as `TF`. |
| `FF` | All are graded, and none resolved. | `full` | The full review, then students and repairs at each point. |

If `tf_student_continuations` is 0, the shape of `TF` and `FF_partial` is `none` (`search_shape`).
Miles's `miles.rollout.sprout.gsml.search_outcome` is the same function, and Miles recomputes the
label from the roots that it receives. A change of these rules needs the same change in Miles.

### Search states

A search moves through five states. Each review and each round of branches waits until everything
before it has settled. Figure 2 shows the functions of `Search` in `src/sprout/search/rounds.py`
that move `search.state`. `Search.advance()` runs at each tick, after the samples advance.

<p align="center">
  <img src="figures/rl-driver-search-states.svg" width="1000" alt="The five states of a group's search and the functions of Search that move it between them. roots goes to reviewing through _review_roots() when every root has settled, the shape of the roots' label is not none and the group is not cancelled. roots goes to done when the shape is none, when the group is cancelled, or for a stored search document from before outcome-keyed review. reviewing goes to branching through _plan_branches() when the review succeeded and at least one branch was planned. reviewing goes to done when the review did not succeed or no branch could be made. branching goes to chaining through _next_round() when every branch has settled, the shape is full, a repair round is left, the group is not cancelled and a point qualifies. branching goes to done when there is no next round. chaining goes back to branching through _plan_chain() when the review of the round succeeded and at least one branch was planned. chaining goes to done when that review did not succeed or no branch could be made. done is terminal. In done, search.error, search.chain_error or search.chain_end, when it is set, gives the reason."/>
</p>
<p align="center"><sub>Figure 2: A search runs the roots, a review, the branches, and then one review and one set of branches for each later repair round.</sub></p>

In `done`, `search.error` gives the reason for an end after the roots or their review, when it is
set, for example `every root resolved the task`. `search.chain_end` or `search.chain_error` gives
the reason for an end in a later round. These rules also apply:

- A search in `branching` waits for every branch of the group, not only for the last round.
- If a Run Store call fails or a review plan is malformed, the search keeps its state and records
  the error in `search.last_error`. The next tick tries the same step again.
- A group is ready only when its search is `done` and every stage has settled. A search that ended
  early returns the samples that it has, so `actual_samples` can be less than `max_samples`.

### The review

The review is one Run Store `review` job: SPROUT's analyst and reviewer, run by the policy model
under the hindsight prompts (the section "Review" of [RUNSTORE.md](RUNSTORE.md)). Each review
reviews each parent on its own (`points_per_attempt: true`), and it picks at most `points` points in
each parent. The search submits these reviews:

| Review | Sample id of its key | Parents | `guidance`, `candidates` | What it writes |
| --- | --- | --- | --- | --- |
| After `FF` roots: `full` | `search` | Each root whose verdict is `false` | `assistant-turn` and `candidates`, or `none` and 1 when `candidates` is 0 | At each point, at most `candidates` repair turns. It merges the hindsight samples that propose the same commands into one turn. |
| After `TF` or `FF_partial` roots: `points_only` | `search` | The same | `none`, 1 | The points only. |
| Repair round r ≥ 2: `chain` | `search-r<r>` | Each repair of round r − 1 that qualifies ([Repair rounds](#repair-rounds)) | as `full`, 1, with `points` 1 | One point in each parent repair, and one repair turn at it. |

The review reads the grade of each parent through `REVIEW_GRADE_KEYS` in
`src/sprout/search/rounds.py`. Only its leak checks read the keys about the hidden tests (the
section "Leak checks" of [RUNSTORE.md](RUNSTORE.md)). The `leak_terms` of a repair turn are the
hidden test names or paths that the turn mentions and that the history before the point never
showed. `[]` is clean. `null` means that the grade had no facts to compare against.

### Branches and slots

Each branch continues a root, or a repair, from a recovery point that a review picked. Figure 3
shows the branches of one failed root under the settings of [the launcher](#the-launcher).

<p align="center">
  <img src="figures/rl-driver-search-tree.svg" width="1000" alt="The trajectories that a search makes from one failed root of an FF task, under the launcher's search settings: one point, one student, one repair turn with one repair, and three repair rounds. Root A, in the first slot and in the group root, failed. The review in slot search picks point P, a recovery point of root A. From P, one student in the group student:P continues with nothing added, and one repair in the group repair:P:1 continues after the reviewer's turn 1, inserted at P. If repair 1 has a verdict, a student at P was graded and nothing at P resolved, the review in slot search-r2 picks point Q1 in the history of repair 1, and repair 2 in the group repair:P:2 continues after turn 2, inserted at Q1. In the same way, the review in slot search-r3 picks point Q2 in repair 2, and repair 3 in the group repair:P:3 continues after turn 3, inserted at Q2. Root B, in the second slot and also in the group root, failed too and gets the same four branches at its own point. The task uses 2 roots plus 2 times 4 branches, 10 slots."/>
</p>
<p align="center"><sub>Figure 3: Under the launcher's settings, each failed root of an FF task gets one student and a chain of three repairs at one point.</sub></p>

For each point of the review after the roots, in review order, `_plan_branches()` makes these
branches:

1. The students: `student_continuations` under `full`, or `tf_student_continuations` under
   `points_only`. A student has the guidance `none`, so nothing follows the restored history.
2. Under `full` only, the repairs: `candidate_continuations` repairs for each repair turn, in turn
   order. A repair is an `assistant-turn` branch that inserts the repair turn.

Before a repair runs, the worker moves the `content` of the inserted turn into `reasoning_content`
(`render_reasoning` in `src/sprout/contract/continuation.py`). The turn then has the shape of a turn
that the reasoning policy samples itself.

Branches fill the free slots in the order that the search plans them. A branch gets the grade
template of its parent. Its `max_turns` is the parent's `max_turns` minus the `message_step` of the
point, and at least 1. Its `deadline_at` is `wall_time_seconds` after the driver plans it. If no
slot is left, or the parent has no available recovery point at the step of the point, the search
skips the branch or the point. It records the reason in `search.skipped`.

`SearchSpec.branches()` is the most branches that any label can make. `max_samples` must be at least
`roots` + `branches()`, and Miles allocates the slots by the same formula:

```text
points × max(roots × (student_continuations + candidates × candidate_continuations × repair_rounds),
             (roots − 1) × tf_student_continuations)
```

Under `FF`, every root failed. Under `TF` and `FF_partial`, at most `roots − 1` roots failed. The
second column is the search of `examples/rl_driver/search_result.example.json`:

| Field | The shared example | The launcher |
| --- | --- | --- |
| `roots`, `points` | 2, 1 | 2, 1 |
| `candidates`, `candidate_continuations` | 2, 2 | 1, 1 |
| `student_continuations`, `tf_student_continuations` | 2, 2 | 1, 2 |
| `repair_rounds` | 1 | 3 |
| `branches()` | 1 × max(2 × (2 + 2 × 2 × 1), 1 × 2) = 12 | 1 × max(2 × (1 + 1 × 1 × 3), 1 × 2) = 8 |
| Slots: `roots` + `branches()` | 14 | 10 |

### Repair rounds

A later repair round gives a repair that did not resolve the task one more inserted turn, at a point
in its own history. Only the `full` shape has repair rounds. When every branch of round r and its
grade have settled, `_next_round()` looks for the points that qualify. A point qualifies when both
of these are true:

- A student at the point exported a trajectory and got a verdict.
- No branch at the point resolved the task. A repair counts as resolved only when its
  `touched_test_paths` is `[]`.

At each point that qualifies, the round-r repairs that have a verdict are the parents of the review
of round r + 1. Each new repair inserts the turn of that review, and it continues from the recovery
point in its parent repair. A repair of a later round follows these rules:

- It keeps the `point_id` of the point where its chain started. Its `group` is
  `repair:<point_id>:<n>`, where n is the round.
- Its turn limit is what its parent repair had left at the new point.
- Its `leak_terms` is the union of the `leak_terms` of every turn on the chain, or `null` if one of
  them is `null`.

### `metadata.search`

`metadata.search` gives the place of a trajectory in its search, and it is `null` without a search.
`examples/rl_driver/search_result.example.json` shows it for each kind of trajectory. Each kind
carries the label (`outcome`) and the `root_counts`. Miles rejects a group whose label or counts
disagree with the roots that it receives. These fields need a definition:

- `point` is the number of the point in the review, from 1. `step` is its tool step, and
  `message_step` is its message step.
- `multiplicity` is how many of the `samples` hindsight samples at the point gave the candidate.
- `leak_judge`, when the review ran the judge, is the Jev check of each inserted turn on the chain,
  in round order. A turn is clean only when its `leak` is `false`.
- `chain_parent_job_id`, `chain_point_id` and `chain_step` name the parent repair of a later round,
  and the point in it.

`GET /miles-executions/{id}` shows the whole search of a Miles group, with its points and its
skipped entries.

## Provenance and hint removal

`metadata.provenance` says which `messages` the policy did not sample in this trajectory:

- `prefix_messages`: the number of leading messages that the branch restored from its point
- `inserted_messages`: the index of each assistant turn that a reviewer inserted, in this job or in
  an ancestor branch

A rollout from scratch has `0` and `[]`. Miles does not train the inserted turns as policy output in
this trajectory, but their tool results are real. A sample without provenance exports no
trajectory. A sample whose messages are not in their cleaned form exports none either, because a
second cleaning would shift the indices.

Before a user-hint branch runs, the worker wraps the hint in `<sprout_training_hint>` markers. The
export (`clean_messages` in `src/sprout/contract/messages.py`) removes the marked spans from system
and user messages only, and it drops a message that held only a hint. If `clean_messages` refuses
the history, the trajectory fails.

Miles re-tokenizes the hint-free messages and computes the logprobs. These logprobs are not the
behavior probabilities of the sampling with the hint. For this path, do not turn on any of these:

- use-rollout-logprobs
- skipping the actor forward
- TIS
- rollout token replay

## The Miles side

This section summarizes the Miles branch `sprout-grpo`. The tests of this repository do not check
it. `--rollout-function-path` selects `SproutRolloutFn`, which brings its own `--sprout-rollout-*`
arguments. In the Miles directory `examples/swe-rebench-sprout/`, `prepare_data.py` writes Miles's
`miles.jsonl` and the config of this driver. Miles treats the groups in these ways:

- Miles normalizes a group without a search within the group, as in GRPO. Such a group must return
  in full.
- Miles leaves a group out of the step, and counts it in `rollout/sprout/groups_failed`, when Sprout
  rejects or fails the group, or returns a shape that Miles cannot train.
- In a search group, Miles makes no sample from a missing trajectory. Only
  `rollout/sprout/ungradable` counts it.

Miles always sends `top_k`, -1 when it is off. Without it, the engine uses the `top_k` of the
checkpoint's `generation_config`, 20 for Qwen3, while the trainer scores with the full softmax.

With `--sprout-rollout-search-points` above 0, each group is one search with 2 roots. GSML
(`gsml.assemble_search_group` and `rewards.post_process_gsml`) gives each trainable token the
advantage `A = β·z + λ·lambda_share` of its sample, with z within the conditional group. Only `FF`
tasks get λ. At a point where a student resolved, λ goes to the resolving students. At another
point, it goes to the eligible repairs and to their â samples: each inserted turn as a single-turn
target. A repair whose `touched_test_paths` is not `[]` counts as unresolved. A candidate whose
`leak_terms` is not `[]`, `null` included, is ineligible.

### The launcher

`run_qwen3_8_27b.py` trains on one machine with 8 GPUs: a TP 2 × CP 4 trainer beside SGLang engines
of 2 GPUs each. It runs one search for each task, with the settings of the
[slot table](#branches-and-slots) and 900 s for each review and each round's branches. It sets λ to
0.4, and β_br and ψ to 1.0. `rollout_only.py` runs one complete rollout and computes its advantage
on a machine without GPUs. It saves the result in the format of `--load-debug-rollout-data`.

## The internal execution interface: /execution-groups

`/execution-groups` is an entry point for debugging, not the Miles wire API. The Miles adapter
makes such a group from each Miles request too. `validate_request` in `src/sprout/search/group.py`
checks a group, and `examples/rl_driver/execution-group.example.json` is a template of one. Each
sample of a group has exactly one of these actors:

- `run`: a rollout job from scratch
- `review`: a review job, without a grade
- `branch`: a continuation of a recovery point (the section "Branch" of [RUNSTORE.md](RUNSTORE.md))

A run or a branch can have a `grade` template. No job of a group can set `affinity`
([Host affinity](#host-affinity)).

After the actor succeeds, `_prepare_grade()` in `src/sprout/search/stages.py` finds the source of
its grade:

1. If the template has `"in_place": true` beside its `spec`, the grade runs in the sandbox that the
   actor kept. The actor needs `keep_sandbox: true` (the section "Kept sandboxes" of
   [RUNSTORE.md](RUNSTORE.md)).
2. Otherwise, if the result of the actor has a `final_snapshot_id`, the driver grades that snapshot.
3. Otherwise, the driver grades the available recovery point after the last tool call, if every tool
   call has a response and their depths have no gap. If not, the grade fails without a job. The
   driver never grades an older snapshot instead.

To branch a job of a group, do these steps:

1. Read the `job_id` and the `attempt_id` of the actor from `GET /execution-groups/{id}`.
2. Choose a point whose `available` is `true` in `GET /v1/jobs/{job_id}/recovery-points` of the Run
   Store.
3. Submit a new group with a `branch` sample that names the `job_id` and the `point_id`.

The view of a group (`GET /execution-groups/{id}`) has one of these statuses:

| `status` | Condition |
| --- | --- |
| `queued` | The poller has not advanced the group yet. |
| `running` | The group is not ready. |
| `cancelling` | A cancellation was requested, and the group is not ready yet. |
| `completed` | The group is ready, and every actor and grade succeeded. |
| `failed` | The group is ready, and at least one actor or grade did not succeed. A group that lost a host in each of its runs is also `failed`, with its `error`. |
| `cancelled` | The group is ready after a cancellation request. |

Only a group with `ready: true` is terminal. The view leaves out the submission bodies, which the
ledger keeps. After a failed Run Store call, a stage of the view has a `last_error`.

### Stage states

The coordinator moves each stage of a group through the states of Figure 4. A stage is the actor of
a sample, the grade of a sample, or a review of a search. A stage reaches the Run Store only under
its idempotency key, so a retry never makes a second job.

<p align="center">
  <img src="figures/rl-driver-stage-states.svg" width="1000" alt="Nine states of one stage of a group (an actor, a grade or a review) and what moves a stage between them. Solid arrows are states that the driver sets. Dashed arrows are states that a poll copies from the Run Store job. The dotted arrow is the next run of the group, which the coordinator plans. planned goes to submitting when _dispatch() saves the intent before the POST. submitting goes to queued when the Run Store accepts the POST, or when reconcile-submission attaches the job. planned goes to skipped for the grade of an actor that did not succeed. planned goes to cancelled when the group is cancelled before the POST. planned goes to failed when _prepare_grade() cannot prove the source of a grade. submitting goes to failed when the Run Store refuses the POST with 400, 404, 409 or 422. queued goes to running when a worker claims the job. queued goes to failed with the reason data_lost when the host of the job's affinity key is lost. running goes back to queued when the attempt is lost or failed on the infrastructure and max_infra_retries leaves an attempt, or through release() when the worker shuts down. queued goes to cancelled after a stop request. running goes to succeeded when finish() records a completed or truncated result. running goes to failed when the attempt failed with no attempt left, or with the reason data_lost when the host of the job's key is lost. A grade in running goes to expired when it ran past grade_wall_seconds, and the driver stops its job. After a stage of the current run fails with the reason data_lost, run_again() plans the roots of the group again in a new run, and their stages start in planned. It does so at most 2 times, and not for a cancelled group. succeeded, failed, cancelled, skipped and expired are done."/>
</p>
<p align="center"><sub>Figure 4: The driver sets the states around a submission, each poll copies the state of the Run Store job, and a lost host starts a new run.</sub></p>

`Stages` in `src/sprout/search/stages.py` makes the solid transitions. The dashed transitions belong
to the Run Store and its workers ([Job states](RUNSTORE.md#job-states)), and a poll copies them into
the stage. A stage in a done state does not change. The figure leaves out these cases:

| Case | What the coordinator does |
| --- | --- |
| The POST gets no answer, or a status other than 400, 404, 409 and 422. | `_dispatch()` sets `submission_uncertain`. The stage stays `submitting`, and the next tick sends the POST again under the same key. |
| The Run Store refuses the POST after an uncertain try. | The stage stays `submitting`, because the first try can have made the job. |
| The group is cancelled, and the reply of the POST never arrived. | `_hold()` records `idempotency_key` and `last_error`. The stage stays `submitting` until [`reconcile-submission`](#idempotency-restarts-and-cancellation). |
| The group is cancelled, and the job is `queued` or `running`. | `_poll()` sends a stop request before it reads the job again. |
| A poll fails. | The stage records `last_error`. The next tick polls again. A failed poll never makes a new job. |

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
| `GET /execution-groups/{id}` or `GET /execution-groups/{id}/result` | Returns the view of the group: its status, and the job, the attempt, the state and the result of each actor and grade. |
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
- The ledger id of a Miles group, and of each of its slots, is `miles-` and the SHA-256 of the Miles
  id (`internal_id`). Each driver computes the same ids.
- Before each POST, the coordinator saves the stage as `submitting` in the ledger. The POST carries
  a fixed idempotency key (`Submitter.key()`): `rl-driver/` and a SHA-256 hash of the
  [affinity key](#host-affinity) of the current run, the sample id and the phase.
- The driver never makes a new attempt after an execution failure. `JobSpec.max_infra_retries` and
  the worker decide the infrastructure retries.
- After a restart, the driver follows the same jobs, because the ledger keeps each group that is not
  terminal as active.

The poller ticks every `poll_interval_s` seconds, and at once after each change through `Driver`. A
tick advances the active groups, oldest first. A failed Run Store call for one group does not stop
the other groups.

A `DELETE` on an active group cancels the group. The group deadline cancels a Miles group too.
After a cancellation, the coordinator submits no stage that is still `planned`. At the next poll of
each `queued` or `running` job, it sends `POST /v1/jobs/{id}/stop`, and it follows the job to a done
state. It plans no new review or repair round. A `DELETE` on a terminal group records the
acknowledgement only, and it deletes nothing. The driver has no garbage collection, so the trainer
should deduplicate by group and by sample itself.

If the reply of a submission is still unknown at the cancellation, the driver does not send the POST
again, because that POST can make a new job. Recover such a stage with these steps:

1. Find the job with the stage's `idempotency_key` in the Run Store. The view of the stage shows the
   key.
2. If the group is a Miles group, read its ledger id and the ledger id of the sample with
   `GET /miles-executions/{id}`.
3. Send `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` with the ledger ids and
   the body `{"phase": "actor", "job_id": "<job id>"}`. For a grade, use the phase `grade`. For a
   review, use the phase `review` and the sample `search` or `search-r<r>`.

The driver attaches the job only when the job has the idempotency key of the stage. It makes no job,
and it does not change the state of the job. Otherwise the call fails with 422.

## Host affinity

The jobs of one group run on one host, because the later jobs of a group read the data that the
earlier jobs left on their host ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)).
Figure 5 shows that data, and the runs of a group whose host is lost.

<p align="center">
  <img src="figures/rl-driver-host-affinity.svg" width="1000" alt="The runs of one group and the host that holds their data. In run 1, every job carries the affinity key of the group id, and the Run Store binds the key to host A at the first claim. The root rollouts write their data on host A: snapshots, attempt directories and kept sandboxes. On the same host, a grade reads the final snapshot or the kept sandbox, a review reads the attempt directories of the failed roots, and a branch starts from a snapshot of its parent. When no worker on host A reported a heartbeat for 600 s, host A is lost, the Run Store fails each unfinished job of the key with the reason data_lost, and run_again() starts run 2 under the key group id -r1, with the roots planned again and run 1 kept in earlier_runs. If the host of run 2 is lost, run_again() starts run 3 under the key group id -r2, with runs 1 and 2 kept in earlier_runs. If the host of run 3 is lost too, the group is failed: ready is true, its error names the three keys, and Miles gets no trajectory. A group runs again at most 2 times. A branch takes the key of the job that it continues, and a cancelled group does not run again."/>
</p>
<p align="center"><sub>Figure 5: The jobs of a run read their data on the host of the run's key, and a lost host runs the group again from its roots, at most 2 times.</sub></p>

The group document keeps the key of the current run in `affinity`: the group id, and
`<group id>-r<n>` after the n-th run again. Each job that the coordinator submits with
`POST /v1/jobs` carries the key. A branch takes the key of the job that it continues. The Run Store
binds a key to the host of the first worker that claims a job of the key. After that, only a worker
on that host claims a job of the key.

### A lost host

A host is lost when no worker on it reported a heartbeat for 600 s. The Run Store then fails each
unfinished job of each key that is bound to that host, with the `result.reason` `data_lost`. When a
poll copies such a job into a stage of the current run, `run_again()` in
`src/sprout/search/rerun.py` does these steps at the end of the tick:

1. It appends the current run to `earlier_runs`: its `affinity`, its `samples` and its `search`.
2. It sets `affinity` to `<group id>-r<n>` for the n-th run again.
3. It keeps only the roots. Their actors and grades are `planned` again.
4. It starts the search again in `roots`, with every branch slot free, in the same order.

The next tick submits the roots again under new idempotency keys, so the Run Store makes new jobs.
A cancelled group does not run again. The coordinator stops no job of an earlier run, and the Run
Store keeps them all. The group keeps its `deadline_at`, so a root of a later run gets only the
time that is left. After 2 runs again (`RUNS_AGAIN`), the next loss fails the group, and its
`error` names the keys of the three runs.

## Verification

Run the tests of the driver and the search from the Sprout root:

```bash
python3.12 -m pytest tests/rl_driver tests/search tests/worker/run/test_rollout_runs.py -q
```

Most tests use the fake Run Store of `tests/rl_driver/fake_runstore.py` or of
`tests/search/fake_runstore.py`. They need no model server and no sandbox.
`tests/rl_driver/test_runstore_fidelity.py` checks that the driver does the same on the fake as on
the real Run Store. The real side runs only if `SPROUT_RUNSTORE_TEST_DSN` names a separate test
database. Otherwise it skips, and a skip does not mean that the wiring passed. The full chain, with
Miles's `rollout_only.py` and a podman worker, runs without GPUs and stops before the optimizer.
