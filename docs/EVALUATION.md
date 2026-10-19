# Evaluation and the task gate

`python -m sprout.eval.evaluate` evaluates one Harbor task with one method.
`python -m sprout.eval.gate` checks that the verifiers of a task set give the expected
rewards. Both commands are clients of the [Run Store](RUNSTORE.md), like the
[RL driver](RL_DRIVER.md). Each rollout, grade, review and branch is a Run Store job, and a
worker runs it ([ADR-0002](decisions/0002-eval-and-gate-through-run-store.md)). Neither
command starts a sandbox or runs the agent itself.

Read this document to start the services, choose the options and read the output. If you
change a method, read [The jobs of each method](#the-jobs-of-each-method) first.

## An evaluation is a client of the Run Store

The client process builds the jobs, the search coordinator submits and follows them, and the
worker runs them. Run the client on the worker's host, because the client reads the files
that the jobs leave in the worker's artifact directory. The figure shows the components and
the calls and files between them.

<p align="center">
  <img src="figures/evaluation-components.svg" width="900" alt="The worker's host holds the client process, the Run Store, the worker, the sandboxes, the artifact directory and the output directory. In the client process, Evaluation or Gate builds group requests and hands them to Groups and the search Coordinator, which submit the jobs to the Run Store API over HTTP, poll them and stop them. The Run Store API keeps the jobs in PostgreSQL. The worker claims the jobs in PostgreSQL and finishes them there. The worker starts the sandboxes, on podman or a microVM, and runs the agent and the verifier in them. The worker sends the model requests of rollouts and reviews to the model endpoint, outside the host. The worker writes journals, verifier logs and review plans into its artifact directory, and the client process reads them there. The client process writes summary.json, the group documents in groups/ and the plans into the output directory. For bpo and shepherd, the selector in the client process sends its own requests to the model endpoint."/>
</p>
<p align="center"><sub>The client submits jobs to the Run Store over HTTP, the worker runs them, and the client reads the worker's files on the same host.</sub></p>

Each component has one role:

| Component | Code | What it does |
|---|---|---|
| client | `Evaluation` in `evaluate.py`, `Gate` in `gate.py` | builds the group requests from the task (`jobs.py`), and writes `summary.json`, the plans and `gate.jsonl` |
| coordinator | `Groups` in `groups.py`, `Coordinator` in `sprout.search` | submits each job, polls it, and keeps one group document for each group in `groups/` |
| Run Store | `python -m sprout.runstore serve` | keeps the jobs, their attempts and their recovery points in PostgreSQL |
| worker | `python -m sprout.worker start` | claims each job, runs it, and writes its files into the artifact directory |

A group is one request to the coordinator. It holds one or more samples. Each sample is a
run, a branch or a review, and a run or a branch can have a grade. The coordinator follows
these rules:

- The coordinator submits each job under an idempotency key that starts with
  `eval/<evaluation id>/` or `gate/<run id>/`. The rest of the key comes from the affinity key,
  the sample id and the stage. A second submission of the same stage makes no second job.
- Each run, grade and review job carries the affinity key of its group document as its
  `affinity`. The Run Store runs the jobs of one key on one host. A branch takes the key of the
  job that it continues ([RL_DRIVER.md](RL_DRIVER.md#host-affinity)). The client gives the key,
  as [Affinity keys](#affinity-keys) says.
- The coordinator submits the grade of a sample only after its rollout or branch succeeded. If
  the rollout ends in another final state, the coordinator marks the grade `skipped`.
- If a job of a group ends `failed` with the reason `data_lost`, its host was lost. The
  coordinator does not run the group again. It sets the group's `error`, and the group is ready
  and `failed` at once. The client runs the search again ([A lost host](#a-lost-host)).
- A group is ready when each of its jobs is `succeeded`, `failed`, `cancelled`, `skipped` or
  `expired`, or when a job of it ended with the reason `data_lost`.
- Between two passes over the groups that are not ready, the client waits `--poll-interval`
  seconds (default 5).

### Affinity keys

All jobs of one task's search carry one affinity key, so they run on the host that holds the
search's data ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). A review reads the
attempt directories of the parent and the branches. A grade reads the kept sandbox or the final
snapshot of its rollout. Each client makes its keys in this way:

| Client | Key | Where the output keeps it |
|---|---|---|
| evaluation | `eval-<32 hex digits>`, one for each evaluation, for each of its groups | `affinity` in `summary.json` |
| evaluation with `--parent-output` | the `affinity` of the recorded evaluation's `summary.json` | `affinity` in `summary.json` |
| gate | `gate-<32 hex digits>`, one for each task of each gate run, for both checks of the task | `affinity` in each record of `gate.jsonl` |

A method that uses `--parent-output` takes the recorded key, because each branch carries the key
of the parent. Its review and its grades must run on the same host. If the recorded
`summary.json` names no key, the evaluation raises `ValueError` before it submits a job.

The groups of one evaluation share the key, so each sample has an id of its own in the
evaluation. The review of round `<N>` is the sample `review-<N>`.

### A lost host

When the host of a key is lost ([Run Store](RUNSTORE.md#lost-hosts)), the Run Store fails each
unfinished job of the key, with the reason `data_lost`. The later jobs of the search cannot read
the data of the lost host. So the client runs the search again under a new key, `<key>-r<n>` for
the n-th run again, at most 2 times.

When a group of an evaluation is ready with a job that ended `data_lost`, `Evaluation` in
`evaluate.py` does these steps:

1. If the evaluation uses `--parent-output`, it stops and raises `DataLost`. The parent belongs to
   the recorded evaluation, and its data was on the lost host.
2. If the evaluation ran again 2 times, it stops and raises `DataLost`.
3. It moves each entry of `--output`, except `summary.json` and `earlier-runs/`, into
   `earlier-runs/<key>/`. `<key>` is the key of the run that lost its data. It deletes no file.
4. It appends the run to `earlier_runs` in `summary.json`, with its `affinity`, its `attempts`
   and its `error`.
5. It sets `affinity` to the next key and `attempts` to an empty list.
6. It runs the parent again, then the method. The new key gives each stage a new idempotency
   key, so the Run Store makes a new job for each stage.

After `DataLost`, `summary.json` records `complete: false`, the `error`, and `reason: "data_lost"`.
The process exits with a non-zero status.

These rules also apply:

- The evaluation stops no job of an earlier run. The Run Store keeps every job, attempt and
  result.
- An attempt of an earlier run does not count for `resolved`, even if it resolved the task.

The gate follows the same rule for each task of a gate run, with one difference: it runs again
only the check that lost its job. A check reads no data of another check, so a check that settled
keeps its record. When the group of a check is ready with a job that ended `data_lost`, the gate
does one of these:

- If the checks of the task ran again 2 times, the gate records the check as failed, with
  `reason: "data_lost"`. Then it goes on to the task's next mode.
- Otherwise, the gate writes no record for the lost run. It runs the check again before the
  task's other modes, under the task's next key, in the group `<task>-<mode>-r<n>`.

Each later check of the task runs under the task's latest key, in the group of that key. The group
documents of each run stay in `groups/<run id>/`.

## Before you start

You need these services:

- a Run Store API at `--runstore-url` (default `http://127.0.0.1:18110`). The client reads
  the bearer token from the variable that `--runstore-token-env` names (default
  `SPROUT_RUNSTORE_TOKEN`).
- a worker that serves these profiles:
  - `--profile` for rollouts (default `mini`)
  - `--grade-profile` for grades (default `grade`)
  - `--review-profile` for reviews, with `--method sprout` only (default `review`)

`python -m sprout.services up runstore-api runstore-worker` starts both services, after
the services that they depend on ([services](SERVICES.md)). The worker profiles decide where
the sandboxes run (podman or microVM), the runtime binary and the store. The evaluation and
the gate have no options for them.

Make sure that these conditions are true before you start:

- The worker's environment holds the keys that `--model-key-env`, `--review-key-env` and
  `--leak-judge-key-env` name.
- For `bpo` and `shepherd`, the environment of the client also holds the key of
  `--model-key-env`. The selector sends its own model requests from the client.
- The client runs on the worker's host.
- For an evaluation, the git worktree of the source is clean. The evaluation records its
  commit as `source_commit`. If `git status --porcelain` prints a change, the evaluation
  raises `RuntimeError` before it submits a job.
- For an evaluation, `--output` is not under `/tmp`, `/var/tmp`, `/dev/shm` or `/run`. The
  summary and the group documents are the only record of the evaluation's jobs, and the OS
  empties these directories. `--volatile-ok` lets you use one of them anyway.

If the token variable is empty, the client raises `ValueError` before it submits a job. For
`bpo` and `shepherd`, the client does the same when the variable of `--model-key-env` is
empty.

## How a Harbor task becomes jobs

Each attempt is a rollout job and a grade job. `sprout.eval.jobs` builds both from the task,
as `sprout.contract.harbor_task.load_task` reads it
([ADR-0003](decisions/0003-sprout-runs-harbor-verifier.md)).

Before any job exists, the client refuses a task when the image is not the task's whole
environment. `check_environment` raises `ValueError` for each of these tasks:

- a task without a `docker_image`
- a task without `environment/Dockerfile` whose `environment/` holds files. Harbor uploads
  these files into the sandbox, and a rollout does not.
- a task whose `environment/Dockerfile` is `FROM` the task's own image and adds `COPY` or
  `RUN` steps, or other steps that a sandbox cannot replay

A `Dockerfile` `FROM` another image is the recipe of the prebuilt image, and the client
accepts it.

The rollout job of a task has these fields:

| Field | Value |
|---|---|
| `kind`, `profile` | `rollout`, and `--profile` |
| `agent` | `mini` in an evaluation, `oracle` or `nop` in the gate |
| `prompt` | the task's `instruction.md` without its leading canary lines |
| `harbor_task` | the task directory. The worker starts the sandbox on the task's agent network policy. An `oracle` rollout runs the task's `solution/solve.sh`. |
| `sandbox_image` | the task's `docker_image` |
| `sandbox_resources` | the task's CPUs and memory, when the task sets them |
| `timeout_s` | the task's agent timeout, or 3600 s when the task sets none |
| `checkpoint_enabled` | in an evaluation, true unless `--checkpoint-mode none`. In the gate, false. Each rollout ends with one disk snapshot. |
| `keep_sandbox` | true when the verifier shares the agent's environment |
| `backend` | `image_env` and `image_user`, in both the `podman` and the `microvm` section |
| `max_infra_retries` | 0, because one attempt is one execution |
| `extra.rollout_contract` | `mini` only: the endpoint, the model, the key's variable, `max_tokens`, `reasoning_effort`, `max_turns` and `deadline_at` |
| `extra.mini.environment` | `mini` only: the command timeout `--command-timeout`, and the task's `workdir` as `cwd` when the task names one |

The `deadline_at` of a `mini` rollout is 2 days after its agent timeout, counted from the
time of submission. The worker cuts the rollout's timeout to the time left before `deadline_at`. If
`deadline_at` has passed when a worker takes the rollout, the worker does not start the run.
A wait that long means that no worker serves the profile.

The grade job runs the task's own verifier on the rollout's work. Its spec has these
fields:

- the benchmark `harbor`
- the task directory, as `task_dir`
- `image_env` and `image_user` in `backend`, from the flags of the evaluation or the gate
- a timeout of the task's verifier timeout plus the timeouts of its `[[verifier.collect]]`
  commands

Where the grade runs depends on the task:

| Condition | Where the grade runs | Retries |
|---|---|---|
| The verifier shares the agent's environment. | in the sandbox that the rollout kept: `in_place: true`, which the coordinator turns into `in_place_job` ([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)) | none, because the first attempt destroys the kept sandbox |
| The verifier has an environment of its own. | from the rollout's final snapshot, `snapshot_id` | up to 2 for an infrastructure failure |

A grade in the kept sandbox has these other paths:

- If the kept sandbox is gone, or its 20 minutes pass before the grade runs, the grade scores
  0 with the reason `sandbox_lost`.
- On a microVM profile, the grade fails, because a kept microVM sandbox cannot switch its
  egress to the verifier's network policy.

Both jobs set `image_env` and `image_user` in their `backend`. Then the agent's commands,
`solve.sh` and `test.sh` run as the image's `USER` with its `ENV`, as in Harbor. The worker
reads only the section of the backend that its profile names. [Run Store](RUNSTORE.md)
describes the two settings and the `harbor` grade.

## The jobs of each method

The methods differ only in how they choose the branch points. Each method starts from one
parent attempt, which is a `mini` rollout and its grade. `sprout` then reviews and branches
in rounds. `bpo` and `shepherd` branch from the parent of a baseline evaluation, one point at
a time. The figure shows the jobs and the groups of each method.

<p align="center">
  <img src="figures/evaluation-method-jobs.svg" width="900" alt="Three lanes, one for each kind of method. In the baseline lane, the group parent holds a rollout job of the agent mini and, after the rollout succeeded, a grade job with the benchmark harbor. Then the evaluation ends. With --parent-output, the evaluation reuses the parent of that evaluation and runs no job. In the sprout lane, the group parent holds the same rollout and grade, or the evaluation reuses them with --parent-output. After the parent's grade succeeded, the group review-N holds one review job, which reads each attempt so far and picks k points. Each point becomes a branch job in the group round-N, and each branch job that succeeded gets a grade job. A back edge goes from the grades of round N to the review of the next round, while no attempt resolved the task and N is less than --rounds. In the bpo and shepherd lane, the parent is the parent of a baseline evaluation, not a new job. A selector in the client process, not a job, reads the parent's journal and picks the points. Each point becomes its own group b-NN-step-S with one branch job and, after the branch succeeded, its grade job. A back edge goes from the grade to the next branch, while no attempt resolved the task and branches are left. Blue boxes are rollout jobs, a branch included. Green boxes are grade jobs. The yellow box is a review job. Gray boxes are not jobs."/>
</p>
<p align="center"><sub>Each method runs or reuses one parent attempt, then sprout reviews and branches in rounds, and bpo and shepherd branch from a baseline's parent one point at a time.</sub></p>

With `--parent-output`, a method reuses the parent of that evaluation and runs no parent
job. `bpo` and `shepherd` require it.

| Method | Parent | Branch points | Branches | The method ends when |
|---|---|---|---|---|
| `baseline` | runs it, or reuses it | none | none | the parent's grade settles |
| `sprout` | runs it, or reuses it | A review job in each round reads each attempt so far and picks the points. | one branch job and its grade for each point, all of a round in one group | an attempt resolves the task, `--rounds` rounds have run, or a review gives no branch |
| `bpo` | reuses it | The selector ranks the parent's checkpoints by the policy's first-token entropy. | `--max-rollouts` − 1 branch jobs, one group each, one after another | a branch resolves the task, or each selected branch has run |
| `shepherd` | reuses it | A meta-agent request picks one checkpoint step for all branches. | the same as `bpo` | the same as `bpo` |

A branch job keeps the spec of the job that it branches from, with these settings in it:

- the agent
- the image settings
- the command timeout

The evaluation gives each branch its own `deadline_at`, 2 days after the agent timeout,
counted from the time of submission. The grade of a branch takes the flags of the evaluation
that makes it. `summary.json` records these flags as `image_env` and `image_user`.

### sprout rounds

For each round `<N>`, from 1 to `--rounds`, the evaluation does these steps:

1. If an attempt has resolved the task, the evaluation stops.
2. The evaluation runs one review job in the group `review-<N>`. The job gets each attempt
   row so far and the round's width. The width is the `<N>`-th value of `--branches`, or
   the last value for a later round.
3. The review picks up to the width in points with `--branch-count-mode adaptive`, and
   exactly the width with `fixed`.
4. For each point, the evaluation looks for an available recovery point of the point's
   attempt at the point's tool step.
5. Each point that has a recovery point, and a turn when the guidance needs one, becomes a
   branch in the group `round-<N>`. The branch is named after the point's `run_name`.
6. The coordinator submits all branches of the round at once. The worker's concurrency
   decides how many run at the same time.
7. After all branches of the round and their grades settle, the evaluation adds a row for
   each branch and saves `summary.json`.

A point makes no branch in these cases. `plan-round<N>.json` lists it under `skipped`, with
the reason:

| Case | Reason |
|---|---|
| The guidance is `assistant-turn`, and the review wrote no turn at the point. | `the review wrote no turn there` |
| The attempt has no available recovery point at the point's step. | `no available recovery point` |

If no point of a round makes a branch, the evaluation stops. It sets `stop_reason` to the
plan's `validation_error`, or to `the review selected no branches`. It sets
`branch_schedule_complete` to false.

Each branch gets what its guidance gives:

- `assistant-turn` (default): the turn that the review wrote at the point
- `user-hint`: the review's hint for the point
- `none`: nothing. The branch samples again from the recovery point.

### bpo and shepherd branches

The evaluation does these steps:

1. The selector reads the parent's journal on this host and finds the parent's exact `mini`
   checkpoints. If there is none, the evaluation raises `RuntimeError`.
2. The selector picks `--max-rollouts` − 1 branches:
   - `bpo` asks the endpoint for the first-token log probabilities at each checkpoint that a
     later model request follows. It runs `--bpo-scoring-workers` requests at a time. It
     takes up to `--bpo-max-points` checkpoints, highest entropy first, at least
     `--bpo-min-spacing` completion tokens apart. It deals the branches over them in turn.
   - `shepherd` sends one meta-agent request with the parent's transcript. The answer names
     one eligible checkpoint step and a reason. All branches start at that step.
3. The evaluation writes `plan.json`.
4. For each selected branch in turn, the evaluation runs the branch and its grade in a group
   of its own, `b<NN>-step<S>`. The branch gets no guidance.
5. After each branch, the evaluation adds its row with round 1, and saves `summary.json`.
6. If the branch resolved the task, the evaluation stops.

The selector sends each model request in up to 8 tries, with a wait of up to 30 s between
two tries. It records each request with its tries and its response in the output. If the
selector makes no valid selection, it raises `ValueError`. If the parent has no available
recovery point at a selected step, the evaluation raises `RuntimeError`.

## Evaluate one task

```bash
python -m sprout.eval.evaluate --task-dir ~/datasets/deep-swe/tasks/<task> \
    --dataset-digest sha256:... --preparation-manifest runs/prep.json \
    --output runs/<task>-sprout --method sprout \
    --model <model> --model-endpoint <url> --model-key-env OPENAI_API_KEY --rounds 2 --branches 4 3
```

The evaluation does these steps:

1. It checks the options. For a combination that no evaluation can run, it exits with
   status 2 before any job exists.
2. It checks the token and, for `bpo` and `shepherd`, the model key in its environment. If
   one is empty, it raises `ValueError`.
3. It reads the task and records the commit of the source. If the worktree of the source is
   not clean, it raises `RuntimeError`.
4. It refuses an `--output` that is not empty, with `FileExistsError`.
5. It refuses a task that `check_environment` refuses, with `ValueError`.
6. It refuses a task that the preparation manifest does not pin, with `ValueError`.
7. It sets the affinity key ([Affinity keys](#affinity-keys)). With `--parent-output`, it reads
   the key of the recorded evaluation. If that `summary.json` names none, it raises `ValueError`.
8. It writes the first `summary.json`.
9. It runs the parent: one `mini` rollout and its grade, in the group `parent`. With
   `--parent-output`, it reuses the parent of a completed evaluation instead.
10. For a method other than `baseline`, it checks that the parent is a `mini-swe-agent` run.
    If not, it raises `ValueError`.
11. It runs the method, as [The jobs of each method](#the-jobs-of-each-method) says. If a job
    loses its host, it runs steps 9 to 11 again, as [A lost host](#a-lost-host) says.
12. It checks that the task's `task.toml` did not change. If it changed, the evaluation
    raises `RuntimeError`.
13. It writes `summary.json` with `complete: true`, prints the summary without the
    `attempts`, and exits with status 0.

The preparation manifest pins the task when each of these conditions is true:

- Its `dataset_digest` equals `--dataset-digest`.
- Exactly one row of its `tasks` has the task's `path` and `name`.
- The `task_toml_sha256` of that row equals the sha256 of the task's `task.toml`.

`examples/qwen/run.py` writes such a manifest.

The evaluation reuses the parent of `--parent-output` only when each of these conditions is
true. Otherwise it raises `ValueError`.

- The recorded evaluation ran the same task of the same dataset digest.
- Its `summary.json` has `complete: true`.
- Its parent row has a `job_id`. A parent of a local Harbor trial has none.
- It used the same `--runstore-url`.
- The parent's journal is still on this host.

The evaluation writes `summary.json` at these times:

- before the first job
- after the parent
- after each `sprout` round
- after each `bpo` or `shepherd` branch
- when a lost host makes the evaluation run again
- when an error ends the evaluation
- at the end, with `complete: true`

These are the paths other than the normal path:

- If a rollout does not succeed, or its grade gives no numeric reward, the evaluation raises
  `RuntimeError`. The error names the job. A review job that does not succeed does the same.
- If a job ends `failed` with the reason `data_lost`, the evaluation runs again from a new
  parent, or raises `DataLost`, a `RuntimeError` ([A lost host](#a-lost-host)).
- After step 8, `summary.json` records each error as `error`, and `complete` stays false.
  The exception ends the process with a non-zero status.
- The evaluation has no time limit of its own. It waits for each job until the Run Store ends it.
  A job of a worker that died runs again on another worker ([Run Store](RUNSTORE.md#dead-workers)).

An attempt is resolved when its verifier's reward is 1. A grade in a lost kept sandbox scores
0 with the reason `sandbox_lost`.

## Options of an evaluation

The evaluation has no options for the runtime binary, the sandbox backend, the store, the
sandbox server, its API key or the sandbox TTL. The worker profiles decide them. It has no
option for a final memory snapshot. A verifier that needs the agent's processes grades in
the kept sandbox.

The options of the task and the output:

| Option | Default | What it does |
|---|---|---|
| `--task-dir` | required | the Harbor task directory |
| `--dataset-digest` | required | must equal the `dataset_digest` of the manifest |
| `--preparation-manifest` | required | the JSON file that pins the task |
| `--output` | required | a new or empty directory for the output |
| `--method` | `sprout` | `baseline`, `sprout`, `bpo` or `shepherd` |
| `--parent-output` | none | the output of a completed baseline evaluation, whose parent the methods reuse. `bpo` and `shepherd` require it. |
| `--volatile-ok` | off | lets `--output` be under `/tmp`, `/var/tmp`, `/dev/shm` or `/run` |

The options of the Run Store:

| Option | Default | What it does |
|---|---|---|
| `--runstore-url` | `http://127.0.0.1:18110` | the Run Store's HTTP API |
| `--runstore-token-env` | `SPROUT_RUNSTORE_TOKEN` | the variable of this process that holds the bearer token |
| `--profile` | `mini` | the worker profile of each rollout |
| `--grade-profile` | `grade` | the worker profile of each grade |
| `--review-profile` | `review` | the worker profile of each review |
| `--poll-interval` | 5 | the seconds between two passes over the groups |

The options of the actor, the model that each rollout runs:

| Option | Default | What it does |
|---|---|---|
| `--model`, `--model-endpoint` | required | the model and its Chat Completions endpoint |
| `--model-key-env` | required | the worker's variable that holds the endpoint's key. For `bpo` and `shepherd`, the variable must also be set in this process. |
| `--checkpoint-mode` | `disk_only` | `disk_only` takes a disk snapshot after each turn, which makes the branch points. `none` takes none. |
| `--max-output-tokens` | 64000 | `max_tokens` of each model request |
| `--max-turns` | 300 | the turn limit of a rollout |
| `--command-timeout` | 600 | the seconds that the runtime gives each agent command before it kills the command. A branch keeps the value of its parent. |
| `--reasoning-effort` | none | `reasoning_effort` of each model request |
| `--no-image-env` | off | does not set the image's `ENV` again on the commands that run as its `USER` |
| `--no-image-user` | off | runs the agent's commands, `solve.sh` and `test.sh` as root |

The options of the review, for `sprout`. The review job gets them as the `ReviewSpec` fields
in parentheses:

| Option | Default | What it does |
|---|---|---|
| `--rounds` | 2 | the most rounds of review and branches |
| `--branches` | `4 3` | the width of each round. The last value applies to each later round. |
| `--branch-count-mode` (`count_mode`) | `adaptive` | `adaptive`: the width is a maximum. `fixed`: the width is exact. |
| `--branch-guidance` (`guidance`) | `assistant-turn` | `assistant-turn`, `user-hint` or `none` |
| `--prompts` (`prompts`) | `resources/review/prompts.yaml` | the YAML file of the analyst, reviewer and Shepherd prompts. The review jobs read it on the worker's host. Its sha256 is in the spec. |
| `--structured-review` (`structured`) | off | asks for schema-constrained JSON instead of plain-text plans. The endpoint must accept `json_schema`. |
| `--review-model`, `--review-endpoint` | `--model`, `--model-endpoint` | the model of the analyst and the reviewer, and its endpoint |
| `--review-key-env` | `--model-key-env` | the worker's variable that holds the key of `--review-endpoint` |
| `--review-effort` | `--reasoning-effort` | `reasoning_effort` of the review's requests |
| `--review-output-tokens` | 64000 | `max_tokens` of each review request |
| `--analyst-tokens` | 100000 | the transcript budget of each analyst request, in tokens |
| `--prompt-budget-chars` | 400000 | the size of each analyst and reviewer request, in characters. The least value is 50000. |
| `--reviewer-max-attempts` | 3 | the most replies that the analyst or the reviewer can give before one is valid |
| `--review-timeout` (`request_timeout_s`) | 3600 | the seconds of one analyst or reviewer request. The review job's own limit is 2 × (⌈attempts ÷ 8⌉ + 2) × this value. |
| `--leak-judge` | off | checks each turn that the reviewer writes for hidden-test content. It needs `--method sprout` with `assistant-turn` guidance. |
| `--leak-judge-resamples` | 2 | how many times a flagged turn goes back to the reviewer before the review takes it, marked |
| `--leak-judge-threshold` | 0.75 | the score at or above which the judge flags a turn, in (0, 1] |
| `--leak-judge-model` | `jev-1.13.0` | the judge's model |
| `--leak-judge-key-env` | `TYPESAFE_API_KEY` | the worker's variable that holds the judge's key |

The options of the selectors, for `bpo` and `shepherd`:

| Option | Default | What it does |
|---|---|---|
| `--max-rollouts` | 8 | the total rollouts, the shared parent included. The range is 1 to 8 for `bpo` and `shepherd`, and 1 to 128 otherwise. `sprout` only records it. |
| `--bpo-top-k` | 5 | `top_logprobs` of each scoring request, from 1 to 20 |
| `--bpo-min-spacing` | 64 | the least distance between two selected checkpoints, in completion tokens |
| `--bpo-max-points` | 7 | the most checkpoints that `bpo` selects |
| `--bpo-scoring-workers` | 2 | the scoring requests that run at the same time |
| `--meta-max-tokens` | 64000 | `max_tokens` of the Shepherd request |
| `--meta-transcript-tokens` | 60000 | the transcript budget of the Shepherd request, in tokens |

The evaluation exits with status 2 before any job exists for each of these values:

- `--rounds` below 0, an empty `--branches`, or a width below 1
- `--reviewer-max-attempts` below 1
- `--max-rollouts` or `--bpo-top-k` out of its range
- `--prompt-budget-chars` below 50000
- a `--prompts` file that the evaluation cannot read, or that `load_prompts` refuses
- a `--model-key-env`, `--review-key-env`, `--leak-judge-key-env` or `--runstore-token-env`
  value that is not one variable name
- `bpo` or `shepherd` without `--parent-output`
- a `--command-timeout`, `--review-timeout` or `--poll-interval` that is not positive
- `--leak-judge` without `sprout` and `assistant-turn`, or a resample count or a threshold
  out of its range

## The output of an evaluation

The output directory holds these files:

| File | What it holds |
|---|---|
| `summary.json` | the settings, the `affinity` key, the `attempts` rows, `earlier_runs`, `complete`, `resolved`, `branch_schedule_complete`, and `stop_reason` or `error`. After `DataLost`, also `reason: "data_lost"`. |
| `groups/<group>.json` | the coordinator's document of each group of jobs: `parent`, `review-<N>`, `round-<N>`, or one `b<NN>-step<S>` for each `bpo` or `shepherd` branch |
| `earlier-runs/<key>/` | the files of each run that a lost host ended: its `groups/`, its plans and `hidden-tests.patch` ([A lost host](#a-lost-host)) |
| `plan-round<N>.json` | `sprout`: the round's point review as the review job recorded it, each selected branch with what it is given, the `branches` made, the `skipped` points and the `review_job` |
| `plan.json` | `bpo` and `shepherd`: the selected branches and the eligible steps |
| `scores.json`, `entropy/` | `bpo`: the score of each checkpoint, and each scoring request with its tries and its response |
| `meta-request.json`, `proposal.json` | `shepherd`: the meta-agent request with its tries and its response, and the proposal from it |
| `hidden-tests.patch` | `sprout` with `--leak-judge`: the task's hidden tests, which the Jev judge reads |

Each row of `attempts` describes one attempt:

| Field | What it holds |
|---|---|
| `name`, `round` | `parent` in round 0, or a branch in its round. A `bpo` or `shepherd` branch is in round 1. |
| `status`, `stop_reason` | the agent's own status, such as `completed` or `timeout`, and why a run stopped early |
| `resolved`, `reward`, `reason` | the verifier's verdict. `reason` is null when the verifier wrote the reward. |
| `journal`, `log`, `model_patch` | paths on the worker's host: the journal, the verifier's `test-stdout.txt`, and the `model.patch` in the artifacts |
| `job_id`, `attempt_id`, `grade_job_id` | the jobs in the Run Store |
| `hint` | the hint of a `user-hint` branch, only when it has one |
| `started_at`, `validated_at`, `seconds` | when the coordinator first saw the rollout out of the queue, when the evaluation made the row, and the seconds between them |

Each entry of `earlier_runs` describes one run that a lost host ended, oldest first:

| Field | What it holds |
|---|---|
| `affinity` | the key of the run. Its files are in `earlier-runs/<affinity>/`. |
| `attempts` | the rows of the run's attempts before the loss |
| `error` | the group that lost a job, and the key of the lost host |

## Gate a task set

```bash
python -m sprout.eval.gate --tasks-dir ~/datasets/deep-swe/tasks --output runs/gate --workers 8
python -m sprout.eval.gate ... --task abs-module-cache-flags --mode oracle
```

The gate checks that the task's oracle scores 1 and that doing nothing scores 0. For each
task, the gate runs these two checks, one after the other:

1. an `oracle` rollout, which runs the task's `solution/solve.sh` as Harbor's oracle agent
   does, and its grade, which must score 1
2. a `nop` rollout, which does nothing, and its grade, which must score 0

The `nop` check starts after the `oracle` check settles, whatever its result. In both
checks, `solve.sh` and `test.sh` run as the image's `USER` with its `ENV`, as in Harbor. The
flags `--no-image-user` and `--no-image-env` turn these settings off. The rollouts of the
gate take no per-turn snapshots.

The gate does these steps:

1. It selects the tasks: each `--task`, then the names in the JSON list of `--selection`. If
   you give neither, it selects each directory in `--tasks-dir` that has a `task.toml`, in
   name order.
2. It selects the modes: `--mode both` (default) runs `oracle`, then `nop`.
3. It reads `gate.jsonl` in `--output` and skips each check that a record marks `ok`.
4. It runs the checks of `--workers` tasks at a time (default 8).
5. When a check settles, it appends the check's record to `gate.jsonl` and prints one line.
6. It writes `summary.json` and exits.

`--workers` is 8 by default, so that a grade in a kept sandbox reaches its worker before the
sandbox's 20 minutes pass.

A check passes only when each of these conditions is true:

- The rollout succeeded.
- The grade succeeded.
- The verifier wrote the reward, so `reason` is null.
- The reward is the expected one.

The gate records a check as failed at once, with the error, in these cases:

- The reader or `check_environment` refuses the task.
- The gate cannot read a file of the task.

Then the gate goes on to the task's next mode.

Each record in `gate.jsonl` has these fields:

| Field | What it holds |
|---|---|
| `task`, `mode`, `expected` | the task directory's name, `oracle` or `nop`, and 1 or 0 |
| `affinity` | the key of the check's jobs, when the check was submitted |
| `rollout_job`, `grade_job` | the jobs in the Run Store, when the check was submitted |
| `reward`, `reason` | the verdict, when the rollout and the grade succeeded. `reason` is `data_lost` for a check whose host was lost after 2 runs again. |
| `ok` | whether the check passed |
| `error` | why the check failed, when it did not reach a verdict or the verifier wrote no reward |

`summary.json` has these fields:

| Field | What it holds |
|---|---|
| `tasks`, `modes` | the number of selected tasks, and the modes |
| `ok` | each check that passed, as `<task>/<mode>`, from this run or an earlier one |
| `failed_tasks` | each selected task with a check that has not passed |
| `failures` | the record of each check that failed in this run |

The gate exits with status 1 when a check of this run failed, and with 0 otherwise. A rerun
into the same output skips the checks that passed. It runs the others again as new jobs, and
it never reads an old job. Each run keeps its group documents in `groups/<run id>/`. A check
that runs again after a lost host has the group `<task>-<mode>-r<n>` ([A lost host](#a-lost-host)).
If the token variable is empty, the gate raises `ValueError`. If `--workers` is below 1 or
`--poll-interval` is not positive, the gate exits with status 2.

## Validate without a worker

```bash
SPROUT_RUNSTORE_TEST_DSN="dbname=sprout_runstore_test" python3 -m pytest tests/eval -q
```

These tests run the real coordinator against a real Run Store, in a schema of its own on the
PostgreSQL test database that `SPROUT_RUNSTORE_TEST_DSN` names. A fake worker
(`tests/eval/worker.py`) settles the jobs. Without the variable, the tests that need the Run
Store skip. `tests/sprout_sandbox/test_harbor_grade_live.py` (marked `slow`) checks that the
worker's `harbor` grade gives Harbor's own rewards on the fixture task
([Harbor plugin](HARBOR_PLUGIN.md)).
