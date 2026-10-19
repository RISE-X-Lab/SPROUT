# Evaluation and the task gate

`python -m sprout.eval.evaluate` evaluates one Harbor task with one method.
`python -m sprout.eval.gate` checks that the verifiers of a task set give the expected
rewards. Both commands are clients of the [Run Store](RUNSTORE.md), like the
[RL driver](RL_DRIVER.md). Each rollout, grade, review and branch is a Run Store job, and a
worker runs it ([ADR-0002](decisions/0002-eval-and-gate-through-run-store.md)). Neither
command starts a sandbox or runs the agent itself.

Read this document to evaluate a task or to change the evaluation code. Before you change a
method, read [The jobs of each method](#the-jobs-of-each-method).

## An evaluation is a client of the Run Store

The client builds the jobs, the search coordinator submits and follows them, and the worker
runs them. Run the client on the worker's host, because the client reads the files that the
jobs leave in the worker's artifact directory.

<p align="center">
  <img src="figures/evaluation-components.svg" width="900" alt="The worker's host holds the client process, the Run Store, the worker, the sandboxes, the artifact directory and the output directory. In the client process, Evaluation or Gate builds group requests and hands them to Groups and the search Coordinator, which submit the jobs to the Run Store API over HTTP, poll them and stop them. The Run Store API keeps the jobs in PostgreSQL. The worker claims the jobs in PostgreSQL and finishes them there. The worker starts the sandboxes, on podman or a microVM, and runs the agent and the verifier in them. The worker sends the model requests of rollouts and reviews to the model endpoint, outside the host. The worker writes journals, verifier logs and review plans into its artifact directory, and the client process reads them there. The client process writes summary.json, the group documents in groups/ and the plans into the output directory. For bpo and shepherd, the selector in the client process sends its own requests to the model endpoint."/>
</p>
<p align="center"><sub>The client submits jobs to the Run Store over HTTP, the worker runs them, and the client reads the worker's files on the same host.</sub></p>

| Component | Code | What it does |
|---|---|---|
| client | `Evaluation` in `evaluate.py`, `Gate` in `gate.py` | builds the group requests from the task (`jobs.py`), and writes the output |
| coordinator | `Groups` in `groups.py`, `Coordinator` in `sprout.search` | submits each job, polls it, and keeps one group document for each group in `groups/` |
| Run Store | `python -m sprout.runstore serve` | keeps the jobs, their attempts and their recovery points in PostgreSQL |
| worker | `python -m sprout.worker start` | claims each job, runs it, and writes its files into the artifact directory |

A group is one request to the coordinator, with one or more samples. A sample is a run, a
branch or a review, and a run or a branch can have a grade. The coordinator follows these rules:

- It submits each job under an idempotency key that starts with `eval/<evaluation id>/` or
  `gate/<run id>/`, then a hash of the affinity key, the sample id and the stage. A second
  submission of a stage makes no second job.
- It submits a grade only after its rollout or branch succeeded. If the rollout ends in another
  final state, it marks the grade `skipped`.
- A group is ready when each of its stages is final (`DONE` in `src/sprout/search/stages.py`),
  or when a job of it ended `failed` with the reason `data_lost`.

### Affinity keys

All jobs of one task's search carry one affinity key, so that they run on the host that holds
the search's data ([ADR-0007](decisions/0007-search-affinity-and-host-loss.md)). A review reads
the attempt directories of its parents. A grade reads the kept sandbox or the final snapshot of
its rollout. The coordinator gives each run, grade and review job the key of its group. A branch
takes the key of the job that it continues ([Run Store](RUNSTORE.md#affinity-keys)).

| Client | Key | Recorded in |
|---|---|---|
| evaluation | `eval-<32 hex digits>`, one for each evaluation | `affinity` in `summary.json` |
| evaluation with `--parent-output` | the `affinity` of the recorded evaluation, because each branch carries the key of its parent | `affinity` in `summary.json` |
| gate | `gate-<32 hex digits>`, one for each task of each gate run | `affinity` in each record of `gate.jsonl` |

All groups of one evaluation share its key, so each sample id is unique in the evaluation. For
example, the review of round `<N>` is the sample `review-<N>`.

### A lost host

When the host of a key is lost, the Run Store fails each unfinished job of the key with the
reason `data_lost` ([Run Store](RUNSTORE.md#lost-hosts)). The later jobs of the search cannot
read the lost data, so the coordinator of these two clients does not run the group again. It
sets the group's `error`, and the group is ready and `failed` at once. Then the client runs the
search again under the key `<first key>-r<n>`, at most 2 times (`RUNS_AGAIN` in
`src/sprout/search/rerun.py`). The figure shows what each client does.

<p align="center">
  <img src="figures/evaluation-lost-host.svg" width="900" alt="Two lanes. In the evaluation lane, the evaluation runs the search: the parent, then the method. When a job ends data_lost, the evaluation asks whether it uses --parent-output. If yes, it raises DataLost. If no, it asks whether it ran again 2 times. If yes, it raises DataLost. If no, it keeps the run in earlier-runs/&lt;key&gt;/, and a back edge leads to the search again under the key &lt;first key&gt;-r&lt;n&gt;. In the gate lane, the gate runs a check under the task's latest key. When a job ends data_lost, the gate asks whether the task's checks ran again 2 times. If yes, it records the check as failed with the reason data_lost, then goes on to the task's next mode. If no, it runs the check again first, in the group &lt;task&gt;-&lt;mode&gt;-r&lt;n&gt;, and a back edge leads to the check again under the key &lt;first key&gt;-r&lt;n&gt;. Blue boxes submit jobs, yellow boxes are questions, red boxes end failed, and the gray box keeps files."/>
</p>
<p align="center"><sub>An evaluation runs its parent and its method again, and the gate runs only the check that lost its job, each under a new key.</sub></p>

| Client | What runs again | When it cannot run again |
|---|---|---|
| evaluation | the parent, then the method | After 2 runs again, `Evaluation` raises `DataLost`, a `RuntimeError`. |
| evaluation with `--parent-output` | nothing, because the parent's data was on the lost host | `Evaluation` raises `DataLost` at the first loss. |
| gate | only the check that lost its job, before the task's other modes. A check reads no data of another check. | After 2 runs again, `Gate` records the check as failed, with `reason: "data_lost"`. |

Before an evaluation runs again, it moves each entry of `--output` except `summary.json` and
`earlier-runs/` into `earlier-runs/<key>/`, and appends the run to `earlier_runs`. It deletes no
file and stops no job. An attempt of an earlier run does not count for `resolved`. After
`DataLost`, `summary.json` also records `reason: "data_lost"`.

## Before you start

You need a Run Store API and a worker on the client's host.
`python -m sprout.services up runstore-api runstore-worker` starts both
([services](SERVICES.md)). The worker profiles decide the sandbox backend, the runtime binary
and the store, so the evaluation and the gate have no options for them. The worker's
environment must hold the keys that `--model-key-env`, `--review-key-env` and
`--leak-judge-key-env` name. The worker must serve these profiles:

- `--profile` (default `mini`) for each rollout
- `--grade-profile` (default `grade`) for each grade
- `--review-profile` (default `review`) for each review of `--method sprout`

Before it submits a job, the client checks these conditions, in this order:

| Condition | Client | Otherwise |
|---|---|---|
| The option values are valid. For an evaluation, `--output` is not under `/tmp`, `/var/tmp`, `/dev/shm` or `/run`, which the OS empties, unless you pass `--volatile-ok`. | both | exit status 2 |
| In the client's environment, the variable of `--runstore-token-env` (default `SPROUT_RUNSTORE_TOKEN`) holds the Run Store token. | both | `ValueError` |
| For `bpo` and `shepherd`, the client's environment holds the key of `--model-key-env`, because the selector sends its own model requests. | evaluation | `ValueError` |
| `load_task` accepts the task. | evaluation | `ValueError` |
| The git worktree of the source is clean, because the summary records its commit as `source_commit`. | evaluation | `RuntimeError` |
| `--output` is new or empty. | evaluation | `FileExistsError` |
| `check_environment` accepts the task ([How a Harbor task becomes jobs](#how-a-harbor-task-becomes-jobs)). | evaluation | `ValueError` |
| The preparation manifest pins the task. | evaluation | `ValueError` |
| With `--parent-output`, the recorded `summary.json` names an affinity key. | evaluation | `ValueError` |

The preparation manifest pins the task when each of these conditions is true.
`examples/qwen/run.py` writes such a manifest.

- Its `dataset_digest` equals `--dataset-digest`.
- Exactly one row of its `tasks` has the task's `path` and `name`.
- The `task_toml_sha256` of that row equals the sha256 of the task's `task.toml`.

## How a Harbor task becomes jobs

Each attempt is a rollout job and a grade job. `sprout.eval.jobs` builds both from the task, as
`sprout.contract.harbor_task.load_task` reads it
([ADR-0003](decisions/0003-sprout-runs-harbor-verifier.md)). Before any job exists,
`check_environment` raises `ValueError` for each task whose image is not its whole environment:

- a task without a `docker_image`
- a task without `environment/Dockerfile` whose `environment/` holds files. Harbor uploads these
  files into the sandbox, and a rollout does not.
- a task whose `environment/Dockerfile` is `FROM` the task's own image, and adds `COPY` or
  `RUN` steps or a step that `parse_overlay` refuses

A `Dockerfile` `FROM` another image is the recipe of the prebuilt image, and the client
accepts it. The rollout job takes these values:

| Field | Value |
|---|---|
| `agent` | `mini` in an evaluation, `oracle` or `nop` in the gate |
| `prompt`, `harbor_task` | the task's `instruction.md` without its leading canary lines, and the task directory. The worker starts the sandbox on the task's agent network policy. |
| `sandbox_image`, `sandbox_resources` | the task's `docker_image`, and its CPUs and memory when the task sets them |
| `timeout_s` | the task's agent timeout, or 3600 s when the task sets none |
| `checkpoint_enabled` | in an evaluation, true unless `--checkpoint-mode none`. In the gate, false. |
| `keep_sandbox` | true when the verifier shares the agent's environment |
| `max_infra_retries` | 0, because one attempt is one execution |
| `extra` | `mini` only: the rollout contract (model, endpoint, key variable, sampling, `max_turns`, `deadline_at`), the command timeout, and the task's `workdir` as `cwd` |

The `deadline_at` of a `mini` rollout, and of each branch, is 2 days after the agent timeout,
counted from when the evaluation makes the job. If it has passed when a worker takes the
rollout, the worker does not start the run ([Run Store](RUNSTORE.md#timeouts)). A wait that long
means that no worker serves the profile.

The grade job is a `harbor` grade of the task's own verifier
([Run Store](RUNSTORE.md#the-harbor-grader)). Its timeout is the verifier timeout plus the
timeouts of the task's `[[verifier.collect]]` commands. Its source depends on the task:

| Condition | Source of the grade | Retries after an infrastructure failure |
|---|---|---|
| The verifier shares the agent's environment. | the [kept sandbox](RUNSTORE.md#kept-sandboxes) of the rollout: `in_place: true`, which the coordinator turns into `in_place_job` ([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)) | none, because the first attempt destroys the kept sandbox |
| The verifier has an environment of its own. | the rollout's final snapshot, `snapshot_id` | up to 2 |

On a microVM profile, a grade in the kept sandbox fails, because a kept microVM sandbox cannot
switch its egress to the verifier's network policy.

Both jobs set `image_env` and `image_user` in each section of their `backend`, unless you pass
`--no-image-env` or `--no-image-user`. Then the agent's commands, `solve.sh` and `test.sh` run
as the image's `USER` with its `ENV`, as in Harbor ([Run Store](RUNSTORE.md#image-user-and-env),
[ADR-0005](decisions/0005-eval-runs-as-the-image-user.md)).

## The jobs of each method

The methods differ only in how they choose the branch points. Each method starts from one
parent attempt, which is a `mini` rollout and its grade. The figure shows the jobs and the
groups of each method.

<p align="center">
  <img src="figures/evaluation-method-jobs.svg" width="900" alt="Three lanes, one for each kind of method. In the baseline lane, the group parent holds a rollout job of the agent mini and, after the rollout succeeded, a grade job with the benchmark harbor. Then the evaluation ends. With --parent-output, the evaluation reuses the parent of that evaluation and runs no job. In the sprout lane, the group parent holds the same rollout and grade, or the evaluation reuses them with --parent-output. After the parent's grade succeeded, the group review-N holds one review job, which reads each attempt so far and picks k points. Each point becomes a branch job in the group round-N, and each branch job that succeeded gets a grade job. A back edge goes from the grades of round N to the review of the next round, while no attempt resolved the task and N is less than --rounds. In the bpo and shepherd lane, the parent is the parent of a baseline evaluation, not a new job. A selector in the client process, not a job, reads the parent's journal and picks the points. Each point becomes its own group b-NN-step-S with one branch job and, after the branch succeeded, its grade job. A back edge goes from the grade to the next branch, while no attempt resolved the task and branches are left. Blue boxes are rollout jobs, a branch included. Green boxes are grade jobs. The yellow box is a review job. Gray boxes are not jobs."/>
</p>
<p align="center"><sub>Each method runs or reuses one parent attempt, then sprout reviews and branches in rounds, and bpo and shepherd branch from a baseline's parent one point at a time.</sub></p>

| Method | Parent | Branch points | The method ends when |
|---|---|---|---|
| `baseline` | runs it, or reuses it with `--parent-output` | none | the parent's grade settles |
| `sprout` | runs it, or reuses it | A review job in each round reads each attempt so far and picks the points. | an attempt resolves the task, `--rounds` rounds have run, or a round makes no branch |
| `bpo` | reuses it, so `--parent-output` is required | The selector ranks the parent's checkpoints by the policy's first-token entropy. | a branch resolves the task, or each selected branch has run |
| `shepherd` | the same as `bpo` | One meta-agent request picks one checkpoint step for all branches. | the same as `bpo` |

A branch keeps the spec of the job that it continues, with its agent, its image settings and its
command timeout ([Run Store](RUNSTORE.md#branch)). The grade of a branch takes the image flags of
the evaluation that makes it.

### sprout rounds

For each round `<N>`, from 1 to `--rounds`, the evaluation does these steps:

1. If an attempt has resolved the task, the evaluation stops.
2. It runs one review job in the group `review-<N>`. The job reads each attempt row so far, and
   picks up to the round's width in points, or exactly the width with
   `--branch-count-mode fixed`. The width is the `<N>`-th value of `--branches`, or the last one.
3. It finds the available recovery point of each point's attempt at the point's tool step, and
   writes `plan-round<N>.json`. If no point makes a branch, it sets `stop_reason` and
   `branch_schedule_complete: false`, and stops.
4. It submits all branches of the round at once, in the group `round-<N>`. Each branch has the
   name of its point's `run_name`. The worker's concurrency decides how many run at a time.
5. After all branches and their grades settle, it adds a row for each branch and saves
   `summary.json`.

`--branch-guidance` decides what each branch gets, and when a point makes no branch. The plan
lists each point that makes no branch under `skipped`, with the reason:

| `--branch-guidance` | The branch gets | A point makes no branch when |
|---|---|---|
| `assistant-turn` (default) | the turn that the review wrote at the point | the review wrote no turn there, or the attempt has no available recovery point at the step |
| `user-hint` | the review's hint for the point | the attempt has no available recovery point at the step |
| `none` | nothing. The branch samples again from the recovery point. | the same as `user-hint` |

### bpo and shepherd branches

The evaluation does these steps:

1. The selector reads the parent's journal on this host and finds the parent's exact `mini`
   checkpoints. If there is none, the evaluation raises `RuntimeError`.
2. The selector picks `--max-rollouts` − 1 branches, and the evaluation writes `plan.json`:
   - `bpo` asks the endpoint for the first-token log probabilities at each checkpoint that a
     later model request follows. It takes up to `--bpo-max-points` checkpoints, highest
     entropy first, at least `--bpo-min-spacing` completion tokens apart. It deals the branches
     over them in turn.
   - `shepherd` sends one meta-agent request with the parent's transcript. The answer names one
     eligible checkpoint step and a reason. All branches start at that step.
3. For each selected branch in turn, the evaluation runs the branch without guidance, and its
   grade, in a group of its own, `b<NN>-step<S>`. It adds the branch's row in round 1 and saves
   `summary.json`. If the branch resolved the task, the evaluation stops.

The selector sends each model request in up to 8 tries, with a wait of up to 30 s between two
tries. If the selector makes no valid selection, it raises `ValueError`. If the parent has no
available recovery point at a selected step, the evaluation raises `RuntimeError`.

## Evaluate one task

```bash
python -m sprout.eval.evaluate --task-dir ~/datasets/deep-swe/tasks/<task> \
    --dataset-digest sha256:... --preparation-manifest runs/prep.json \
    --output runs/<task>-sprout --method sprout \
    --model <model> --model-endpoint <url> --model-key-env OPENAI_API_KEY --rounds 2 --branches 4 3
```

The evaluation does these steps:

1. It checks the conditions of [Before you start](#before-you-start), and writes the first
   `summary.json`.
2. It runs the parent in the group `parent`: one `mini` rollout and its grade. With
   `--parent-output`, it reuses the parent of that evaluation instead.
3. For a method other than `baseline`, it checks that the parent is a `mini-swe-agent` run. If
   not, it raises `ValueError`.
4. It runs the method. If a job loses its host, it runs steps 2 to 4 again
   ([A lost host](#a-lost-host)). It saves `summary.json` after each group of attempts settles.
5. It checks that the task's `task.toml` did not change. If it changed, it raises
   `RuntimeError`.
6. It writes `summary.json` with `complete: true`, prints the summary without the `attempts`,
   and exits with status 0.

The evaluation reuses the parent of `--parent-output` only when each of these conditions is
true. Otherwise it raises `ValueError`.

- The recorded evaluation is complete, and ran the same task of the same dataset digest.
- Its parent row has a `job_id`. A parent of a local Harbor trial has none.
- It used the same `--runstore-url`.
- The parent's journal is still on this host.

These are the paths other than the normal path:

- If a rollout does not succeed, or its grade gives no numeric reward, the evaluation raises
  `RuntimeError`, which names the job. A review job that does not succeed does the same.
- After step 1, the evaluation records each error as `error` in `summary.json`, and `complete`
  stays false. The exception ends the process with a non-zero status.
- The evaluation has no time limit of its own. It waits for each job until the Run Store ends
  it. A job whose worker died runs again on a worker of the same host
  ([Run Store](RUNSTORE.md#dead-workers)).

An attempt is resolved when its verifier's reward is 1.

## Options of an evaluation

`python -m sprout.eval.evaluate --help` lists each option and its default. These options change
what the evaluation runs:

| Option | Default | What it does |
|---|---|---|
| `--method` | `sprout` | `baseline`, `sprout`, `bpo` or `shepherd` |
| `--parent-output` | none | the output of a completed baseline evaluation, whose parent the method reuses. `bpo` and `shepherd` require it. |
| `--checkpoint-mode` | `disk_only` | `disk_only` takes a disk snapshot after each turn, which makes the branch points. `none` takes none. |
| `--command-timeout` | 600 | the seconds that the runtime gives each agent command. A branch keeps the value of its parent. |
| `--rounds`, `--branches`, `--branch-count-mode` | 2, `4 3`, `adaptive` | `sprout`: the most rounds, the width of each round, and whether the width is a maximum (`adaptive`) or exact (`fixed`) |
| `--branch-guidance` | `assistant-turn` | `sprout`: what each branch gets ([sprout rounds](#sprout-rounds)) |
| `--review-model`, `--review-endpoint`, `--review-key-env`, `--review-effort` | the actor's value | `sprout`: the model of the analyst and the reviewer, its endpoint, the worker's variable of its key, and its `reasoning_effort` |
| `--prompts` | `resources/review/prompts.yaml` | the prompts of the analyst, the reviewer and Shepherd. The review jobs read the file on the worker's host, and the spec has its sha256. |
| `--review-timeout` | 3600 | the seconds of one analyst or reviewer request. The review job's own limit is 2 × (⌈attempts ÷ 8⌉ + 2) × this value. |
| `--leak-judge` | off | checks each turn that the reviewer writes for hidden-test content ([Run Store](RUNSTORE.md#leak-checks)). It needs `sprout` with `assistant-turn`. |
| `--max-rollouts` | 8 | `bpo` and `shepherd`: the total rollouts, the parent included, at most 8. `sprout` only records it. |

The `--bpo-*` and `--meta-*` options tune the two selectors. The evaluation exits with status 2
for a value out of its range, for a `--prompts` file that `load_prompts` refuses, and for a
`*-key-env` or `--runstore-token-env` value that is not one variable name.

## The output of an evaluation

The output directory holds these files:

| File | What it holds |
|---|---|
| `summary.json` | the settings, the `affinity` key, the `attempts` rows, `complete`, `resolved`, `branch_schedule_complete`, and `stop_reason` or `error`. Each entry of `earlier_runs` has the `affinity`, the `attempts` and the `error` of a run that a lost host ended. |
| `groups/<group>.json` | the coordinator's document of each group: `parent`, `review-<N>`, `round-<N>`, or `b<NN>-step<S>` |
| `earlier-runs/<key>/` | the files of each run that a lost host ended ([A lost host](#a-lost-host)) |
| `plan-round<N>.json` | `sprout`: the round's point review from the review job, each selected branch with what it gets, the `branches` made, the `skipped` points and the `review_job` |
| `plan.json` | `bpo` and `shepherd`: the selected branches and the eligible steps |
| `scores.json`, `entropy/` | `bpo`: the score of each checkpoint, and each scoring request with its tries |
| `meta-request.json`, `proposal.json` | `shepherd`: the meta-agent request with its tries, and the proposal from it |
| `hidden-tests.patch` | `sprout` with `--leak-judge`: the task's hidden tests, which the Jev judge reads |

Each row of `attempts` describes one attempt:

| Field | What it holds |
|---|---|
| `name`, `round`, `hint` | `parent` in round 0, or a branch in its round. A `bpo` or `shepherd` branch is in round 1. A `user-hint` branch has its `hint`. |
| `status`, `stop_reason` | the agent's own status, such as `completed` or `timeout`, and why a run stopped early |
| `resolved`, `reward`, `reason` | the verifier's verdict. `reason` is null when the verifier wrote the reward. |
| `journal`, `log`, `model_patch` | paths on the worker's host: the journal, the verifier's `test-stdout.txt`, and the `model.patch` in the artifacts |
| `job_id`, `attempt_id`, `grade_job_id` | the jobs in the Run Store |
| `started_at`, `validated_at`, `seconds` | when the coordinator first saw the rollout out of the queue, when the evaluation made the row, and the seconds between them |

## Gate a task set

```bash
python -m sprout.eval.gate --tasks-dir ~/datasets/deep-swe/tasks --output runs/gate --workers 8
python -m sprout.eval.gate ... --task abs-module-cache-flags --mode oracle
```

The gate checks that the task's oracle scores 1 and that doing nothing scores 0. Each check is a
rollout without per-turn snapshots, and its `harbor` grade. The `oracle` rollout runs the task's
`solution/solve.sh`, as Harbor's oracle agent does. The `nop` rollout does nothing.

The gate does these steps:

1. It selects the tasks: each `--task`, then the names in the JSON list of `--selection`. If
   you give neither, it selects each directory in `--tasks-dir` that has a `task.toml`, in name
   order.
2. It selects the modes: `--mode both` (default) runs `oracle`, then `nop`.
3. It skips each check that a record in `gate.jsonl` of `--output` marks `ok`.
4. It runs the checks of `--workers` tasks at a time (default 8). The `nop` check of a task
   starts after its `oracle` check settles, whatever the result.
5. When a check settles, it appends the check's record to `gate.jsonl` and prints one line.
6. It writes `summary.json`. It exits with status 1 if a check of this run failed, and with 0
   otherwise.

A check passes only when the rollout and the grade succeeded, and the verifier wrote the
expected reward, so that `reason` is null. If the reader or `check_environment` refuses the
task, or the gate cannot read a file of the task, the gate records the check as failed at once.
Then it goes on to the task's next mode.

| File | What it holds |
|---|---|
| `gate.jsonl` | one record for each check: `task`, `mode`, `expected` and `ok`. `affinity`, `rollout_job` and `grade_job` when the check was submitted. `reward` and `reason` when it reached a verdict. `error` when it reached no verdict or the verifier wrote no reward. |
| `summary.json` | `tasks`, the number of selected tasks, and `modes`. `ok`, each passed check as `<task>/<mode>`, from this run or an earlier one. `failed_tasks`, each task with a check that has not passed. `failures`, the records of the checks that failed in this run. |
| `groups/<run id>/` | the group documents of each gate run |

A rerun into the same output skips the checks that passed. It runs the others again as new
jobs, and never reads an old job.

## Validate without a worker

```bash
SPROUT_RUNSTORE_TEST_DSN="dbname=sprout_runstore_test" python3 -m pytest tests/eval -q
```

These tests run the real coordinator against a real Run Store, in a schema of its own on the
PostgreSQL test database that `SPROUT_RUNSTORE_TEST_DSN` names. A fake worker
(`tests/eval/worker.py`) settles the jobs. Without the variable, the tests that need the Run
Store skip. `tests/sprout_sandbox/test_harbor_grade_live.py`, marked `slow`, checks that the
worker's `harbor` grade gives Harbor's own rewards on the fixture task
([Harbor plugin](HARBOR_PLUGIN.md)).
