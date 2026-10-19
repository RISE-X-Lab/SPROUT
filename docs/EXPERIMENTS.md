# The Qwen experiments

This guide runs the 12 Qwen experiments: three datasets, each with four methods. You run
the baseline of a dataset first. The three branch methods then continue the first rollout
of each task that the baseline failed. At the end, `python -m examples.qwen.compare`
exports one comparison of five rows and four metrics for each dataset. Read this guide to
set up a host, and to run, check, resume and export the experiments. The
[experiment protocol](../examples/qwen/PROTOCOL.md) defines the budget, the methods and
the metrics.

This guide uses these terms:

- A **rollout** is one run of the agent on one task, graded by the task's own verifier.
- The **parent** is the first baseline rollout of a task, in `tasks/<task>/baseline/attempt-1`.
- A **branch** is a rollout that continues another rollout from one of its checkpoints.
- A **stage** is one `sprout.eval.evaluate` process. It runs one baseline rollout, or one
  branch method on one task.
- A **root** is the `--output` directory of one run of `examples.qwen.run`.

Run all commands in Bash from the root of the Sprout checkout, unless a step says otherwise.

Each experiment is one dataset with one method. The column headings are the values of
`--method`, and each cell is the number of an experiment:

| Dataset | `--dataset` | `baseline` | `sprout` | `bpo` | `shepherd` |
|---|---|---|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | 1 | 2 | 3 | 4 |
| TerminalBench 2.1 | `terminalbench21` | 5 | 6 | 7 | 8 |
| DeepSWE | `deepswe` | 9 | 10 | 11 | 12 |

## 1. Install the runtime and dependencies

One host runs the whole experiment. The runner and the evaluator read the journals that the
worker writes into its artifact directory, so they must run on the worker's host. Figure 1
shows the processes on that host and what each one calls or writes.

<p align="center">
  <img src="figures/experiments-components.svg" width="1000" alt="One experiment host holds seven parts. The runner, examples.qwen.run, runs one thread per task and starts one sprout.eval.evaluate process, the evaluator, for each stage. The evaluator submits the jobs of its stage to runstore-api, the Run Store API, and polls them. runstore-worker, the worker, claims the jobs from the Run Store and starts the sandboxes, on podman with a btrfs store or on AgentENV microVMs, and takes their checkpoints. Two parts are files. The runner writes the manifest, the stage receipts and the metrics into the root that --output names, and the evaluator writes each stage's summary.json there. The worker writes the journals and the verifier logs into its artifact_root directory, and the evaluator reads them from there. Solid arrows mean that a process starts or calls another. Dotted arrows show data that a process writes into files or reads from them."/>
</p>
<p align="center"><sub>Figure 1. The runner starts one evaluator per stage, the evaluator submits jobs to the Run Store, and the worker runs them in sandboxes on the same host.</sub></p>

Each part does these things:

| Part | What it does |
|---|---|
| `examples.qwen.run`, the runner | Runs each selected task in a thread of its own, `--workers` tasks at a time (default 16). Starts one evaluator for each stage of a task. Writes the manifest, the stage receipts and the metrics into the root. |
| `sprout.eval.evaluate`, the evaluator | Submits the rollout, grade, review and branch jobs of one stage to the Run Store, and polls them. Writes `summary.json` into the stage directory. For `bpo` and `shepherd`, it also sends the selector requests to the model endpoint itself. |
| `runstore-api` | The Run Store API at `http://127.0.0.1:18110`. It keeps the jobs in the PostgreSQL service `db`. |
| `runstore-worker` | Claims the jobs, runs each one in a sandbox of its profile's backend, and sends the actor, analyst and reviewer requests to the model endpoint. Writes the journals and the verifier logs into its `artifact_root`. |
| sandboxes | Podman containers on a btrfs store, or AgentENV microVMs (Section 8). |

The packaged `services.yaml` starts the worker with `--concurrency 1`, so the worker runs one
job at a time. To run more jobs at the same time, raise `--concurrency` in the
`runstore-worker` command of `services.yaml`.

The host must have these prerequisites:

- Python 3.12 or later, and Go 1.25 or later
- one sandbox backend (Section 8):
  - rootless podman 6 on a btrfs store, the default backend. `ops/setup_podman.sh` installs
    this podman and sets up the host.
  - a working microVM snapshot service: the customized Firecracker stack (AgentENV), which
    you deploy separately
- network access to the registries of the task images
- a git clone of the complete Sprout repository. The runner imports other Sprout modules,
  and it records the commit of the checkout.

Install Sprout and start the services:

1. Clone the repository and install the package into a virtual environment:

   ```bash
   git clone https://github.com/dreamyang-liu/sprout.git sprout
   cd sprout
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[mini,runstore]"
   export LITELLM_LOCAL_MODEL_COST_MAP=True
   ```

2. Build the runtime binary. `go generate` fetches the ripgrep that the runtime embeds.

   ```bash
   (cd runtime && go generate ./... && go build -o sprout-runtime .)
   ```

3. Write the service configuration:

   ```bash
   python -m sprout.services init
   ```

   The command writes `services.yaml`, `runstore.json` and the other service files into
   `~/.config/sprout`, and prints what to edit. It does not overwrite a file that exists
   ([services](SERVICES.md)).
4. Edit the profiles in `runstore.json` as `init` prints, for example the podman `store`.
5. Add the model key to the `worker_env` of the `mini` profile and of the `review` profile:

   ```json
   "QWEN_API_KEY": {"$env": "QWEN_API_KEY"}
   ```

   Each rollout job and each review job names `QWEN_API_KEY` as its key (`--model-key-env`).
   Apart from system variables such as `PATH` and `HOME`, the worker passes a job only the
   variables that a `$env` reference of the job names. If you use the microVM profiles, add
   the entry to `mini-microvm` too.
6. Set the model endpoint and the key in this shell. Replace the placeholder URL with your
   endpoint, and keep credentials out of Git.

   ```bash
   export QWEN_BASE_URL=http://your-private-model-endpoint
   read -rs -p 'Qwen API key: ' QWEN_API_KEY; echo
   export QWEN_API_KEY
   ```

7. Start the Run Store and a worker from this shell:

   ```bash
   python -m sprout.services up runstore-api runstore-worker
   ```

   The command also starts their dependencies: the database `db`, and the checks of the
   podman and the store. The worker inherits `QWEN_API_KEY` from this shell. If the worker
   ran before step 6, restart it with `python -m sprout.services restart runstore-worker`.
8. Export the service variables into this shell:

   ```bash
   eval "$(python -m sprout.services env)"
   ```

   The shell then has `SPROUT_RUNSTORE_TOKEN`, `SPROUT_PODMAN_BIN`, `SPROUT_PODMAN_STORE`
   and the other exports of the services, such as the proxy variables. The podman and the
   store are the ones that the worker uses.

For `bpo`, `shepherd` and `all`, the runner needs `QWEN_API_KEY` in its own shell too. The
selectors of these methods call the model from the evaluator on this host.

The worker's profiles select the sandbox backend and the runtime binary of each job. The
runner names one profile for each kind of job. These are the profiles of
`resources/runstore/config.example.json`, which `init` copies:

| Profile | Runner option | Jobs | Backend |
|---|---|---|---|
| `mini` | `--profile` (default) | rollouts and their branches | podman |
| `grade` | `--grade-profile` (default) | grades | podman |
| `review` | `--review-profile` (default) | the review job of each SPROUT round | no sandbox |
| `mini-microvm` | `--profile mini-microvm` | rollouts and their branches | microVM |
| `grade-microvm` | `--grade-profile grade-microvm` | grades | microVM |

## 2. Download the official datasets and select tasks

Download the datasets and their environment images yourself. This repository holds the
experiment code and the HARD-51 selection list, but no task. The runner reads the task
directories, not only the tabular metadata of a dataset. Each task directory must keep
these files:

- `task.toml`
- the instruction
- the environment configuration
- the official verifier scripts

If the image of a task is not its whole agent environment, the evaluator refuses the task
before any job exists ([evaluation](EVALUATION.md)).

| Dataset | `--dataset` | Official sources | Task directory and selection |
|---|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | [Scale benchmark page](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), [V2 dataset and setup instructions](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2), [HARD-51 task IDs](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/hard51_ids.txt) | the `v2/tasks` directory of the official repository, with `examples/qwen/swebenchpro-v2-hard/selection.json` from Sprout |
| TerminalBench 2.1 | `terminalbench21` | [dataset repository](https://github.com/harbor-framework/terminal-bench-2-1), [Harbor Hub dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) | the `tasks` directory of the repository, with a selection file that you write |
| DeepSWE | `deepswe` | [benchmark website](https://deepswe.datacurve.ai/), [dataset repository](https://github.com/datacurve-ai/deep-swe) | the `tasks` directory of the repository, with a selection file that you write |

Take the HARD-51 task directories and verifiers from the official GitHub repository. The
Scale benchmark page links to that repository. It publishes the V2 tasks under `v2/tasks`
and defines HARD-51 in `v2/hard51_ids.txt`. The same dataset is on
[Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro), where the `hard`
configuration selects HARD-51. The 51 task IDs in Sprout were checked against the official
list on September 27, 2026.

A selection file is a JSON array of task directory names. For example,
`["largest-eigenval"]` selects one TerminalBench task. The runner also accepts a JSON object,
and then reads its first value as the array. Each `--task <name>` adds one more task. The
runner refuses a run in these cases:

- `--task` and `--selection` give no task. The runner never expands a subset.
- A task name occurs two times.
- A task name is not one directory name, for example `a/b`, `.` or `..`.

Prepare the tasks:

1. Clone the task repositories into a directory that you own:

   ```bash
   export DATASETS="$HOME/datasets"
   mkdir -p "$DATASETS"
   git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$DATASETS/SWE-bench_Pro-os"
   git clone https://github.com/harbor-framework/terminal-bench-2-1.git "$DATASETS/terminal-bench-2-1"
   git clone https://github.com/datacurve-ai/deep-swe.git "$DATASETS/deep-swe"
   mkdir -p "$DATASETS/cohorts"
   ```

2. Record the commit of each dataset that you evaluate.
3. Follow the upstream release instructions for the environment images and the verification
   requirements.
4. Write `terminalbench21.json` and `deepswe.json` into `$DATASETS/cohorts`, with the tasks
   that you evaluate. HARD-51 needs no file: its list is in Sprout.
5. Set the variables and the Bash arrays of the runs, one time for each shell session. Adjust
   the paths to your machine.

   ```bash
   export PRO_TASKS="$DATASETS/SWE-bench_Pro-os/v2/tasks"
   export TB_TASKS="$DATASETS/terminal-bench-2-1/tasks"
   export DEEP_TASKS="$DATASETS/deep-swe/tasks"
   export TB_SELECTION="$DATASETS/cohorts/terminalbench21.json"
   export DEEP_SELECTION="$DATASETS/cohorts/deepswe.json"
   export RUNS="$HOME/sprout-runs/qwen"

   COMMON=(--workers 16)
   PRO=(--dataset swebenchpro-v2-hard --tasks-dir "$PRO_TASKS" --selection "$PWD/examples/qwen/swebenchpro-v2-hard/selection.json")
   TB=(--dataset terminalbench21 --tasks-dir "$TB_TASKS" --selection "$TB_SELECTION")
   DEEP=(--dataset deepswe --tasks-dir "$DEEP_TASKS" --selection "$DEEP_SELECTION")
   ```

Keep the task files unchanged across the four methods, and use the same selection for all
four. The runner pins the task files by their sha256 digest, and a branch method refuses a
baseline with another digest. If you change a dataset version, check the compatibility
again.

Keep `$RUNS` outside the Sprout checkout. Output in the checkout can show as untracked files,
and the runner refuses to start when the checkout has uncommitted changes or untracked files.
Also keep `$RUNS` out of `/tmp`, `/var/tmp`, `/dev/shm` and `/run`. The OS empties these
directories on reboot, so the evaluator refuses an output there.

In a new terminal, do these steps again:

1. Activate the virtual environment.
2. Set `QWEN_BASE_URL` and `QWEN_API_KEY` (Section 1, step 6).
3. Export the service variables (Section 1, step 8).
4. Set the variables and the Bash arrays of step 5 above.

## 3. Run one of the 12 experiments

Run the baseline of a dataset to completion first. Then run SPROUT, BPO and Shepherd, each
into a root of its own, with `--baseline-root` set to the baseline root. All three branch
methods then continue the same parent of each task, and `examples.qwen.compare` can put them
in one comparison. Figure 2 shows the four roots and what reads each one.

<p align="center">
  <img src="figures/experiments-roots.svg" width="1000" alt="One dataset has four roots and one comparison. examples.qwen.run with --method baseline writes the baseline root, $RUNS/&lt;dataset&gt;-baseline, which holds manifest-baseline.json, tasks/&lt;task&gt;/baseline/attempt-1 to attempt-N and metrics-baseline.json. The runs with --method sprout, bpo and shepherd each write a root of their own, which holds manifest-&lt;method&gt;.json and tasks/&lt;task&gt;/&lt;method&gt;/. Each of these three runs reads the baseline root through --baseline-root. examples.qwen.compare reads the four roots through --baseline, --sprout, --bpo and --shepherd, and writes comparison.csv and comparison.json into the directory that --output names. Each arrow is a data flow, labeled with the option that names the root or the output directory."/>
</p>
<p align="center"><sub>Figure 2. Each branch method reads the baseline root through <code>--baseline-root</code>, and <code>examples.qwen.compare</code> reads all four roots.</sub></p>

A branch method shares these settings with its baseline. The runner reads them from
`manifest-baseline.json` of the baseline root:

| Manifest key | Set by |
|---|---|
| `dataset` | `--dataset` |
| `dataset_digest` | the files of the selected tasks |
| `tasks` | `--task` and `--selection` |
| `model` | `--model` |
| `reasoning_effort` | fixed: `high` |
| `max_turns` | `--max-turns` |
| `max_output_tokens` | `--max-output-tokens` |
| `command_timeout` | `--command-timeout` |
| `runstore_url` | `--runstore-url` |
| `profile` | `--profile` |
| `grade_profile` | `--grade-profile` |
| `baseline_rollouts` | `--baseline-rollouts` |
| `image_user` | `--no-image-user` |
| `image_env` | `--no-image-env` |

If one of these settings differs, the runner stops before the first stage with
`Shared baseline configuration mismatch: <key>`. A baseline manifest written before a key
existed does not have that key. For such a manifest, the runner uses the value of the runs
before the key:

| Missing key | Value |
|---|---|
| `command_timeout` | 60 seconds |
| `baseline_rollouts` | 4 |
| `image_user` | on |
| `image_env` | on |

A baseline without `image_user` and `image_env` ran Harbor trials, which run the commands as
the image's `USER` with its `ENV`. The other settings are not shared, for example the review
settings and the prompts.

### What the runner does

Before the first stage, the runner does these steps in this order. If a check fails, the
runner stops, and no stage starts.

1. It validates the options. An invalid combination exits with a usage error.
2. It checks that each selected task has a `task.toml`, and computes the digest of the task
   files.
3. If `--plan` is set, it prints the plan and exits 0 (Section 5).
4. It checks that the variable of `--runstore-token-env` holds a token. The default is
   `SPROUT_RUNSTORE_TOKEN`.
5. For `bpo`, `shepherd` and `all`, it checks that the variable of `--model-key-env` holds a
   key. The default is `QWEN_API_KEY`.
6. It checks that `git status --porcelain` prints nothing. If not, it stops with
   `Commit source changes before running`.
7. With `--baseline-root`, it compares the shared settings (table above).
8. If the root has a `manifest-<method>.json`, it compares that file with the new manifest
   in full (Section 7).
9. It creates the lock `.<method>.lock` in the root. If the lock exists, it stops with
   `FileExistsError`.

Then the runner writes `manifest-<method>.json` and `preparation.json`, and runs the tasks.
Figure 3 shows what it does for one task.

<p align="center">
  <img src="figures/protocol-task-flow.svg" width="1000" alt="The runner runs the baseline rollout attempt-1, the parent, with a checkpoint after each turn. If the parent failed, the runner writes retained-checkpoints.json into attempt-1. Then the runner runs attempt-2 to attempt-N, where N is --baseline-rollouts, with only a final snapshot each. Then the runner checks whether attempt-1 resolved the task. If yes, each requested branch method gets the status skipped_initial_success and runs no branch. If no, each of sprout, bpo and shepherd runs as one stage from attempt-1, with at most --max-rollouts rollouts, and stops at the first resolved rollout. Such a stage requires retained-checkpoints.json. If a stage fails, a baseline stage included, the task gets the status infrastructure_error and stays incomplete."/>
</p>
<p align="center"><sub>Figure 3. A task runs its baseline rollouts, and a branch method runs only when the parent failed.</sub></p>

For each task, the runner does these steps in the thread of the task:

1. It runs the baseline rollouts `attempt-1` to `attempt-N` one after another. N is
   `--baseline-rollouts` (default 4). Only `attempt-1` takes per-turn disk checkpoints.
   With `--baseline-root`, the runner reads these rollouts from the baseline root and runs
   none.
2. If `attempt-1` failed, the runner writes `retained-checkpoints.json` into `attempt-1`
   before it runs `attempt-2`. If `attempt-1` has no restorable checkpoint, the runner
   records `retention_error` and continues the baseline.
3. If `attempt-1` resolved the task, the runner gives each requested branch method the status
   `skipped_initial_success`.
4. If `attempt-1` failed, the runner runs each requested branch method as one stage from
   `attempt-1`, in the order `sprout`, `bpo`, `shepherd`. If `attempt-1` has no
   `retained-checkpoints.json`, the task fails with
   `Shared failure lacks retained-checkpoints.json`.
5. It writes the state of the task to `tasks/<task>/<method>.task.json`, and prints one JSON
   line with `task`, `complete`, `initial_resolved` and `error`.

If a stage fails, the runner records the error in the task state, with the status
`infrastructure_error`. The task does not run its remaining stages, and the other tasks
continue. After the last task, the runner writes `metrics-<method>.json`. It exits 0 if
every task completed, and 2 if not.

### What each branch method does

The three branch methods differ in how they pick branch points and in what a branch gets at
its start. Figure 4 shows their fork trees under the default budget. The
[experiment protocol](../examples/qwen/PROTOCOL.md#how-each-method-branches) gives the exact
rules.

<p align="center">
  <img src="figures/protocol-branch-trees.svg" width="1000" alt="Three fork trees over the same failed parent, attempt-1. Each horizontal line is a rollout, time runs to the right, and a dot is a checkpoint. sprout: before each round, a review job reads every rollout so far. In round 1, the reviewer picks up to 4 checkpoints on the parent, and a branch starts at each one with an assistant turn that the reviewer wrote. In round 2, the reviewer picks up to 3 checkpoints on any rollout so far, here on round-1 branches. The branches of a round run at the same time, and the rounds stop at the first resolved rollout. bpo: the selector scores the checkpoints of the parent by first-token entropy and takes up to 7 of them, at least 64 completion tokens apart. It deals 7 branches over them in turn, here 2, 2, 2 and 1 over 4 points. Each branch samples from the restored state with no guidance. shepherd: a meta-agent picks one checkpoint of the parent, and all 7 branches start there with no guidance. bpo and shepherd branches run one at a time and stop at the first resolved one."/>
</p>
<p align="center"><sub>Figure 4. SPROUT branches where a reviewer points, on any rollout so far, and BPO and Shepherd branch only from checkpoints of the parent.</sub></p>

| Method | Branch points | What a branch gets | Order of the branches |
|---|---|---|---|
| `sprout` | checkpoints on any rollout so far, which the reviewer picks in each round | the next assistant turn, which the reviewer writes | The branches of a round run at the same time. The rounds run one after another. |
| `bpo` | up to 7 checkpoints of the parent with the highest first-token entropy, at least 64 completion tokens apart | nothing: the actor samples from the restored state | one at a time |
| `shepherd` | one checkpoint of the parent, which a meta-agent picks | nothing: the actor samples from the restored state | one at a time |

Each branch method stops at the first resolved rollout, or when its budget ends.

### Defaults

The runner uses these defaults:

| Setting | Default | Option | Shared with the baseline |
|---|---|---|---|
| agent | mini-swe-agent 2.4.6 | none | not applicable |
| actor model | `qwen3.8-27b` | `--model` | yes |
| model endpoint | `$QWEN_BASE_URL` | `--model-endpoint` | no |
| variable that holds the model key | `QWEN_API_KEY` | `--model-key-env` | no |
| reasoning effort of the actor | `high` | none | yes |
| what a SPROUT branch gets | an assistant turn that the reviewer writes | none | not applicable |
| model turns per rollout | 300 | `--max-turns` | yes |
| output tokens per model request | 64000 | `--max-output-tokens` | yes |
| time limit of each agent command | 600 s | `--command-timeout` | yes |
| size of each analyst or reviewer request | 400,000 characters, about 100k tokens | `--prompt-budget-chars` | no |
| time limit of each analyst or reviewer request | 3600 s | `--review-timeout` | no |
| baseline rollouts per task | 4 | `--baseline-rollouts` | yes |
| rollouts per branch method and task, the parent included | 8 | `--max-rollouts` | no |
| tasks at the same time | 16 | `--workers` | no |

These rules apply to the defaults:

- If a baseline ran under the earlier 60-second default of `--command-timeout`, give each of
  its branch methods `--command-timeout 60`.
- `--prompt-budget-chars` must be at least 50000. If a reviewer request is larger than the
  budget, the review windows and clips it in a fixed order. `plan-roundN.json` records what
  the review dropped.
- If the endpoint refuses an analyst or reviewer request with HTTP 400, the review sends it
  again with half the budget, at most two times.
- If an analyst or reviewer request times out, the review sends it one more time, with half
  the output budget.
- The baseline runs all its rollouts, also after one of them resolves the task.
  `--baseline-rollouts 1` gives pass@1 alone.
- With `--method sprout` and no `--baseline-root`, the runner first runs the baseline
  rollouts into the same root. With `--baseline-rollouts 1`, SPROUT then branches each failed
  task right after its one rollout.
- `bpo` and `shepherd` require `--baseline-root`.
- A branch method skips each task that its parent resolved.

### Budgets and SPROUT schedules

The budget of a branch method is the number of rollouts on one task, the parent included.
The default is 8: the parent and seven new rollouts. The SPROUT schedule gives the number of
branches in each round. These options set the budget and the schedule:

| Options | SPROUT schedule | Budget |
|---|---|---|
| none | 4 branches, then 3 | 8 |
| `--max-rollouts B`, with B from 2 to 8 | min(4, B − 1) branches, then B − 5 if B is more than 5 | B |
| `--branches W1 W2 …` | one round for each count | `--max-rollouts`. It must be at least 1 + W1 + W2 + …, and at most 128. |
| `--sprout-rounds R --sprout-branches W` | R rounds of W branches. R is 1 to 16, and W is 1 to 4. | 1 + R × W, at most 17 |

These rules apply to the schedules:

- With `--branch-count-mode adaptive`, the default, each count is a maximum. With
  `--branch-count-mode fixed`, the reviewer must pick exactly that count.
- `--max-rollouts` above 8 requires `--branches` or `--sprout-rounds`.
- `--sprout-rounds` requires `--method sprout`, and it cannot go with `--branches`. For a
  schedule wider than 17 rollouts, use `--branches`.
- `bpo` and `shepherd` allow at most 8 rollouts. The evaluator refuses a larger budget for
  them.
- The rounds stop at the first resolved rollout, so a run can use fewer rollouts than its
  budget.

For one wide SPROUT round of N branches, pass `--branches N --branch-count-mode fixed
--max-rollouts N+1`. Use integers, for example `--branches 32 --branch-count-mode fixed
--max-rollouts 33`. The runner records the schedule in the manifest as `branch_caps`, and
passes it to the evaluator. The evaluator has the same limit of 128 rollouts.

To compare the widths 1, 2, 4, 8, 16 and 32, run each width into a root of its own. Give
each width the same `--baseline-root` and `--baseline-rollouts 1`. That baseline must have
run with `--baseline-rollouts 1`.

### The two Harbor protocol flags

The agent's commands, `solve.sh` and `test.sh` run the way Harbor's own trial ran them,
unless a flag turns off a part of it. `examples.qwen.run` and `python -m sprout.eval.evaluate`
take the same two flags ([evaluation](EVALUATION.md)). The runner passes each flag to every
stage, and the manifest records each setting:

| Setting | Default | Option that turns it off | Manifest key |
|---|---|---|---|
| the agent's commands, `solve.sh` and `test.sh` run as the image's `USER` | on | `--no-image-user` | `image_user` |
| those commands get the image's `ENV` again after the change of user | on | `--no-image-env` | `image_env` |

The backend of the worker profile applies both settings ([Run Store](RUNSTORE.md),
**Image USER and ENV**). A root resumes only under the settings that wrote it, because the
runner compares the whole manifest. A branch method must use the settings of its baseline,
because both are shared settings. A branch rollout runs with the settings of its parent
rollout. The grade of a branch uses the flags of the branch method's command.

The local Harbor trials of earlier runs also gave mini a tool primer after the instruction.
The worker gives none. Mini gets the instruction as it is, and each model request gives the
bash tool in its `tools` field.

### Prompts

All analyst, reviewer and Shepherd prompts are in `resources/review/prompts.yaml`. To run an
edited copy, pass `--prompts <path>`. The runner passes it to every evaluator. These files
record the sha256 of the prompts file, so you can tell apart runs under different prompts:

- the manifest, as `prompts_sha256`
- `summary.json` of each stage, as `prompts`
- each `plan-roundN.json`, as `prompts`

The reviewer prompt asks for a turn in three parts:

1. a probe of the suspected defect, from a scratch script under `/tmp`, an inline heredoc or
   existing tests
2. the fix
3. the same probe again, and the existing tests nearest to the change

This is the style of the Kimi K3 branch turns that converted at 0.73 in the fixed 4+3
analysis. The validator refuses a reviewer turn whose command contains
`COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`, because the actor decides when to submit. The
reviewer then gets the error and writes the plan again, up to 3 tries in total.

### Analyst and reviewer settings

The analyst and the reviewer can use another model or another endpoint than the actor. These
options affect only the `sprout` stages:

| Option | What it sets | Default |
|---|---|---|
| `--review-model` | the analyst and reviewer model | `--model` |
| `--review-endpoint` | their Chat Completions endpoint | `--model-endpoint` |
| `--review-key-env` | the worker's variable that holds the key of that endpoint | `--model-key-env` |
| `--review-effort` | their reasoning effort | the actor's, `high` |
| `--review-output-tokens` | the output budget of each analyst and reviewer request, thinking included | `--max-output-tokens` |
| `--structured-review` | answers in schema-constrained JSON instead of fenced plain-text plans. The endpoint must support `json_schema`. | off |

The actor that continues each branch always uses `--model`. The manifest records these
settings under `review`:

- `model`
- `own_endpoint`: whether the review had an endpoint of its own. The manifest never records
  the URL or the key.
- `effort`
- `output_tokens`
- `leak_judge`, for a judged run only

None of them is a shared setting, so a branch method can review with another model than the
one its baseline ran on. The manifest records `structured_review` apart from `review`.

`--leak-judge` checks each turn that the reviewer writes against the task's hidden tests,
before the branch takes the turn. It requires `--method sprout`.
The worker's environment must hold `TYPESAFE_API_KEY`. If the judge flags a turn, the turn
goes back to the reviewer, at most `--leak-judge-resamples` times (default 2). Then the
branch takes the turn, marked. `--leak-judge-threshold` (default 0.75) is the score at which
the judge flags a turn.

### SWE-bench-Pro v2 HARD-51: experiments 1–4

```bash
# 1. baseline
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/swebenchpro-v2-hard-baseline"
# 2. SPROUT
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-sprout"
# 3. entropy-based BPO
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-bpo"
# 4. Shepherd
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/swebenchpro-v2-hard-baseline" --output "$RUNS/swebenchpro-v2-hard-shepherd"
```

### TerminalBench 2.1: experiments 5–8

```bash
# 5. baseline
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/terminalbench21-baseline"
# 6. SPROUT
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-sprout"
# 7. entropy-based BPO
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-bpo"
# 8. Shepherd
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-shepherd"
```

### DeepSWE: experiments 9–12

```bash
# 9. baseline
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/deepswe-baseline"
# 10. SPROUT
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method sprout --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-sprout"
# 11. entropy-based BPO
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method bpo --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-bpo"
# 12. Shepherd
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method shepherd --baseline-root "$RUNS/deepswe-baseline" --output "$RUNS/deepswe-shepherd"
```

Each command runs 16 tasks at the same time by default. Commands that run at the same time
add up. To use fewer resources, reduce `--workers` in `COMMON`. The worker's `--concurrency`
limits the jobs that run at the same time on the host (Section 1).

## 4. Optional: run all four methods for one dataset

Use either the separate runs of Section 3 or this combined mode. With `--method all`, the
runner runs the baseline rollouts of each task, then `sprout`, `bpo` and `shepherd` in this
order, all in one root.

```bash
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method all --output "$RUNS/swebenchpro-v2-hard-all"
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method all --output "$RUNS/terminalbench21-all"
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method all --output "$RUNS/deepswe-all"
```

The runner also writes `manifest-baseline.json` into an `all` root. So a later run can name
that root as `--baseline-root`.

## 5. Preflight checks and one-task smoke tests

To check a command of Section 3 or 4 without model calls or sandboxes, add `--plan` to it.
With `--plan`, the runner does these steps, then exits 0:

1. It validates the options.
2. It checks that each selected task has a `task.toml`, and computes the digest of the task
   files.
3. It prints the manifest, and one evaluator command for each method.

The runner writes no file, calls no model and submits no job. It does not check the
credentials, the Run Store or the images. Only a smoke test checks the network, the images
and the official verifiers.

Each of these commands selects one task and runs all four methods with a budget of two:

```bash
python -m examples.qwen.run --dataset terminalbench21 --method all --tasks-dir "$TB_TASKS" --task largest-eigenval "${COMMON[@]}" --workers 1 --max-turns 2 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/terminalbench21-smoke"
python -m examples.qwen.run --dataset deepswe --method all --tasks-dir "$DEEP_TASKS" --task superjson-error-stack-serialization "${COMMON[@]}" --workers 1 --max-turns 3 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/deepswe-smoke"
```

A smoke command still runs four baseline rollouts. Each branch method has a budget of two
rollouts: the parent and one branch. If the parent resolves the task, the branch methods
skip it. Smoke results check the integration. They are not benchmark scores.

## 6. Export the five-method, four-metric comparison

After the four separate runs of a dataset finish (Section 3), combine their roots:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m examples.qwen.compare \
  --baseline "$RUNS/$DS-baseline" \
  --sprout "$RUNS/$DS-sprout" \
  --bpo "$RUNS/$DS-bpo" \
  --shepherd "$RUNS/$DS-shepherd" \
  --output "$RUNS/$DS-comparison"
```

The command writes `comparison.csv` and `comparison.json` into `--output`, and prints the
JSON. It refuses the roots in these cases:

- A root's manifest differs from the baseline's on a shared setting (Section 3). The error is
  `Incomparable <method> input: <key>`.
- The `baseline_root` of a branch method is not the `--baseline` root. The error is
  `<method> did not use the selected shared baseline`.

The comparison has these five rows:

1. pass@1
2. pass@4 (baseline). The number is `baseline_rollouts`.
3. entropy-based (BPO)
4. Shepherd
5. SPROUT

The four metrics are Resolve Rate, Recovery, Steps and Tokens (M). The
[protocol](../examples/qwen/PROTOCOL.md) defines them. Each row also gives `complete`,
`completed_tasks` and `total_tasks`.

After the runner has written `metrics-<method>.json`, `python -m ops.report runs <root>`
prints its summary.

For the combined `all` mode of Section 4, write a JSON report with the five-row
`comparison` array:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m examples.qwen.report "$RUNS/$DS-all" --method all --save "$RUNS/$DS-all/comparison.json"
```

The runner writes the same report into `metrics-all.json` when it finishes.

## 7. Resume runs and retain checkpoints

To resume a root, run the original command again, with the same options and from the same
commit. The runner then reuses each completed stage, and runs each stage that did not start.

These rules apply to a resumed run:

- The runner compares the new manifest with `manifest-<method>.json` in full. The manifest
  includes `source_commit` and `workers`. If one value differs, the runner stops with
  `Experiment configuration changed; use a fresh output`.
- To continue a root that an older commit wrote, start a fresh root. Name the old baseline
  root in `--baseline-root`, which compares only the shared settings.
- An incomplete stage stays as it is. When the runner reaches its receipt again, the task
  fails with `Incomplete stage retained at <stage>; choose a fresh experiment root to retry`.
  Keep the outputs and the usage records of that stage. Then retry in a fresh root.
- While a runner runs, it holds `.<method>.lock` in the root. If a runner was killed, the
  lock stays, and the next run stops with `FileExistsError`. `python -m ops.report runs`
  shows the pid in the lock and whether it is alive. Remove the lock only when that pid is
  dead.
- `--baseline-root` accepts only a root that this runner wrote. It needs the manifests and
  the stage receipts, so the outputs of other controllers do not work there.

A failed parent keeps its checkpoints for later strategies. Its `attempt-1` holds
`retained-checkpoints.json`. The journals and the exact references to the conversation
prefixes stay in the worker's artifact directory. A recovery also needs the Run Store's
records of the parent job, and the snapshots. Keep all of these:

- the root
- the worker's artifact directory
- the Run Store database
- the snapshot storage: the podman store, or the microVM service

The experiment package deletes no snapshot. Do not sweep the snapshots of a retained parent.
If you do, no later strategy can branch from that parent.

Disk-only snapshots do not keep RAM, live processes or tmpfs. The verifier of a TerminalBench
2.1 task runs in the agent's sandbox, so its grade runs in the sandbox that the rollout kept.
The worker keeps that sandbox for 20 minutes after the rollout ends. If the grade has not
started by then, the rollout scores 0 ([ADR-0004](decisions/0004-grade-in-kept-sandbox.md)).

BPO and Shepherd are inference-time adaptations of the published methods. The
[experiment protocol](../examples/qwen/PROTOCOL.md) gives the budget, the accounting and the
recovery details.

These commands show the state of a root:

| Command | What it prints |
|---|---|
| `python -m ops.report runs "$RUNS/deepswe-baseline"` | the progress and the results of each method, with the current turn of each live rollout. It also reads a root whose stages failed or whose runner was killed. A stage that did not finish shows as `stale`. A receipt with no log beside it shows as `unlaunched`, because no runner started it. |
| `python -m ops.report task "$RUNS/deepswe-sprout" <task> --format mermaid` | the baseline rollouts of one task, its analyst and reviewer rounds, and its branches |
| `python -m ops.report snapshots "$RUNS/deepswe-baseline/tasks/<task>/baseline/attempt-1"` | each snapshot of a retained parent, and whether the store still has it |

## 8. The two backends: podman (the default) and the microVM service

Podman is the default backend. After the store exists, the podman backend runs everything as
your user. The microVM service is the alternative, but it cannot grade TerminalBench 2.1.

### Podman

The profiles `mini` and `grade` run on the rootless podman 6 of this host. That is the
podman that `ops/setup_podman.sh` installs and that `SPROUT_PODMAN_BIN` names. It needs
`btrfs-progs`. The podman backend works like this:

- Each sandbox of a rollout is a container. Its rootfs is a btrfs subvolume of the profile's
  `store`. The default store is `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`.
- A checkpoint is a read-only btrfs snapshot of the rootfs. When the sandbox wrote nothing
  since the last snapshot, the median snapshot took 93 ms.
- A branch is a container that starts from a checkpoint.

Root is necessary one time, to make the store. `ops/setup_podman.sh` runs this command and
the rest of the host setup:

```bash
sudo ops/btrfs_store.sh "$HOME/sprout-btrfs" /var/lib/sprout-btrfs-rootless.img 200G
```

The podman backend has these consequences:

- The first rollout on each task image extracts the image's rootfs into the store: `podman
  pull`, then `podman export | tar`. This occurs one time for each image. For a 3 GB DeepSWE
  image, it took about a minute. Each later container from that image started in less than
  a second.
- `ops/prepull_images.py` does the extraction before a batch.
- Snapshots are disk-only. So a new grade from a TerminalBench 2.1 snapshot cannot use live
  processes.
- The `--baseline-root` of a branch method must have run through the same Run Store and
  profiles, because its snapshots are subvolumes of that store.
- The runtime binary must run inside the image. A musl image needs the static build
  `CGO_ENABLED=0 go build`.

### The microVM service

To use the microVM service, pass the worker's microVM profiles to every command. Each
sandbox is then a Firecracker microVM, and a checkpoint is an incremental disk layer.

A grade in a kept microVM sandbox cannot change the egress of that sandbox, so the grade job
fails ([Run Store](RUNSTORE.md)). Thus the evaluation fails for each task whose verifier
shares the agent's environment, as in TerminalBench 2.1. A shared baseline and its branch
methods must run on the same profiles:

```bash
VM=(--profile mini-microvm --grade-profile grade-microvm)
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" "${VM[@]}" --method baseline --output "$RUNS/terminalbench21-baseline"
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" "${VM[@]}" --method sprout --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-sprout"
```

### Cleanup

Use the same two commands on either backend:

| Command | What it does |
|---|---|
| `python -m sprout.worker sweep <journal> --yes` | deletes the snapshots that the journal names. Without `--yes`, it only prints them. |
| `python -m sprout.worker reap` | removes the sandboxes that a killed run left |

Run Store jobs take their backend from the worker profile: podman, unless the profile names
`microvm` ([Run Store](RUNSTORE.md)).
