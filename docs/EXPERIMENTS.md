# The Qwen experiments

This guide runs the 12 Qwen experiments: three datasets, each with four methods. You run the
baseline of a dataset first. The three branch methods then continue the first rollout of each
task that the baseline failed. `python -m examples.qwen.compare` exports one comparison for each
dataset. Read this guide to set up a host, and to run, check, resume and export the experiments.
The [experiment protocol](../examples/qwen/PROTOCOL.md) defines the budget, the methods and the
metrics.

This guide uses these terms:

- A **rollout** is one run of the agent on one task, graded by the task's own verifier.
- The **parent** is the first baseline rollout of a task, in `tasks/<task>/baseline/attempt-1`.
- A **branch** is a rollout that continues another rollout from one of its checkpoints.
- A **stage** is one `sprout.eval.evaluate` process. It runs one baseline rollout, or one
  branch method on one task.
- A **root** is the `--output` directory of one run of `examples.qwen.run`.

Each experiment is one dataset with one method. Each cell under a method is the number of an
experiment. Section 2 defines the Bash array of each dataset:

| Dataset | `--dataset` | Bash array | `baseline` | `sprout` | `bpo` | `shepherd` |
|---|---|---|---|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | `PRO` | 1 | 2 | 3 | 4 |
| TerminalBench 2.1 | `terminalbench21` | `TB` | 5 | 6 | 7 | 8 |
| DeepSWE | `deepswe` | `DEEP` | 9 | 10 | 11 | 12 |

Run all commands in Bash from the root of the Sprout checkout.

## 1. Install the runtime and dependencies

One host runs the whole experiment. The runner and the evaluator read the journals that the
worker writes into its `artifact_root`, so they must run on the worker's host. Figure 1 shows the
processes on that host.

<p align="center">
  <img src="figures/experiments-components.svg" width="1000" alt="One experiment host holds seven parts. The runner, examples.qwen.run, runs one thread per task and starts one sprout.eval.evaluate process, the evaluator, for each stage. The evaluator submits the jobs of its stage to runstore-api, the Run Store API, and polls them. runstore-worker, the worker, claims the jobs from the Run Store and starts the sandboxes, on podman with a btrfs store or on AgentENV microVMs, and takes their checkpoints. Two parts are files. The runner writes the manifest, the stage receipts and the metrics into the root that --output names, and the evaluator writes each stage's summary.json there. The worker writes the journals and the verifier logs into its artifact_root directory, and the evaluator reads them from there. Solid arrows mean that a process starts or calls another. Dotted arrows show data that a process writes into files or reads from them."/>
</p>
<p align="center"><sub>Figure 1. The runner starts one evaluator per stage, the evaluator submits jobs to the Run Store, and the worker runs them in sandboxes on the same host.</sub></p>

The worker sends the model requests of the rollouts and the reviews. The evaluator sends the
`bpo` and `shepherd` selector requests itself. [Evaluation](EVALUATION.md) describes the
evaluator, and [Services](SERVICES.md) the two services. The packaged `services.yaml` starts the
worker with `--concurrency 1`, so the worker runs one job at a time. To run more, raise
`--concurrency` in its `runstore-worker` command.

The host needs Linux, with Python 3.12 or later and Go 1.25 or later. It also needs network
access to the registries of the task images. Clone the whole repository, because the runner
imports other Sprout modules and records the commit of the checkout. Install Sprout and start
the services:

1. Install Sprout with the podman backend (Section 8) and the Run Store extra.
   [Set up the host](../README.md#1-set-up-the-host) lists what `ops/setup_podman.sh` installs.

   ```bash
   git clone https://github.com/dreamyang-liu/sprout.git sprout && cd sprout
   ops/setup_podman.sh
   source .venv/bin/activate
   pip install -e ".[runstore]"
   ```

2. Set the model endpoint and its key in this shell. Keep credentials out of Git.
   [Model endpoints](MODEL_ENDPOINTS.md) says what the endpoint must provide.

   ```bash
   export QWEN_BASE_URL=http://your-private-model-endpoint LITELLM_LOCAL_MODEL_COST_MAP=True
   read -rs -p 'Qwen API key: ' QWEN_API_KEY; echo; export QWEN_API_KEY
   ```

3. Run `python -m sprout.services setup --model-key-env QWEN_API_KEY`, and answer its questions.
   The setup writes the service files into `~/.config/sprout`, with `$QWEN_BASE_URL` as the
   default endpoint. It adds `QWEN_API_KEY` to the `worker_env` of each profile whose jobs call
   the model, so that the worker passes the key to those jobs.
   [Guided setup](SERVICES.md#guided-setup) gives each question, and the steps by hand.
4. If the setup did not start them, run
   `python -m sprout.services up runstore-api runstore-worker`. The worker inherits
   `QWEN_API_KEY` from this shell. If a worker ran before step 2 or 3, run
   `python -m sprout.services restart runstore-worker`.
5. Export the service variables, such as `SPROUT_RUNSTORE_TOKEN`, with
   `eval "$(python -m sprout.services env)"`.

The runner needs `QWEN_API_KEY` in its own shell for `bpo` and `shepherd`, alone or within `all`,
because the evaluator sends their selector requests from this host.

## 2. Download the official datasets and select tasks

Download the datasets and their environment images yourself. This repository holds the
experiment code and the HARD-51 selection, but no task. The runner reads the task directories,
and each one must keep these files:

- `task.toml`
- the instruction
- the environment configuration
- the official verifier scripts

| Dataset | Official sources | Task directory and selection |
|---|---|---|
| SWE-bench-Pro v2 HARD-51 | [Scale benchmark page](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2), [V2 dataset and setup instructions](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2), [HARD-51 task IDs](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/hard51_ids.txt), [Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro) (configuration `hard`) | `v2/tasks` of the official repository, with `examples/qwen/swebenchpro-v2-hard/selection.json` from Sprout. Its 51 IDs matched the official list on 2026-09-27. |
| TerminalBench 2.1 | [dataset repository](https://github.com/harbor-framework/terminal-bench-2-1), [Harbor Hub dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest) | `tasks` of the repository, with a selection file that you write |
| DeepSWE | [benchmark website](https://deepswe.datacurve.ai/), [dataset repository](https://github.com/datacurve-ai/deep-swe) | `tasks` of the repository, with a selection file that you write |

A selection file is a JSON array of task directory names, such as `["largest-eigenval"]`. In a
JSON object, the runner reads the first value as the array. Each `--task <name>` adds one task.
The runner refuses a run in these cases:

- `--task` and `--selection` give no task. The runner never expands a subset.
- A task name occurs two times.
- A task name is not one directory name, for example `a/b`, `.` or `..`.

Prepare the tasks:

1. Clone the task repositories into a directory that you own:

   ```bash
   export DATASETS="$HOME/datasets"
   mkdir -p "$DATASETS/cohorts"
   git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$DATASETS/SWE-bench_Pro-os"
   git clone https://github.com/harbor-framework/terminal-bench-2-1.git "$DATASETS/terminal-bench-2-1"
   git clone https://github.com/datacurve-ai/deep-swe.git "$DATASETS/deep-swe"
   ```

2. Record the commit of each dataset that you evaluate.
3. Get the environment images and the verification requirements as the upstream release says.
4. Write `terminalbench21.json` and `deepswe.json` into `$DATASETS/cohorts`. HARD-51 needs no
   file.
5. Set the variables and the Bash arrays of the runs. Adjust the paths to your machine.

   ```bash
   export PRO_TASKS="$DATASETS/SWE-bench_Pro-os/v2/tasks"
   export TB_TASKS="$DATASETS/terminal-bench-2-1/tasks"
   export DEEP_TASKS="$DATASETS/deep-swe/tasks"
   export RUNS="$HOME/sprout-runs/qwen"

   COMMON=(--workers 16)
   PRO=(--dataset swebenchpro-v2-hard --tasks-dir "$PRO_TASKS" --selection "$PWD/examples/qwen/swebenchpro-v2-hard/selection.json")
   TB=(--dataset terminalbench21 --tasks-dir "$TB_TASKS" --selection "$DATASETS/cohorts/terminalbench21.json")
   DEEP=(--dataset deepswe --tasks-dir "$DEEP_TASKS" --selection "$DATASETS/cohorts/deepswe.json")
   ```

Use the same task files and the same selection for all four methods. The runner pins the task
files by their sha256 digest, and a branch method refuses a baseline with another digest.

Keep `$RUNS` outside the Sprout checkout, because the runner refuses a checkout with untracked
files. Also keep it out of a directory that the OS empties on reboot, such as `/tmp`. The
evaluator refuses an output there.

In a new terminal, do these steps again:

1. Activate `.venv`.
2. Set `QWEN_BASE_URL` and `QWEN_API_KEY` (Section 1, step 2).
3. Export the service variables (Section 1, step 5).
4. Set `DATASETS` (step 1), and the variables and the Bash arrays of step 5.

## 3. Run one of the 12 experiments

Run the baseline of a dataset to completion first. Then run each branch method into a root of
its own, with `--baseline-root` set to the baseline root. All three methods then continue the
same parent of each task, and `examples.qwen.compare` can put them in one comparison. Figure 2
shows the four roots and what reads each one.

<p align="center">
  <img src="figures/experiments-roots.svg" width="1000" alt="One dataset has four roots and one comparison. examples.qwen.run with --method baseline writes the baseline root, $RUNS/&lt;dataset&gt;-baseline, which holds manifest-baseline.json, tasks/&lt;task&gt;/baseline/attempt-1 to attempt-N and metrics-baseline.json. The runs with --method sprout, bpo and shepherd each write a root of their own, which holds manifest-&lt;method&gt;.json and tasks/&lt;task&gt;/&lt;method&gt;/. Each of these three runs reads the baseline root through --baseline-root. examples.qwen.compare reads the four roots through --baseline, --sprout, --bpo and --shepherd, and writes comparison.csv and comparison.json into the directory that --output names. Each arrow is a data flow, labeled with the option that names the root or the output directory."/>
</p>
<p align="center"><sub>Figure 2. Each branch method reads the baseline root through <code>--baseline-root</code>, and <code>examples.qwen.compare</code> reads all four roots.</sub></p>

The 12 commands differ only in the dataset and the method. Set `DS` to the `--dataset` of the
table at the top, and `DATA` to its Bash array. Then run the baseline, and then each branch method:

```bash
DS=swebenchpro-v2-hard; DATA=("${PRO[@]}")  # or terminalbench21 with TB, deepswe with DEEP
python -m examples.qwen.run "${DATA[@]}" "${COMMON[@]}" --method baseline --output "$RUNS/$DS-baseline"
for METHOD in sprout bpo shepherd; do
  python -m examples.qwen.run "${DATA[@]}" "${COMMON[@]}" --method "$METHOD" \
    --baseline-root "$RUNS/$DS-baseline" --output "$RUNS/$DS-$METHOD"
done
```

The loop runs the branch methods one after another. You can also run them at the same time, each
in a shell of its own. Then their `--workers` add up. The worker's `--concurrency` limits the jobs
that run at the same time on the host (Section 1).

### What the runner does

Before the first stage, the runner makes these checks in this order. If a check fails, the
runner stops, and no stage starts:

1. It validates the options. An invalid combination exits with a usage error.
2. It checks that each task has a `task.toml`, and computes the digest of the task files.
3. With `--plan`, it prints the plan and exits 0 (Section 5).
4. It checks that the variable of `--runstore-token-env` holds the Run Store token.
5. For `bpo` and `shepherd`, alone or within `all`, it checks that the variable of
   `--model-key-env` holds a key.
6. If `git status --porcelain` prints a line, it stops with `Commit source changes before running`.
7. With `--baseline-root`, it compares the [shared settings](#defaults-and-shared-settings).
8. If the root has a `manifest-<method>.json`, it compares it with the new manifest (Section 7).
9. It creates the lock `.<method>.lock`. If the lock exists, it stops with `FileExistsError`.

Then the runner writes `manifest-<method>.json` and `preparation.json`, and runs each task as
Figure 3 shows. The [protocol](../examples/qwen/PROTOCOL.md#budget-and-the-shared-parent) gives
the budget rules, and Figure 5 in Section 7 shows what the runner does with each stage.

<p align="center">
  <img src="figures/protocol-task-flow.svg" width="1000" alt="The runner runs the baseline rollout attempt-1, the parent, with a checkpoint after each turn. If the parent failed, the runner writes retained-checkpoints.json into attempt-1. Then the runner runs attempt-2 to attempt-N, where N is --baseline-rollouts, with only a final snapshot each. Then the runner checks whether attempt-1 resolved the task. If yes, each requested branch method gets the status skipped_initial_success and runs no branch. If no, each of sprout, bpo and shepherd runs as one stage from attempt-1, with at most --max-rollouts rollouts, and stops at the first resolved rollout. Such a stage requires retained-checkpoints.json. If a stage fails, a baseline stage included, the task gets the status infrastructure_error and stays incomplete."/>
</p>
<p align="center"><sub>Figure 3. A task runs its baseline rollouts, and a branch method runs only when the parent failed.</sub></p>

The runner writes the state of each task to `tasks/<task>/<method>.task.json`, and prints a
one-line JSON summary of it. At the end, it writes `metrics-<method>.json`. It exits 0 if every
task completed, and 2 if not.

### What each branch method does

The three branch methods differ in where they branch and in what a branch gets at its start.
Figure 4 shows their fork trees under the default budget. The
[protocol](../examples/qwen/PROTOCOL.md#how-each-method-branches) gives the rules of each method.

<p align="center">
  <img src="figures/protocol-branch-trees.svg" width="1000" alt="Three fork trees over the same failed parent, attempt-1. Each horizontal line is a rollout, time runs to the right, and a dot is a checkpoint. sprout: before each round, a review job reads every rollout so far. In round 1, the reviewer picks up to 4 checkpoints on the parent, and a branch starts at each one with an assistant turn that the reviewer wrote. In round 2, the reviewer picks up to 3 checkpoints on any rollout so far, here on round-1 branches. The branches of a round run at the same time, and the rounds stop at the first resolved rollout. bpo: the selector scores the checkpoints of the parent by first-token entropy and takes up to 7 of them, at least 64 completion tokens apart. It deals 7 branches over them in turn, here 2, 2, 2 and 1 over 4 points. Each branch samples from the restored state with no guidance. shepherd: a meta-agent picks one checkpoint of the parent, and all 7 branches start there with no guidance. bpo and shepherd branches run one at a time and stop at the first resolved one."/>
</p>
<p align="center"><sub>Figure 4. SPROUT branches where a reviewer points, on any rollout so far, and BPO and Shepherd branch only from checkpoints of the parent.</sub></p>

### Defaults and shared settings

The runner uses these defaults. A branch method must use the shared settings of its baseline,
which the runner reads from `manifest-baseline.json` in the baseline root. If one differs, the
runner stops before the first stage with `Shared baseline configuration mismatch: <key>`.

| Setting | Default | Option | Shared as |
|---|---|---|---|
| task set | none | `--dataset`, `--task`, `--selection` | `dataset`, `tasks`, and `dataset_digest` of the task files |
| agent | mini-swe-agent 2.4.6 | none | not shared |
| actor model, and its reasoning effort | `qwen3.8-27b`, `high` | `--model`, none | `model`, `reasoning_effort` |
| model endpoint, and the variable of its key | `$QWEN_BASE_URL`, `QWEN_API_KEY` | `--model-endpoint`, `--model-key-env` | not shared |
| model turns per rollout | 300 | `--max-turns` | `max_turns` |
| output tokens per model request | 64000 | `--max-output-tokens` | `max_output_tokens` |
| time limit of each agent command | 600 s | `--command-timeout` | `command_timeout`, 60 s if missing |
| Run Store, rollout profile and grade profile | `http://127.0.0.1:18110`, `mini`, `grade` | `--runstore-url`, `--profile`, `--grade-profile` | `runstore_url`, `profile`, `grade_profile` |
| the agent's commands, `solve.sh` and `test.sh` run as the image's `USER`, with its `ENV`, as Harbor runs them | on | `--no-image-user`, `--no-image-env` | `image_user`, `image_env`, on if missing |
| size of each analyst or reviewer request | 400,000 characters, about 100k tokens. The minimum is 50000. | `--prompt-budget-chars` | not shared |
| time limit of each analyst or reviewer request | 3600 s | `--review-timeout` | not shared |
| baseline rollouts per task | 4. With 1, the baseline gives pass@1 alone. | `--baseline-rollouts` | `baseline_rollouts`, 4 if missing |
| rollouts per branch method and task, the parent included | 8 | `--max-rollouts` | not shared |
| tasks at the same time | 16 | `--workers` | not shared |

A baseline manifest written before a key existed lacks the key. The runner then uses the value
that the last column gives. So give a branch method the options of its baseline. For example, a
baseline that ran under the earlier 60-second default needs `--command-timeout 60`.

With `--method sprout` and no `--baseline-root`, the runner first runs the baseline rollouts into
the same root. `bpo` and `shepherd` require `--baseline-root`. The Run Store guide says how the
backend applies the [image flags](RUNSTORE.md#image-user-and-env), and how a review shrinks or
retries its [requests](RUNSTORE.md#review-requests).

### Budgets and SPROUT schedules

The budget of a branch method is the number of its rollouts on one task, the parent included.
The SPROUT schedule gives the number of branches in each round. These options set both:

| Options | SPROUT schedule | Budget |
|---|---|---|
| none | 4 branches, then 3 | 8 |
| `--max-rollouts B`, with B from 2 to 8 | min(4, B − 1) branches, then B − 5 if B is more than 5 | B |
| `--branches W1 W2 …` | one round for each count | `--max-rollouts`, from 1 + W1 + W2 + … to 128 |
| `--sprout-rounds R --sprout-branches W`, only with `--method sprout` and without `--branches` | R rounds of W branches. R is 1 to 16, and W is 1 to 4. | 1 + R × W, at most 17 |

These rules apply to the schedules:

- With `--branch-count-mode adaptive`, the default, each count is a maximum. With
  `--branch-count-mode fixed`, the reviewer must pick exactly that count.
- `bpo` and `shepherd` allow at most 8 rollouts.
- The rounds stop at the first resolved rollout, so a run can use fewer rollouts than its budget.

For one wide SPROUT round of 32 branches, pass `--branches 32 --branch-count-mode fixed
--max-rollouts 33`. To compare widths, run each width into a root of its own, with the same
`--baseline-root` and `--baseline-rollouts 1`. That baseline must have run with
`--baseline-rollouts 1`.

### Review settings and prompts

By default, the analyst and the reviewer use the settings of the actor. The
`--review-*` options change them for the `sprout` stages, and `--help` lists these options. The
actor of each branch always uses `--model`. The review settings are not shared settings. The
manifest records them under `review`, but never the endpoint URL or a key. These options also
change a run:

| Option | Effect |
|---|---|
| `--prompts <path>` | The analyst, reviewer and Shepherd prompts come from the file instead of `resources/review/prompts.yaml`. The sha256 of the file goes into the manifest as `prompts_sha256`, and into each `summary.json` and `plan-roundN.json` as `prompts`. |
| `--structured-review` | The analyst and the reviewer answer in schema-constrained JSON. The endpoint must support `json_schema` ([`--structured-review`](MODEL_ENDPOINTS.md#--structured-review)). |
| `--leak-judge` | A judge checks each reviewer turn against the hidden tests, before a branch takes it ([Leak checks](RUNSTORE.md#leak-checks)). The worker needs `TYPESAFE_API_KEY`. Only for `sprout`. |

## 4. Optional: run all four methods for one dataset

With `--method all`, the runner runs the baseline rollouts of each task, then the three branch
methods in the column order of the table at the top, all in one root. Set `DS` and `DATA` as in
Section 3:

```bash
python -m examples.qwen.run "${DATA[@]}" "${COMMON[@]}" --method all --output "$RUNS/$DS-all"
```

The runner also writes `manifest-baseline.json` into an `all` root, so a later run can name that
root as `--baseline-root`.

## 5. Preflight checks and one-task smoke tests

To check a command of Section 3 or 4 without model calls or sandboxes, add `--plan` to it. The
runner then does steps 1 to 3 of [What the runner does](#what-the-runner-does). It prints the
manifest and one evaluator command for each method, and exits 0. It writes no file, and it
reaches neither the model nor the Run Store.

Only a smoke test runs the task images and their official verifiers. Each of these
commands runs all four methods on one task, with a budget of two:

```bash
python -m examples.qwen.run --dataset terminalbench21 --method all --tasks-dir "$TB_TASKS" --task largest-eigenval --workers 1 --max-turns 2 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/terminalbench21-smoke"
python -m examples.qwen.run --dataset deepswe --method all --tasks-dir "$DEEP_TASKS" --task superjson-error-stack-serialization --workers 1 --max-turns 3 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/deepswe-smoke"
```

A smoke command still runs four baseline rollouts, and each branch method gets one branch. Smoke
results check the integration. They are not benchmark scores.

## 6. Export the five-method, four-metric comparison

After the four separate runs of a dataset finish (Section 3), combine their roots:

```bash
python -m examples.qwen.compare --baseline "$RUNS/$DS-baseline" --sprout "$RUNS/$DS-sprout" \
  --bpo "$RUNS/$DS-bpo" --shepherd "$RUNS/$DS-shepherd" --output "$RUNS/$DS-comparison"
```

The command writes `comparison.csv` and `comparison.json` into `--output`, and prints the JSON.
It refuses the roots in these cases:

- A root's manifest differs from the baseline's on a shared setting. The error is
  `Incomparable <method> input: <key>`.
- The `baseline_root` of a branch method is not the `--baseline` root. The error is
  `<method> did not use the selected shared baseline`.

The comparison has the five rows and the four metrics of the
[protocol](../examples/qwen/PROTOCOL.md#metrics). Each row also gives its task coverage, as the
protocol defines it.

For an `all` root (Section 4), the runner writes the same report into `metrics-all.json` when it
finishes. To write it again, run:

```bash
python -m examples.qwen.report "$RUNS/$DS-all" --method all --save "$RUNS/$DS-all/comparison.json"
```

## 7. Resume runs and retain checkpoints

To resume a root, run the original command again, with the same options and from the same
commit. The runner reuses each completed stage, and runs each stage that did not start. Figure 5
shows what the runner does with a stage when it reaches it.

<p align="center">
  <img src="figures/experiments-stage-reuse.svg" width="1000" alt="The tests that examples.qwen.run makes for each stage of a task, from the top, on the first run and on each later run in the same root. First: does the receipt &lt;stage&gt;.stage.json exist? If yes, the runner reads complete in the receipt. If complete is true, the runner reuses the stage: it reads the attempts from summary.json and runs no job. If complete is false, the task fails with Incomplete stage retained at &lt;stage&gt;. If no receipt exists, the runner asks whether the stage directory is empty or absent. If not, the task fails with FileExistsError. If yes, the runner writes the receipt with complete false, then runs sprout.eval.evaluate for the stage. Then it asks whether the evaluator exited 0 and the stage checks passed. If yes, the runner writes complete true into the receipt, and the attempts go to the task. If no, the receipt keeps complete false with the error, and the task fails. A task that fails gets the status infrastructure_error and runs no more stages. The other tasks go on."/>
</p>
<p align="center"><sub>Figure 5. The runner reuses a stage whose receipt is complete, runs a stage that has no receipt, and fails the task at an incomplete one.</sub></p>

The stage checks are those of `attempts()` and `stage()` in `examples/qwen/run.py`, with the
rollout counts of the [protocol](../examples/qwen/PROTOCOL.md#budget-and-the-shared-parent).
These rules apply to a resumed run:

- The runner compares the new manifest with `manifest-<method>.json` in full, `source_commit`
  and `workers` included. If one value differs, it stops with
  `Experiment configuration changed; use a fresh output`.
- To continue the work of an older commit, start a fresh root. Name the old baseline root in
  `--baseline-root`, which compares only the shared settings.
- To retry an incomplete stage, use a fresh root. Keep the outputs and the usage records of the
  old stage.
- A killed runner leaves its lock `.<method>.lock`, and the next run stops with
  `FileExistsError`. `python -m ops.report runs` shows the pid in the lock and whether it is
  alive. Remove the lock only when that pid is dead.
- `--baseline-root` accepts only a root that this runner wrote, because it reads the manifests
  and the stage receipts.

A failed parent keeps its checkpoints, so that later strategies can branch from it
([Retained checkpoints](../examples/qwen/PROTOCOL.md#retained-checkpoints)). A branch from it
needs all of these. Keep them:

- the root
- the worker's `artifact_root`, which holds the journals
- the Run Store database
- the snapshot storage: the podman store, or the microVM service

The experiment package deletes no snapshot. If you sweep the snapshots of a retained parent, no
later strategy can branch from that parent.

These commands show the state of a root:

| Command | What it prints |
|---|---|
| `python -m ops.report runs "$RUNS/deepswe-baseline"` | The progress and the results of each method, with the turn of each live rollout. It also reads a killed run. A stage that did not finish shows as `stale`. A receipt with no log beside it shows as `unlaunched`. |
| `python -m ops.report task "$RUNS/deepswe-sprout" <task> --format mermaid` | The baseline rollouts of one task, its analyst and reviewer rounds, and its branches |
| `python -m ops.report snapshots "$RUNS/deepswe-baseline/tasks/<task>/baseline/attempt-1"` | Each snapshot of a retained parent, and whether the store still has it |

## 8. The two backends: podman (the default) and the microVM service

Podman is the default backend. The microVM service is the alternative, but it cannot grade
TerminalBench 2.1. The worker profile of each job selects its backend. The setup copies the
profiles from `resources/runstore/config.example.json`:

| | podman | microVM |
|---|---|---|
| Rollout profile, `--profile` | `mini`, the default | `mini-microvm` |
| Grade profile, `--grade-profile` | `grade`, the default | `grade-microvm` |
| Sandbox | a rootless podman 6 container, whose rootfs is a btrfs subvolume of the profile's `store` | a Firecracker microVM of AgentENV, which you deploy separately |
| Checkpoint | a read-only btrfs snapshot of the rootfs | an incremental disk layer |
| Setup | `ops/setup_podman.sh` | the AgentENV service |
| A grade whose verifier shares the agent's sandbox, as in TerminalBench 2.1 | runs in the sandbox that the rollout kept | fails, because a kept microVM cannot switch its egress to the verifier's network policy |

Both backends take disk-only snapshots, without RAM or live processes, so such a grade
[needs the kept sandbox](../examples/qwen/PROTOCOL.md#grading). The `review` profile, the default
of `--review-profile`, runs each SPROUT review without a sandbox.

To use the microVM service, add `--profile mini-microvm --grade-profile grade-microvm` to each
command of a dataset, the baseline and its branch methods alike. Use it only for a dataset whose
verifier has a sandbox of its own, such as DeepSWE.

### Podman

The store is `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`. `ops/setup_podman.sh` makes it with
`sudo ops/btrfs_store.sh "$HOME/sprout-btrfs" /var/lib/sprout-btrfs-rootless.img 200G`, the only
step that needs root. The podman backend has these consequences:

- The first rollout on each task image extracts the image's rootfs into the store. For a 3 GB
  DeepSWE image, this took about a minute. Each later container from that image started in less
  than a second. `ops/prepull_images.py` does the extraction before a batch.
- The runtime binary must run inside the image. A musl image needs the static build
  `CGO_ENABLED=0 go build`.

### Cleanup

Use the same two commands on either backend:

| Command | What it does |
|---|---|
| `python -m sprout.worker sweep <journal> --yes` | Deletes the snapshots that the journal names. Without `--yes`, it only prints them. |
| `python -m sprout.worker reap` | Removes the sandboxes that a killed run left |
