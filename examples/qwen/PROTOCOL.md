# Experiment protocol

This protocol fixes what every Qwen experiment holds constant, so that the rows of a
comparison differ only in the method. Every task runs a fixed number of baseline rollouts.
Each branch method continues the same failed parent, under the same budget of eight
rollouts. The metrics count shared work one time, and unknown usage stays unknown. Read this
protocol before you interpret or publish a comparison, and before you delete experiment
data. The [experiments guide](../../docs/EXPERIMENTS.md) gives the commands and the terms:
rollout, parent, branch, stage and root.

## Fixed settings

All runs use these settings:

| Setting | Value | Where the code sets it |
|---|---|---|
| agent | mini-swe-agent 2.4.6 | `agent` in the manifest |
| actor model | Qwen `qwen3.8-27b` | the default of `--model` |
| reasoning effort | `high` | `REASONING_EFFORT` in `examples/qwen/run.py` |
| branch points | exact closed assistant-turn boundaries | `branching` in the manifest: `assistant-turn` |
| user and environment of the agent's commands, `solve.sh` and `test.sh` | the image's `USER` and `ENV`, as in Harbor, by default. A branch method must use the values of its baseline. | `image_user` and `image_env` in the manifest, from `--no-image-user` and `--no-image-env` |
| task files | pinned by sha256 | `dataset_digest` in the manifest, and `preparation.json` |
| cohort | explicit, from `--task` or `--selection`. The runner never expands a subset. | `parse_args` in `examples/qwen/run.py` |
| task concurrency | 16 by default. The smoke tests use 1. | `--workers` |

The runner and the evaluator pin the task files in these steps:

1. The runner computes one sha256 digest over all files of the selected tasks, outside
   `.git` and `__pycache__`. The manifest records it as `dataset_digest`.
2. The runner writes `preparation.json` with the sha256 of the `task.toml` of each task.
3. Before a stage starts, the evaluator checks the task and its `task.toml` against
   `preparation.json`.
4. After the stage, the evaluator checks that `task.toml` did not change.

If a check fails, the stage fails.

## Budget and the shared parent

Every branch method starts from the same failed rollout, with the same budget. So the methods
differ only in where they branch and in what a branch gets at its start. Figure 1 shows the
steps for one task.

<p align="center">
  <img src="../../docs/figures/protocol-task-flow.svg" width="1000" alt="The runner runs the baseline rollout attempt-1, the parent, with a checkpoint after each turn. If the parent failed, the runner writes retained-checkpoints.json into attempt-1. Then the runner runs attempt-2 to attempt-N, where N is --baseline-rollouts, with only a final snapshot each. Then the runner checks whether attempt-1 resolved the task. If yes, each requested branch method gets the status skipped_initial_success and runs no branch. If no, each of sprout, bpo and shepherd runs as one stage from attempt-1, with at most --max-rollouts rollouts, and stops at the first resolved rollout. Such a stage requires retained-checkpoints.json. If a stage fails, a baseline stage included, the task gets the status infrastructure_error and stays incomplete."/>
</p>
<p align="center"><sub>Figure 1. A task runs its baseline rollouts, and a branch method runs only when the parent failed.</sub></p>

These rules give the budget:

- The runner runs four independent baseline rollouts on every task, also when the first one
  resolves the task. `--baseline-rollouts N` changes the count, and the manifest records it.
- Each branch method continues baseline rollout 1, the parent. It does not take the best of
  the baseline rollouts.
- If the parent resolves the task, no branch method runs on that task.
- If the parent fails, each branch method grades at most seven new rollouts under the
  default budget. With the parent, that is at most eight. `--max-rollouts` sets this budget.

After each stage, the runner checks the number of graded rollouts. If one of these checks
fails, the stage fails:

- A baseline stage grades exactly one rollout.
- A branch stage on a failed parent grades at least one branch.
- A stage grades at most `--max-rollouts` rollouts.

## How each method branches

The three branch methods differ in how they pick branch points and in what a branch gets at
its start. SPROUT can branch on any rollout so far, and it gives each branch a first turn
that a reviewer wrote. BPO and Shepherd branch only on the parent, and a branch gets nothing.
Figure 2 shows the three fork trees.

<p align="center">
  <img src="../../docs/figures/protocol-branch-trees.svg" width="1000" alt="Three fork trees over the same failed parent, attempt-1. Each horizontal line is a rollout, time runs to the right, and a dot is a checkpoint. sprout: before each round, a review job reads every rollout so far. In round 1, the reviewer picks up to 4 checkpoints on the parent, and a branch starts at each one with an assistant turn that the reviewer wrote. In round 2, the reviewer picks up to 3 checkpoints on any rollout so far, here on round-1 branches. The branches of a round run at the same time, and the rounds stop at the first resolved rollout. bpo: the selector scores the checkpoints of the parent by first-token entropy and takes up to 7 of them, at least 64 completion tokens apart. It deals 7 branches over them in turn, here 2, 2, 2 and 1 over 4 points. Each branch samples from the restored state with no guidance. shepherd: a meta-agent picks one checkpoint of the parent, and all 7 branches start there with no guidance. bpo and shepherd branches run one at a time and stop at the first resolved one."/>
</p>
<p align="center"><sub>Figure 2. SPROUT branches where a reviewer points, on any rollout so far, and BPO and Shepherd branch only from checkpoints of the parent.</sub></p>

| Method | Branch points | What a branch gets | Order of the branches | The method stops at |
|---|---|---|---|---|
| `sprout` | In each round, the reviewer picks checkpoints on any rollout so far. | the next assistant turn, which the reviewer writes | The rounds run one after another. The branches of a round run at the same time. | the first resolved rollout, the end of the last round, or a review that picks no branch |
| `bpo` | up to 7 checkpoints of the parent with the highest first-token entropy, at least 64 completion tokens apart | nothing: the actor samples from the restored state | one at a time | the first resolved rollout, or the seventh branch |
| `shepherd` | one checkpoint of the parent, which a meta-agent picks | nothing: the actor samples from the restored state | one at a time | the first resolved rollout, or the seventh branch |

### SPROUT

SPROUT uses the analyst and reviewer control flow of Sprout, and it replaces the next
assistant turn at each branch point. Each round does these steps:

1. A review job reads every rollout so far: the parent and each earlier branch. An analyst
   reads each rollout. Then the reviewer picks the branch points and writes the first
   assistant turn of each branch.
2. The evaluator starts one branch at each picked point. The branches of the round run at the
   same time.
3. If a branch resolves the task, the rounds stop.

These rules apply to the rounds:

- Under the default budget, the rounds have caps of 4 and then 3 branches. Each cap is a
  maximum, because the default `--branch-count-mode` is `adaptive`.
- The reviewer's turn probes the suspected defect, fixes it, and then runs the probe again.
- The validator refuses a reviewer turn that submits. The reviewer then gets the error and
  answers again, up to 3 tries in total.
- If a review picks no branch, the evaluator records `branch_schedule_complete: false` and
  stops. The runner then counts the stage as incomplete.

### BPO

BPO ranks the exact backbone states of the parent by the entropy of the first visible
continuation token. Then it samples siblings without hints. The selector does these steps:

1. It takes each checkpoint of the parent that a provider request follows.
2. For each of these checkpoints, it asks the actor's endpoint for the first visible token,
   with `top_logprobs` 5. Two requests run at the same time.
3. It computes a lower bound on the entropy of that token. The bound uses the top 5 tokens,
   and the residual mass as one more outcome.
4. It takes the checkpoints with the highest entropy first. It skips a checkpoint that is
   closer than 64 completion tokens to a checkpoint that it took. It stops at 7 checkpoints.
5. It deals the 7 branches over the taken checkpoints in turn.

The evaluator then runs the branches one at a time, and stops at the first resolved one. If
the endpoint returns no log probabilities, or a log probability is not valid, the stage
fails. The selector never replaces a missing log probability with a random score.

### Shepherd

Shepherd lets a Qwen meta-agent choose one exact state, then samples siblings without hints:

1. The meta-agent reads the task, the parent's transcript and the eligible checkpoint steps.
   The evaluator cuts the transcript to 60000 tokens. The meta-agent is the actor model on
   the actor's endpoint, with temperature 1.0 and top_p 0.95.
2. The meta-agent names one checkpoint step, with a reason.
3. All 7 branches start from that step, with no guidance. They run one at a time and stop at
   the first resolved one.

If the response is incomplete, the step is not eligible, or the reason is empty, the stage
fails. An invalid selection never falls back to another step.

### Selector requests

The BPO and Shepherd selectors send their requests from the evaluator, to the actor's
`--model` and `--model-endpoint`. The `--review-*` options do not apply to them. Each request
gets up to 8 tries of at most 1800 s each. After a backoff of at most 30 s, the selector
tries again in these cases:

- a timeout or a transport error
- HTTP status 408, 409, 425, 429, 500, 502, 503 or 504
- a response body without exactly one choice

The selector records each try in `meta-request.json`, or in `entropy/step-<N>.request.json`
for BPO.

BPO and Shepherd are **inference-time policy adaptations**. They are not the original
training algorithms, and this protocol does not claim to reproduce their published training
results.

## Retained checkpoints

A failed parent keeps its checkpoints permanently, so that future strategies can branch from
it. When the parent fails, the runner writes `retained-checkpoints.json` into `attempt-1`.
The file holds these items:

- each branch point: its disk-only checkpoint, and the exact reference to the native
  transcript prefix (path, byte length, sha256 and cut)
- the snapshot IDs of the stage
- the canonical journal, and its sha256
- the backend and its store
- how the backend stores the snapshots
- the limits of a restore

The backends store the snapshots of a parent in these ways:

| Backend | Snapshots | Native transcript |
|---|---|---|
| `podman` | read-only btrfs snapshots of the sandbox's rootfs, in the store | stored one time |
| `microvm` | the AgentENV content-addressed shared base, plus disk deltas | stored one time |

So a checkpoint does not copy the transcript or the base. Do not remove the output
directory of a retained parent, and do not sweep its snapshot IDs.

These rules apply to the checkpoints of an experiment:

- A disk-only restore does not keep live processes, tmpfs, RAM or external services.
- Baseline rollouts 2 to N take no per-turn checkpoints. They keep only the final snapshot,
  which every rollout takes for its grade.
- A branch inherits the spec of its parent, so a branch takes per-turn checkpoints too.
  SPROUT's later rounds branch from them. The SPROUT intermediate checkpoints stay available
  after the experiment.
- A successful parent gets no `retained-checkpoints.json`. Its snapshots stay on the backend
  until the operator applies a separate retention policy.
- The experiment package deletes no snapshot, image, layer or run artifact, and it includes
  no deletion script.

## Grading

The grade of each rollout runs the task's official verifier in one of two ways:

- A verifier that runs in the agent's sandbox, as in TerminalBench 2.1, cannot use a
  disk-only snapshot, because a restore loses the live processes. So the grade runs in the
  sandbox that the rollout kept. The worker keeps that sandbox for 20 minutes after the
  rollout ends. If the grade has not started by then, the rollout scores 0
  ([ADR-0004](../../docs/decisions/0004-grade-in-kept-sandbox.md)). The Run Store does not
  retry such a grade, because its first attempt destroys the sandbox.
- For a patch-graded dataset, the grade starts a sandbox from the final disk snapshot of the
  rollout. It collects the artifacts there, and then runs the verifier in a sandbox of its
  own.

The time limits of the actor and the verifier come from the official task definition:

- the rollout: the agent timeout of the task, or 3600 s when the task gives none
- the grade: the verifier timeout of the task, plus the timeouts of its collect commands

## Metrics

The primary comparison has five rows and four metrics. The rows are these:

1. pass@1: the parent alone
2. pass@4 (baseline): all baseline rollouts. The number is `baseline_rollouts`.
3. entropy-based (BPO)
4. Shepherd
5. SPROUT

Each row reports these metrics:

| Metric | Definition | Null when |
|---|---|---|
| Resolve Rate | the tasks that the row resolved within its actual sampling budget, divided by all selected tasks | a task is incomplete |
| Recovery | the initially failed tasks that a later rollout of the row resolved, divided by all initially failed tasks. For pass@1 it is 0 when at least one task failed initially. | a task is incomplete, or no task failed initially |
| Steps | the agent-environment interactions that the row newly executed | never |
| Tokens (M) | the input and output tokens of every actor, critic and selector call, divided by 1e6 | the usage of one call is unknown |

These rules apply to the metrics:

- Cached input is already part of the input, and reasoning output is already part of the
  output. The report does not add either one a second time.
- An incomplete cohort has a null Resolve Rate and Recovery, and an explicit coverage:
  `complete`, `completed_tasks` and `total_tasks`.
- On an incomplete run, Steps and Tokens (M) give the use so far. `tokens_complete` says
  whether every call reported its tokens.
- `metrics-<method>.json` keeps the initial and additional breakdowns and the earlier
  diagnostic fields, for example `Acc` and `Success@5`. So the existing result files stay
  readable.

## Accounting

The report counts the work that each method executed, and counts shared work one time:

- Each row counts the shared initial work one time. Do not add the initial phases of the
  four methods: the parent ran one time.
- The report counts each journal one time, by its sha256. It does not count the copied
  parent journal of a branch stage again.
- Only a journal with a `run.started` event counts as new execution.
- The stage wall time covers setup, actor, critic, snapshot, restore and verification.
- The usage of the critic calls and of the BPO probe requests counts in the additional
  phase.
- A missing provider price is null, not zero. The known charges stay available as lower
  bounds, in `known_cost_usd`.
- The report assumes no dollar price for host CPU or storage.
- A budget failure keeps its steps, tokens and costs.

## Failures, retries and reuse

An infrastructure failure makes a task incomplete, and it never spends a rollout silently:

- If a stage fails, the runner keeps all of its output and usage. The task is incomplete. It
  is not an ordinary negative reward.
- No automatic retry spends an extra rollout. The runner does not run a failed stage again,
  and each rollout job allows no infrastructure retry.
- An HTTP retry stays visible. The report counts each retried or failed model request as a
  call whose usage is unknown. Then the token total of its phase is null, and `known_<key>`
  gives the lower bound.
- Unreported usage and billing stay unknown.

The analyst and reviewer requests follow these rules:

- Each request can take up to `--review-timeout` seconds. The default is 3600 s, 60 minutes
  for high reasoning.
- If a request times out, the review sends it one more time with half the output budget.
- The analyst and the reviewer can use a model or an endpoint of their own:
  `--review-model`, `--review-endpoint`, `--review-key-env` and `--review-effort`. The actor
  keeps `--model`.
- Each request spends at most `--review-output-tokens`.
- The manifest records the review model, whether it had an endpoint of its own, its effort
  and its output budget. It never records a key.

The selector requests of BPO and Shepherd follow the rules of
[Selector requests](#selector-requests).

The runner reuses a completed stage only under the same manifest. An incomplete stage
requires a fresh root, so the runner never erases a failed execution history.
