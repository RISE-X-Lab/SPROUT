# RL driver

A standalone RL scheduling entry point. It listens on **127.0.0.1:11001** by default
and talks HTTP to the Run Store at **127.0.0.1:18110**. The driver only maps
groups/samples to jobs, submits them, polls them and aggregates the results; the agent,
tools, sandboxes, checkpoints and grader all run in the Run Store's separate workers /
Orchestrator. Sandboxes are rootless podman containers on this host by default (btrfs
store); a worker profile whose `backend` is `microvm` uses AgentENV microVMs instead.
The driver starts no Run Store, worker, model or training, and shutting it down does not
stop jobs already submitted.

```text
Miles POST /rollout-groups
    → rl_driver :11001   sample allocation + RunSpec translation, SQLite ledger
    → Run Store :18110   PostgreSQL jobs / attempts / events / recovery points
    → workers → Orchestrator → mini-swe-agent + Sprout tools (ToolAccess)
                          └→ inference gateway → the requested model_endpoint
    → final snapshot → graded by tasks[task_id].grade (a separately restored sandbox)
    → GET /rollout-groups/{id}: hint-stripped messages + reward
```

## Start and configuration

Run from the Sprout root; the Run Store and the workers must already be running:

```bash
python3.12 -m pip install -e ".[rl]"
export SPROUT_RUNSTORE_TOKEN='<existing-runstore-token>'
python3.12 -m sprout.rl_driver --config src/sprout/rl_driver/config.example.json [--host H] [--port P]
```

`python -m sprout.services up rl-driver` starts it too: it first makes sure the Run Store
(database, API) is running, then starts the driver with `~/.config/sprout/rl_driver.json`
and the generated `SPROUT_RL_DRIVER_TOKEN` (see `docs/SERVICES.md`).

Top-level fields of the configuration (`config.example.json`):

| Field | Default | Meaning |
| --- | --- | --- |
| `runstore_url` | `http://127.0.0.1:18110` | Run Store address; must not carry credentials or a query |
| `runstore_token_env` | `SPROUT_RUNSTORE_TOKEN` | the environment variable holding the Run Store token (required) |
| `driver_token_env` | `SPROUT_RL_DRIVER_TOKEN` | the environment variable holding callers' bearer token; `null` = no authentication, for local use |
| `ledger` | required | SQLite path; a relative path is resolved against **the directory of the configuration file** |
| `poll_interval_s` | `1` | interval between Run Store polls |
| `miles` | none | the Miles adapter configuration; without it `/rollout-groups` returns 503 |

The `miles` section: `profile` (the worker profile in the Run Store, default `mini`),
`run_defaults` (merged into every RunSpec; the slot can only be `mini-swe-agent`, and it
must not contain `extra.rollout_contract`), `image_resources` (`cpu`/`memory_mb` per
sample), `tasks` (per task_id, a grade JobSpec template without `snapshot_id`),
`api_key_env`, `max_samples` (default 1000).

Tokens are read only from the environment and never written to the ledger. A ledger is
bound to the Run Store URL it was first used with, and only one driver process may hold
it at a time (a file lock); execution concurrency is up to the workers. When the
inference endpoint needs authentication, `miles.api_key_env` names a variable in the
**worker's environment**, passed in through the worker profile's `worker_env`; the model
address comes from the request, and no other route's credentials are reused.

## The Miles protocol

A `POST /rollout-groups` request has no version field, and unknown fields are always
refused. The fields are defined in `messages.py` (`MessageRequest`); a complete example
is `request.example.json`:

- `rollout_job_id`, `rollout_id`, `prompt_group_id`, `task_id`, `image`,
  `model_endpoint`, and an optional `model` (otherwise `run_defaults.model`).
- `prompt`: text or a single user message; any other form is refused before queueing.
- `sample_slots` (`sample_slot_id` + `sample_index`, each unique), `max_samples`,
  `minimum_returned_samples`, with 1 ≤ min ≤ max ≤ the number of slots.
- `max_turns`: a positive integer, applied to **each trajectory** on its own. One turn is
  one model call; the tool calls its response triggers take no turn of their own, and
  tool calls are counted but not limited. `max_model_calls` / `max_tool_calls` are
  refused with an explicit error.
- `budgets`: only `max_wall_time_seconds`, which bounds the execution phase (counted from
  submission, queueing included).
- `finalization_timeout_seconds` (default 1800): extra time reserved for the final
  snapshot, the message export and grading.
- `sampling_params`: `temperature`, `top_p`, `top_k`, `stop`, `reasoning_effort` and the
  output length (`max_tokens` / `max_new_tokens` / `max_output_tokens`, which must be
  equal when more than one is given), mapped onto Chat Completions with the output length
  always sent as `max_tokens`; any other field is refused. The `model_endpoint` must
  support `/v1/chat/completions`; it need not return token records.

`image` is handed as-is to the worker's environment preparation; `prime/primeintellect/...`
resolves to the upstream `docker.io/swerebenchv2/...`. Resources come from
`image_resources`.

`tasks[task_id].grade` must exist. The `swe-rebench-v2` grader (`grade_swerebench` in
`src/sprout/runstore/grading.py`): `dataset_path` / `dataset_sha256` pin the task row,
`parser_path` points at the SWE-rebench log-parser file, and `grader_revision` is that
file's `sha256:<digest>`. Grading uses two sandboxes: the one restored from the final
snapshot only exports the agent's diff against `base_commit`; the tests run in a fresh
sandbox of the task image, after that diff and the test_patch are applied. Patches, test
scripts and logs move in and out of sandboxes through `upload_file` / `download_file`,
never inside a command (see Official grading in `docs/RUNSTORE.md`). `resolved=true` only
when every FAIL_TO_PASS and PASS_TO_PASS test passes. `progress` is the share of
FAIL_TO_PASS tests that passed, 0 once any PASS_TO_PASS test fails (and 0 for an empty
patch); the overall pass share, PASS_TO_PASS included, is recorded separately as
`expected_pass_share` and takes no part in weighting.

Every completed grade also carries three facts about the task's hidden tests:
`test_patch_paths` (the files the test_patch touches, old and new paths both, sorted and
deduplicated), `touched_test_paths` (those of them the agent's diff changed too) and
`fail_to_pass` (the task row's FAIL_TO_PASS). Neither the analyst's nor the reviewer's
prompt shows any of the three. A grading failure the agent itself can cause is a
completed 0 (`resolved: false`, `progress: 0.0`, the `reason` named, `error_output`
attached): `model_patch_unavailable` (`git diff` failed in the snapshot, e.g. `.git` or
the base commit was deleted), `model_patch_apply_failed`, `test_patch_apply_failed`,
`tests_timed_out`, `test_runner_killed` (the test script exited with something other than
0 or 1, e.g. killed by the OOM killer). When a sandbox fails to restore or start, a file
cannot be moved in or out, the parser or dataset check does not match, or the parser
errors, the grade job fails and the sample is missing (below).

Grading has a wall-clock limit, `grade_wall_seconds` (adapter configuration, default 600,
at least 240), counted from when the job leaves the queue: a grade past it is no longer
waited for (the driver records `expired`; the Run Store still finishes the job), the test
command's `timeout_s` is the limit minus 120 seconds, and grade jobs are not retried. A
trajectory whose grade failed, expired or gave no verdict is **missing**, not a 0: it is
returned as usual, `reward` is still the 0.0 the wire needs, `metadata.progress` and
`metadata.touched_test_paths` are `null`, and `metadata.zero_reason` names the cause:
`grade_failed`, `grade_expired`, `grade_cancelled`, `grade_skipped`, or
`grade_no_verdict` for a grade that succeeded without a boolean `resolved`; it is `null`
when there is a verdict. A slot whose actor failed has no trajectory; it is listed in the
result's `failed_samples` (`sample_slot_id`, `reason`) and is missing too. In a search
request Miles makes no sample from a missing trajectory and counts it in no statistic;
`rollout/sprout/reward_mean` covers only trajectories with a verdict, and the missing ones
count toward `rollout/sprout/ungradable`. Groups without search are not handled this way
yet.

### Stopping and truncation

On reaching `max_turns` or the execution time limit, the worker first stops the agent,
waits for its tools to finish and saves the final snapshot, then submits the grade. The
trajectory is marked `status: "truncated"` with `stop_reason` `max_turns_reached` or
`timeout`, and its reward still comes from real grading. A rollout that fails midway and
will not be rerun (a crash, a lost sandbox connection, an uncertain command result, a
stop) is cut at its last usable recovery point and graded and returned the same way, with
`stop_reason: "interrupted"`; one that completed no turn at all still counts as failed and
is listed in `failed_samples`. In a cancelled group, queued jobs are cancelled and running
jobs stopped (`POST /v1/jobs/{id}/stop`); grades past their limit are stopped too. A child
process has a 120-second termination window, and the lease/lost-contact checks stay in
force. An uncertain execution result, a safe snapshot that cannot be saved, or a native
record that does not form a valid trajectory is an explicit failure. Tool operations
already under way finish under the drain/cleanup rules; stopping every side effect
instantly is not promised.

### Results

`POST /rollout-groups` returns a `rollout_job_id`; poll `GET /rollout-groups/{id}` until
`status` leaves `queued` / `running`; once the result has been persisted or consumed,
`DELETE /rollout-groups/{id}` (the example request is
`src/sprout/rl_driver/request.example.json`, and the result the adapter returns for it is
`result.example.json` in the same directory; `tests/rl_driver/test_message_rollout.py`
asserts that the two agree, and Miles's tests use both files as fixtures). The second
shared fixture is `search_result.example.json` in the same directory: one search whose two
roots both failed (`FF`), one point per root, and at each point 2 students and 2 repair
turns with 2 repairs each, 14 trajectories in all; `tests/rl_driver/test_gsml_wiring.py`
asserts that the driver returns exactly it for that scenario, and Miles's GSML tests read
a byte-for-byte copy of it (`tests/fast/rollout/sprout/fixtures/sprout_search_result.example.json`).
Whether these three examples are byte-for-byte identical to Miles's copies is checked by
Miles's `test_gsml_wiring.py`; it has to find a Sprout checkout (`$SPROUT_REPO`, or a
`sprout` / `SPROUT` directory beside Miles) and skips when it finds none. If you change an
example here, copy it into Miles's fixtures.

Group `status`: `queued`, `running`, `completed`, `early_stopped` (at least
`minimum_returned_samples` valid trajectories but fewer than all), `failed` (fewer than
the minimum), `cancelled`. A search request's `minimum_returned_samples` is 1: a root
whose actor failed is listed in `failed_samples` and does not fail the group; when every
root fails, the group is `failed` with no trajectories, and Miles treats it as an empty
group. Each trajectory carries its original `sample_slot_id`, `branch_id` /
`parent_branch_id`, the hint-stripped `messages`, `tools`, `reward` (0/1), `status`
(`completed` / `truncated`), `stop_reason` and `metadata` (`job_id` / `attempt_id`,
`source_image`, `graded_snapshot_id`, `progress`, `origin`, `provenance`, `search`,
`logprob_context`, `zero_reason`, and last `touched_test_paths`: the test_patch files the
graded diff changed too, `[]` for none, `null` when the grade is missing or the result
lacks the field). `consumed_budget` sums `model_calls` / `tool_calls`.

### Search: hindsight branches after the root rollouts

A request may carry an optional `search` (`SearchSpec` in `messages.py`): `roots` rollouts
from scratch take the first `roots` slots. Once every root has finished, the driver counts
the planned roots (`planned`), the roots that got a verdict (`graded`: a trajectory was
exported and its grade gave a verdict) and those of them that resolved (`resolved`); a
root whose actor failed, whose export was unavailable or whose grade is missing counts
toward neither. The three counts give the task's label (`search_outcome` in
`src/sprout/rl_driver/outcome.py`; Miles's `miles.rollout.sprout.gsml.search_outcome` is
the same function), and the label decides what runs next:

| Label | Roots | Then (`shape`) |
| --- | --- | --- |
| `TT` | all graded, all resolved | no review, no branches (`none`) |
| `TT_partial` | every graded root resolved, some missing | the same |
| `ungraded` | none graded | the same |
| `TF` | some resolved, some failed | a review for points alone, `tf_student_continuations` students per point (`points_only`) |
| `FF_partial` | none resolved, some failed and some missing | as `TF` |
| `FF` | all graded, all failed | the full review, students and repairs at each point (`full`) |

The review is one Run Store `review` job (SPROUT's analyst and reviewer, run by the same
policy model under the hindsight prompts; see the Review section of `docs/RUNSTORE.md`), at
most one per group. It reads the failed roots, each with the grade result's
`driver.REVIEW_GRADE_KEYS` (`resolved`, `progress`, `test_statuses`, `model_patch`, `log`,
`reason`, `error_output`, plus `fail_to_pass` and `test_patch_paths`, which only the leak
check uses), and the job's `context.rl_driver.outcome` records the label; it reviews each
failed root on its own and picks at most `points` branch points on each
(`points_per_attempt`). A `full` review also writes `candidates` repair turns at each
point, independently, each candidate with its `leak_terms` (the hidden test names or test
file paths the turn mentions that the history before its point never showed); then each
point runs `student_continuations` students, and each repair turn `candidate_continuations`
repairs. A `points_only` review is `guidance: none` with `candidates` 1: it only picks
points, runs `tf_student_continuations` students at each point and no repairs; when
`tf_student_continuations` is 0, these two labels neither review nor branch. A student is
a `none` branch, with nothing added; a repair is an assistant-turn branch that inserts a
repair turn, with `assistant_turn_rendering` `reasoning`: before it executes, the
reviewer-written `content` moves into `reasoning_content` and `content` is left empty
(`render_reasoning` in `src/sprout/harness/agents/mini_swe/continuation.py`), the same
shape as a turn the reasoning policy samples itself.

**Multi-round repair** (`repair_rounds`, default 1): under `full`, once round r's branches
have all finished and been graded, a point that had a student graded, no student resolve
and no repair of any round resolve by R′ (a repair that changed a hidden test file does
not count) hands its repairs that failed grading in round r to round r + 1's review
(`ReviewSpec.round_no`; each repair reviewed on its own, picking 1 point in its history
and writing 1 repair turn), which inserts one more repair turn and continues from that
repair's recovery point. With more than 1 round, `candidates` and
`candidate_continuations` must both be 1; students run in round 1 only. A repair on the
chain still belongs to the point the chain started at (`point_id`; `group` is
`repair:<point_id>:<round>`, the candidate number being the round), and also carries
`round`, `chain_parent_job_id`, `chain_point_id` and `chain_step`; its turn limit is what
its parent repair has left at that point; its `leak_terms` is the union over every turn on
the chain (`null` if any of them is `null`). For a round r > 1, the review's submission
slot is `search-r<r>` (what `reconcile-submission` uses); on failure `search.chain_error`
gives the reason, and when there is no next round `search.chain_end` does. The group's
deadline is `SearchSpec.search_seconds` plus each round's review and branches and the
grading of the rounds before it. For a resolved chain, Miles makes one â sample from each
round's inserted turn, splitting that candidate's share equally among them; if any turn on
the chain is not in the canonical rendering, the whole chain is ineligible.

Branches fill the remaining slots in the order they are created. `SearchSpec.branches()` is
the most branches any label can produce,
`points * max(roots * (student_continuations + candidates * candidate_continuations), (roots - 1) * tf_student_continuations)`
(under `FF` every root failed; under `TF` and `FF_partial` at most `roots - 1` did);
`max_samples` must be at least `roots + branches()`, and Miles allocates slots by the same
formula. For example, with roots 2, points 1, candidates 2, student and candidate
continuations 2 each and tf 2, `branches()` = max(2 × (2 + 2 × 2), 1 × 2) = 12, 14 slots
in all. `minimum_returned_samples` must not exceed `roots` (Miles sends 1). Each branch's
turn limit is what its parent rollout has left at that point, and its deadline is
`wall_time_seconds` from when the branch is planned; the review job's limit is
`review_seconds`; the group's overall deadline adds these two and the finalization time to
the roots' execution deadline. The adapter configuration needs a `review` section:
`profile` (the worker profile that runs reviews) and the reviewer's model-call settings
(`max_output_tokens`, `prompt_chars`, `transcript_tokens`, `temperature`,
`reasoning_effort`, `max_attempts`), and optionally `leak_judge` (`threshold`, `model`,
`resamples`, `api_key_env`, each optional; the TypeSafe key is in the review worker's
environment, `TYPESAFE_API_KEY` by default). With `leak_judge` configured, the review
checks each repair turn with `leak_terms` and with Jev before taking it (Jev reads the test
patch grading actually applied, which the grade result gives as the `test_patch` path and
the driver passes to the review with `REVIEW_GRADE_KEYS`); when either flags the turn, it
goes back to the reviewer to be rewritten, up to `resamples` times (default 2), with the
reviewer told only which branch wrote specific content the actor could not have known, not
which part; a turn still flagged once the tries run out is executed anyway, and the result
is recorded in the metadata (`docs/LEAK_JUDGE.md`).

Each trajectory in the result has one more field, `group`: `root` for a root,
`student:<point_id>` or `repair:<point_id>:<candidate>` for a branch (`null` without
search); Miles computes z-scores within a group. `metadata.search` is `null` without
search, and otherwise:

- a root: `kind: "root"`, `outcome` (the label from the table above), `root_counts`
  (`planned`, `graded`, `resolved`);
- a student: `kind`, the branch point (`point`, `point_id`, `step`, `message_step`,
  `parent_job_id`), `outcome`, `root_counts`;
- a repair: everything a student has, plus `candidate`, `multiplicity` (how many times the
  candidate appeared among the independent hindsight samples), `samples` (the point's
  number of hindsight samples), the review's `leak_terms` (`[]` is clean, `null` when the
  grade had no facts to compare against) and `assistant_turn_rendering: "reasoning"`; when
  the review has `leak_judge` configured, also `leak_judge`: the check of every inserted
  turn on the chain, in round order, each `{leak, leak_terms, jev}`, clean only when `leak`
  is `false` (`true` is flagged, `null` means one of the checks could not run). Without
  it the key is absent.

Miles recomputes the label and the counts from the returned roots and rejects a group
whose `outcome` or `root_counts` disagree. The `search` of `GET /miles-executions/{id}` has
`outcome`, `shape`, `root_counts`, `points`, `skipped` and `error`. When there is no review
(e.g. `every root resolved the task`), the review fails or no point is usable, the group
returns its roots alone and `search.error` says why; unfilled slots are not failures, and
`actual_samples` may be less than `max_samples`.

`tf_student_continuations` is a new field, and both sides check the full key set of
`search`, so Sprout and Miles have to be deployed together. Search requests stored before
the upgrade lack the field: after the upgrade, `GET` / `DELETE /rollout-groups/{id}` on
those groups returns 400; a search still in its root phase ends after the roots, and one
that should have been reviewed gets the `search.error`
`search document predates outcome-keyed review`. Before deploying, let every search group
finish and be consumed.

`metadata.provenance` says which of the `messages` the policy did not sample in this
trajectory: the first `prefix_messages` were restored from the branch point, and
`inserted_messages` lists the indices of every externally inserted assistant turn (this
job's or an ancestor branch's). Their tool results are real, but in this trajectory they
are not trained as policy output (Miles's GSML trains only the eligible inserted turns,
separately, as â samples; below). A rollout from scratch has `0` and `[]`. A trajectory
without provenance, or whose messages are not in cleaned form (cleaning them again would
shift the indices), fails outright.

Messages are exported from the worker's actual native history, tool calls and results
kept. A user-hint branch's hint is wrapped in `<sprout_training_hint>...</sprout_training_hint>`
before it executes; the export deletes these spans from system/user messages only
(developer counts as system), and assistant/tool text is left as it is. An unmarked hint
cannot be removed reliably; an incomplete tool call, non-text content or an unsupported
role fails the trajectory. The driver treats samples independently: it does no branch
selection or sample weighting.

Miles re-tokenizes the hint-stripped messages and computes logprobs, which are not the
behaviour probabilities of sampling with the hint. Do not enable use-rollout-logprobs,
skip the actor forward, TIS or rollout token replay for this path.

The Miles side is `miles.rollout.sprout.rollout_fn.SproutRolloutFn` (Miles branch
`sprout-grpo`; selected with `--rollout-function-path`, it brings its own
`--sprout-rollout-*` arguments), with an example in its `examples/swe-rebench-sprout/`:
`prepare_data.py` writes Miles's `miles.jsonl` and this driver's configuration (grade data
and parser pinned by SHA-256). Each prompt group is one request with
`n_samples_per_prompt` slots (plus the branches' slots for a search). A group without
search is normalized within the group as in GRPO and must return in full. A group that is
rejected, fails in Sprout or returns a shape Miles cannot train is left out of the step and
counted in `rollout/sprout/groups_failed`; the step fails when every group fails or none
has a trainable sample. Miles always sends `top_k` (`--rollout-top-k`, -1 when off), and
Sprout forwards it unchanged, so the engine does not fall back to the checkpoint's
`generation_config` (top_k 20 for Qwen3) and truncate the sampling distribution while the
trainer scores with the full softmax; `--rollout-top-p` below 1 or a positive
`--rollout-top-k` makes Miles derive sampling-support replay, which is incompatible with
re-tokenized trajectories, and is refused at construction.

With `--sprout-rollout-search-points` above 0, each group is one search (the number of
roots is `n_samples_per_prompt`, which must be 2), credited by GSML:
`gsml.assemble_search_group` gives each sample a credit, and `rewards.post_process_gsml`
(`--custom-reward-post-process-path`) turns it into the advantage of each of that sample's
trainable tokens, `A = β·z + λ·lambda_share` (β is (1 − λ)/√2 for a root and β_br for a
branch; an â sample's is `λ·ψ·lambda_share`). z is computed within the conditional group,
over the members with a verdict only. Only `FF` tasks get λ, split by point: at a point
where a student resolved, it goes to the resolving students; otherwise to the repairs of
eligible repair turns and to the â samples (the inserted turn itself as a single-turn
target). A repair whose `touched_test_paths` is non-empty or not a list counts as
unresolved (R′ = 0), and a candidate whose `leak_terms` is not `[]` (`null` included) is
ineligible. λ is in [0, 1], and β_br and ψ are at least 0;
`--sprout-rollout-gsml-c-pre` and `--sprout-rollout-gsml-kappa-plus` are not implemented
and accept only 0. Construction also requires `--advantage-estimator grpo`,
`--calculate-per-token-loss` and `--use-dynamic-global-batch-size`, and refuses
`--normalize-advantages`; before the first step is submitted, it checks that the chat
template trains the inserted turns' `reasoning_content` (`gsml.check_reasoning_is_trained`).
`--calculate-per-token-loss` makes the loss the sum over every trainable token of the step
(each multiplied by its sample's A), every token weighted equally, divided by the step's
token count; that count differs from step to step (samples with A = 0 included), so the
update is consistent rather than exact.

`run_qwen3_8_27b.py` trains on one 8-GPU machine (a TP 2 × CP 4 trainer colocated with
SGLang engines of 2 GPUs each), one search per task, with the search and GSML arguments
below; a task has at most 2 + max(2 × (1 + 3), 2) = 10 slots:

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

Miles's own defaults: `--sprout-rollout-search-points` is 0 (no search),
`--sprout-rollout-search-candidates` is 4, and both time limits are 3600 seconds; the
other `--sprout-rollout-search-*` and `--sprout-rollout-gsml-*` defaults are the
launcher's.

`rollout_only.py` runs one complete rollout on a machine without GPUs: it submits, waits,
re-tokenizes with the policy's chat template, computes the advantage `train.py` would
(GRPO within-group normalization without search, the GSML credit with it), and saves the
result in a format `--load-debug-rollout-data` can read. Such a file cannot yet be handed
to `run_qwen3_8_27b.py --load-rollout-data` for training: the launcher turns on
`--use-dynamic-global-batch-size`, so Miles requires the rollout to carry
`dynamic_global_batch_size`, which is recorded only when a step is trimmed to the
trainer's data-parallel size; `rollout_only.py` has no trainer and does not trim
(`disable_rollout_trim_samples`), and `convert_samples_to_train_data` rejects the file
with an assertion.

## The internal execution interface: /execution-groups

`/execution-groups` is a debugging entry point, not the Miles wire API; internally, Miles
requests are translated into such groups too. Call it directly over HTTP; the template is
`src/sprout/rl_driver/execution-group.example.json`.

- Request fields: `rollout_job_id`, `prompt_group_id`, an optional `context`, `samples`
  (1..1000). Each sample has a unique `sample_slot_id` and one of `run`
  (`JobSpec(kind="rollout", spec=RunSpec)`) or `branch` (`job_id` + `point_id` + a
  required `continuation` + optional `overrides`, limited to `model`, `timeout_s` and
  `budget_usd`; and an optional `budget`, the branch's own `max_turns` / `deadline_at`,
  which overrides the inherited rollout contract and is refused when the parent job has
  no contract). `continuation` decides what the actor receives after the restored prefix;
  its three modes are those of SPROUT's `--branch-guidance`, executed by the same code
  (see the Branch section of `docs/RUNSTORE.md`): `assistant-turn` (insert and actually
  execute an assistant turn), `user-hint` (one user message carrying the hint), `none`
  (nothing added). A plain run's `context` is written into the Run Store job's
  `context.rl_driver`.
- Optional `grade`: a grade JobSpec template, its `spec` holding GradeSpec fields with
  **`snapshot_id` omitted**; the benchmark is `swe-rebench-v2`, and the dataset/parser
  paths must be valid on the worker. After the actor succeeds, the driver picks the
  recovery point available at the final complete tool boundary (or the final snapshot the
  worker captured) and submits the grade; when it cannot prove the final boundary it
  fails, never falling back to an earlier snapshot. `resolved: false` is a valid result
  and does not rerun the actor.
- Branches: first take the `available` points from the Run Store's
  `GET /v1/jobs/{id}/recovery-points` (the job id is in the actor record of the group's
  status), then submit a `branch` under a new group ID. The caller chooses the recovery
  point; the Run Store validates the native prefix and the snapshot at submission and at
  execution, and the driver never restores a sandbox or copies a session itself.

Group status: `queued`, `running`, `quarantined`, `cancelling`, `completed`, `failed`,
`cancelled`; only `ready: true` is terminal. If any actor/grade fails, the group is
`failed` and the other results are kept. `quarantined` needs manual handling, and `wait`
returns it. Each actor/grade records job_id, attempt_id, state/phase and result/error;
artifact paths in results belong to the worker host.

## HTTP interface

With a driver token configured, every endpoint requires `Authorization: Bearer <token>`.

| Endpoint | Behaviour |
| --- | --- |
| `GET /health` | `role`, `miles_configured` |
| `POST /rollout-groups` | submit a Miles request, 202 |
| `GET /rollout-groups/{id}` | the message trajectories and rewards |
| `DELETE /rollout-groups/{id}` | an active group: request cancellation; a terminal group: acknowledge consumption |
| `GET /miles-executions/{id}` | the internal execution state behind a Miles group |
| `POST /execution-groups` | submit an internal execution group, 202 |
| `GET /execution-groups/{id}` (or `/result`) | group status and actor/grade results; query events and recovery points in the Run Store directly |
| `DELETE /execution-groups/{id}` | an active group: 202/cancelling; a terminal group: records `acknowledged_at` |
| `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` | body `{phase, job_id}`, below |

Error codes: a bad argument is 400 (`/rollout-*`) or 422, the same ID with a different
request 409, an unknown ID 404, a failed Run Store query 502 (retry the query; nothing is
submitted twice).

## Idempotency, restarts and cancellation

- `rollout_job_id` is the idempotent identity: resubmitting the same request returns the
  original group; a different request gets 409.
- Before each remote submission the intent is written to the ledger, and the submission
  uses a stable idempotency key, so a retry after a lost response creates no second job.
  The driver never creates a new attempt because an execution failed; infrastructure
  retries are managed by `JobSpec.max_infra_retries` and the worker (0 for Miles
  samples). After a restart it keeps polling the same jobs.
- Cancellation: unsubmitted samples are no longer submitted, queued jobs are cancelled and
  running jobs stopped (`POST /v1/jobs/{id}/stop`), and the driver keeps tracking them to
  a terminal state. A Miles group whose deadline passes takes the same cancellation path
  automatically.
- If a submission's response is still unknown at cancellation, the driver does not resend
  a POST that might create a new job, and keeps the `idempotency_key` in the status. Once
  the operator has found that key's job in the Run Store, they call
  `POST /execution-groups/{id}/samples/{sample}/reconcile-submission` with the body
  `{"phase": "actor"|"grade", "job_id": ...}`; the driver checks the idempotency key and
  resumes tracking, creating no job.
- A DELETE on a terminal group only records a consumption receipt and **deletes no**
  ledger records, execution data or snapshots; there is no TTL/GC. The trainer should
  deduplicate by group/sample identity itself.

## Verification

```bash
python3.12 -m pytest tests/rl_driver tests/harness/test_rollout_contract.py -q
```

With a separate test database in `SPROUT_RUNSTORE_TEST_DSN`, `test_http_postgres.py` also
verifies real TCP → Run Store API → PostgreSQL (controlled worker claims/results, grading
the final snapshot, branch submission, restarts and consumption receipts), each test in a
schema of its own; it runs no model, verifier or sandbox. Without it these tests skip, and
a skip does not mean the wiring passed.

The full chain without GPUs: PostgreSQL + Run Store (`serve` + `worker`, podman
profile) + this driver + Miles's `rollout_only.py` (CPU, needing only the policy's
tokenizer), with any endpoint that serves Chat Completions as the model. It runs the real agent, snapshots
and SWE-rebench grading, and stops short of the optimizer.
