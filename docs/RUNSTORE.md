# Run Store

Authenticated HTTP control plane on PostgreSQL. A worker manager claims jobs
(`FOR UPDATE SKIP LOCKED`) and runs each attempt in a child process: a
`rollout` through `worker.run.runner.Orchestrator`, slot `mini-swe-agent`
only; a `grade` in a fresh sandbox; a `review` (hindsight over finished
attempts, below) against the model endpoint alone. `rl_driver` submits its
Miles rollouts here.

## Boundaries

- **One trusted operator, one host**, shared DB and artifact directory. Clients
  submit executable tasks: keep the listener local or behind an authenticated proxy.
- **Owned sandboxes: podman by default, or microVMs.** A rollout needs a
  `sandbox_image`; its profile's `backend` decides where it runs, as for
  `harness run`: `podman` when it names none (rootless containers whose rootfs
  is a subvolume of a btrfs `store`, `ops/setup_podman.sh`; the worker and its
  attempt children run the podman `SPROUT_PODMAN_BIN` names, inherited from the
  services' `podman` check, or a profile's `podman.podman_bin`, so restart the
  worker after switching podmans), `microvm` when it
  says so (AgentENV). Any other backend is refused, because after every
  attempt the worker establishes that each sandbox the attempt claimed is gone:
  for podman no container, network or rootfs subvolume left (a leaked one is
  removed the way `harness reap` removes it), for a microVM AgentENV answering
  404; an attempt for which that cannot be established is quarantined. The
  worker owns run ID, journal and `cwd`; requests may not set `session`,
  `journal_path`, `run_id`, `cwd`, `keep_sandbox` or native-restoration `extra`
  keys.
- **Recovery points are closed messages**: a finished model response, all its
  tool results, an exact snapshot and a checksum-pinned native prefix. A
  two-tool response exposes only its end; missing/unsafe captures expose nothing.
  Snapshots are disk-only: no processes, memory, extra mounts or side effects.
- **Storage.** PostgreSQL (`schema.sql`) holds requests, payloads, events, tool
  calls and recovery points; native logs and snapshots stay on disk (the
  podman store's `snapshots/`, or AgentENV's layers). Back up both together.
  Nothing is garbage-collected or replayed from cache.

## Start

Needs a dedicated, persistent PostgreSQL database. `python -m sprout.services up
runstore-api runstore-worker` does all of the below -- the PostgreSQL container, the
DSN and token, `init`, `serve`, `worker` -- from `~/.config/sprout/runstore.json`
(`docs/SERVICES.md`). By hand, from the repo root:

```bash
python3 -m pip install -e ".[mini,runstore]"
export SPROUT_RUNSTORE_DSN='host=127.0.0.1 dbname=sprout_runstore user=sprout_runstore'
export SPROUT_RUNSTORE_TOKEN="$(openssl rand -hex 32)"
python3 -m sprout.runstore init   --config resources/runstore/config.example.json
python3 -m sprout.runstore serve  --config resources/runstore/config.example.json   # --host 127.0.0.1 --port 18110
python3 -m sprout.worker start --config resources/runstore/config.example.json --concurrency 8
```

`init` is idempotent; a volatile `artifact_root` (e.g. `/tmp`) is refused. One
`worker` runs at most `--concurrency` attempts (default 1); extra managers are
fenced by DB leases; `--once` claims one job then drains. Pre-pull the task
images before parallel launches (`ops/prepull_images.py`): on podman this
extracts each image's base into the store once instead of inside the first
attempts; on a microVM it builds the template, and concurrent builds of one
template race. The example's `mini`/`grade` profiles are podman (the store at
`~/sprout-btrfs`, made once with `sudo ops/btrfs_store.sh`); `mini-microvm` /
`grade-microvm` are the same on AgentENV.

**Config** (`config.example.json`; adjust paths): `artifact_root` plus named
`profiles`, each with optional `python` (child interpreter), `run_defaults` /
`grade_defaults` (merged under the request `spec` at submission), `env`
(forwarded to the agent), `worker_env` (worker-side only, e.g. gateway upstream
keys) and `snapshot_repository` (on-disk snapshot checks). Inline credentials
are refused; use `{"$env": "VAR"}`; referenced variables not in `env` are blanked
for the agent. Profiles are fingerprinted at submission, so editing one fails
old jobs rather than silently changing them. Gateway: a rollout whose
`extra.rollout_contract` names a `model_endpoint` gets one, and the contract's
`api_key_env` names the provider key in `worker_env`.

**Timeouts.**

- microVM lease: `max(microvm.sandbox_ttl, ceil(timeout_s) + 600)` s (lease only,
  not budget). A podman sandbox has no lease: it lives until the attempt, or the
  worker's cleanup after it, removes it.
- Rollout `timeout_s` (default 3600) covers child startup and the actor;
  message-export (RL) rollouts add 120 s grace and are capped by the group deadline.
- Grade jobs: `timeout_s` for the tests + 1200 s for preparation/collection.
- `shell` defaults to 300 s per call; the supervisor may stop the run first.

**Stopping.** `POST /v1/jobs/{id}/stop` cancels a queued job and marks a running
(or quarantined) one `stop_requested_at`; the worker reads the mark at its next
heartbeat (about a second), stops the attempt's process the way a deadline
does, reclaims its sandboxes and finishes the job `failed` with `failure_kind`
and `stop_reason` `stopped`, never retried. A finished job is left as it is.

**Salvage.** A message-export (RL) rollout that fails and will not run again --
crashed, lost its sandbox connection, ran an uncertain command, was stopped --
is cut back to its last valid recovery point: the messages up to it and its
snapshot, a pair taken before the next turn's first call ran. It finishes
`succeeded` as `truncated` with `stop_reason: "interrupted"` and what went wrong
under `interrupted`, and is graded like a timeout. A configuration fault, or an
attempt with no closed turn, stays failed.

## Submit and observe

```python
import os
from sprout.runstore.client import Client
from sprout.runstore.specs import JobSpec

client = Client("http://127.0.0.1:18110", os.environ["SPROUT_RUNSTORE_TOKEN"])
job_id = client.submit(JobSpec(
    kind="rollout", profile="mini", context={"task_id": "task-1"},
    spec={"prompt": "The task description...", "model": "my-model",
          "sandbox_image": "my-prepared-image-or-snapshot", "timeout_s": 3600},
), idempotency_key="experiment-1/task-1/parent")
print(client.wait(job_id))
```

Same key + same request returns the same job; a different request returns 409.
`wait()` timing out does not cancel the job; quarantined jobs are returned too.
All endpoints require `Authorization: Bearer $SPROUT_RUNSTORE_TOKEN`; events,
tools and recovery-points take `attempt_id` (default: the active attempt).

| Endpoint | Result |
|---|---|
| `POST /v1/jobs` + `Idempotency-Key` | Accepted job (202) |
| `GET /v1/jobs?state=&limit=100` | Recent jobs, newest first (limit capped at 1000) |
| `GET /v1/jobs/{id}` | Frozen request, lease, state, result/error |
| `GET /v1/jobs/{id}/attempts` | Attempts, heartbeat progress, `execution.scope` |
| `GET /v1/jobs/{id}/events?after=0&limit=1000` | Journal events after `seq` |
| `GET /v1/jobs/{id}/tools?after=0&limit=1000` | Ordered tool calls and responses after `depth` |
| `GET /v1/jobs/{id}/recovery-points` | Message boundaries with `available` |
| `GET /v1/jobs/{id}/result` | Result plus `ready` |
| `POST /v1/jobs/{id}/branch` | New queued branch from `point_id` |
| `POST /v1/jobs/{id}/cancel` | Cancel a queued job only |
| `POST /v1/jobs/{id}/stop` | Cancel a queued job, stop a running one (see Stopping) |

## Branch

`client.branch(job_id, point_id, idempotency_key=..., continuation=...)` takes a
point from `client.recovery_points(job_id)` whose `available` is true. A branch
is a new job from the point's snapshot and native prefix with the parent's
profile, slot, prompt, environment and tools; only `model`, `timeout_s` and
`budget_usd` may change (`overrides`). `available` rechecks the prefix checksum
and the snapshot: on podman the store's `snapshots/<id>` subvolume (directory
listing, as `inspect snapshots` does), on a microVM the AgentENV record (and the
on-disk manifest with `snapshot_repository`).

What the actor is given after the restored prefix is the branch's
`continuation`, required, with the same three modes as a SPROUT branch trial
(`--branch-guidance`), applied by the same code
(`contract/continuation.py`):

| `continuation` | after the prefix |
|---|---|
| `{"guidance": "assistant-turn", "assistant_turn": {role, content, tool_calls}}` | the turn runs through the normal tool path before the actor's first request; its real results join the history. Its call ids must be new to the history (checked at submission) and its calls must fit the tools the run offers (checked by mini) |
| `{"guidance": "user-hint", "hint": "..."}` | a user message with the hint, worded as for a Harbor branch; in message-export mode wrapped in `<sprout_training_hint>` and stripped from training exports |
| `{"guidance": "none"}` | nothing: the actor is queried with the prefix alone |

An assistant-turn continuation may also say how the turn enters the history,
`assistant_turn_rendering`. `content`, the default, inserts it as written and is
never stored in a body, so request hashes and idempotency keys are what they
were. `reasoning` checks the turn in that written form, then inserts
`{"role": "assistant", "content": "", "reasoning_content": <its content>, "tool_calls": <its calls>}`
(`render_reasoning`): the shape of a turn a reasoning policy samples, so the
continuation is sampled after the same rendering a trainer scores. The other modes refuse `reasoning`, and
`fork.origin` records it. The RL driver's repairs use it.

A branch inherits its parent's rollout contract, whose `deadline_at` is the
parent group's (possibly already past) and whose `max_turns` restarts from
zero. `budget` (`{"max_turns": N, "deadline_at": T}`, either key) lays the
branch's own limits over that contract and is refused for a parent without one;
a branch meant to run only the turns its parent had left at the point gets
`max_turns = parent's max_turns - the point's message_step`.

A retry of a branch resumes from its own latest point with `none`: what the
branch added is already in that history.

## Review

`JobSpec(kind="review", spec=ReviewSpec)` is SPROUT's analyst and reviewer
(`methods.branching`, the same prompts `methods.evaluate --method sprout` uses)
run by a worker over finished attempts of one task, so that an RL driver can
branch where hindsight says a continuation is worth trying. The spec names the
`problem`, the `model` and `model_endpoint` (the policy itself, under the review
prompts), `api_key_env` (any Chat Completions endpoint and the worker-environment
variable holding its bearer key, as `methods.evaluate --review-endpoint` and
`--review-key-env` name them for a Harbor run), and `parents`: the attempts to
read, each with its
grade job's result (`resolved`, `progress`, `test_statuses`, `reason`,
`error_output`, the `model_patch` and `log` paths, and, for the leak checks
below, `fail_to_pass`, `test_patch_paths` and the `test_patch` path). Every failed parent is analysed
once; one `none` review over all analyses picks `points` branch points in total
(a point two attempts failed at may be picked once), or, with
`points_per_attempt`, one review per failed attempt picks up to `points` in it;
with `guidance: assistant-turn`, `candidates` independent reviews per point, at
`temperature`, each confined to that point, write the turn to insert there.
Proposals with the same commands merge and keep their `multiplicity`. The result lists each point (`job_id`, `attempt_id`,
`step`, `snapshot_id`) with its candidates; the driver maps `step` to the
attempt's recovery point of that `tool_depth` and submits branches. A review
runs no sandbox and needs no `backend`; its profile section is
`review_defaults`, the worker adds `artifact_root`, and every model request is
a receipt under the attempt directory like a SPROUT review's. A message-export result says which of
its `training_messages` no policy sampled (`training_provenance`):
`prefix_messages` leading messages restored from the branch point, and the
indices in `inserted_messages` of every inserted assistant turn, this job's or
an ancestor's. `training_origin` names the branch point.

Each candidate is `{assistant_turn, why, multiplicity, leak_terms}`.
`leak_terms` (`leak.py`) lists the hidden test names and test file paths its
turn (the first of the merged proposals, the one executed) mentions, in its
content, reasoning or commands, that h_s never shows: h_s is the history a
continuation from that point keeps, hints removed as in the training export,
tool outputs included. The hidden vocabulary comes from the parent grade's
`test_patch_paths` (each path of 6 or more characters, and its basename when
that long and not one every repository has, such as `conftest.py`) and
`fail_to_pass` (each name of 8 or more characters, and each
`[A-Za-z0-9_]+` token of it that long and not all digits); PASS_TO_PASS names
are not terms. Neither fact is ever rendered into an analyst or reviewer prompt
(`grade_of`), but the analyst reads the test log and statuses and the reviewer
the analyst's report, which is how a name reaches a turn. `leak_terms` is `[]`
for a clean turn and `null` when the grade carries neither fact. The check is
lexical and only reports: every candidate is still returned and executed, and
whether one may be credited is the trainer's decision.

With `leak_judge` in the spec (`threshold`, `model`, `resamples`,
`api_key_env`; the key read on the worker) every turn a candidate review writes
is also judged by Jev for copied hidden-test content, against the test patch
the grade names, before the review's plan is taken; a turn either check flags
goes back to the reviewer up to `resamples` times (default 2), told only that
it states what the actor could not know, and is then taken as it is. Such a
candidate also carries `leak_judge`: `{leak, leak_terms, jev}`, `leak` True,
False, or `null` when a check could not run (`docs/LEAK_JUDGE.md`).

## Failure handling

`queued -> running -> succeeded | failed`; uncertain ownership/cleanup -> `quarantined`.
- **Expired leases never requeue directly.** Managers fence the old lease, stop
  the child's process group (boot ID/PID/start time) and reconcile its sandboxes
  first. Investigate other quarantines, then `python3 -m sprout.worker reconcile --config ... --job-id ID`.
- **Frozen payload.** Effective spec, profile and recovery point are frozen in
  `rs_attempts.payload` (+ hash) and sent over stdin (16 MiB max) once the
  child's identity is recorded. The child never touches PostgreSQL and rejects
  missing, truncated, mismatched or late input.
- **Retries.** Up to two infrastructure retries (`max_infra_retries` 0..2,
  30/60 s backoff). A rollout retries only from its latest available recovery
  point with remaining wall-clock budget, and with `budget_usd` only if cost was
  measured and priced. Uncertain shell calls are never replayed. `configuration`
  failures quarantine. A grader failure requeues only the grade job.
- **Supervision.** Per-attempt threads with a watchdog that kills the child on
  timeout, lease loss or shutdown; a blocked thread keeps its slot. SIGINT/SIGTERM
  stops dispatch; after 20 s, unsettled work is left to lease-expiry reconciliation.

## Official grading

Queue `JobSpec(kind="grade", profile="grade", spec=asdict(GradeSpec(...)))`.
`GradeSpec` (`specs.py`) pins a JSON/JSONL dataset by SHA256, `instance_id`,
the submission `snapshot_id`, the task `image`, `grader_revision`, `resources`,
`timeout_s`, `verifier_network` (`deny`/`allow`) and the sandbox settings in
`backend` (the grade profile's `grade_defaults` supply them; podman unless it
names microvm, and `verifier_network` becomes that backend's `allow_internet`).
An unresolved grade is a successful job with `resolved: false`.

The one grader, **`swe-rebench-v2`** (`grading.py`), does what the official
`SWE-rebench-V2/scripts/eval.py` does, in two sandboxes: the final snapshot only
yields the agent's diff (`git add -A`, then `git diff --cached --binary` against
`base_commit`, with external diff, textconv, colour and fsmonitor off and the
`a/` `b/` prefixes fixed, so the agent's repository config cannot change it;
saved as `model.patch`); a pristine sandbox from `image` gets that diff and the
row's `test_patch` via `git apply --3way` (falling back to `patch --fuzz=5`),
runs `install_config.test_cmd` in `/<repo name>` and parses its log with the
`install_config.log_parser` function from `parser_path` (`grader_revision` =
`sha256:<hex>` of that file, the upstream `lib/agent/log_parsers.py`). Resolved
= every `FAIL_TO_PASS` and `PASS_TO_PASS` test passes. Every git command runs
with `-c safe.directory='*'`: a repository owned by another uid makes git
refuse, and the refusal would read as the agent's doing.

No file travels inside a command. The runtime runs a command as one `sh -c`
argument, which Linux caps at `MAX_ARG_STRLEN` (128 KiB), and reports the
`E2BIG` of a longer one as a bare exit 1: staging the patches as base64 in the
command failed every grade of a task whose diff or test patch passed about
96 KiB. Each sandbox gets a private directory, `/tmp/sprout-grade-<uuid4 hex>`
made with `mkdir -m 700` (which refuses a path the snapshot already holds), and
the diff, the test patch, the test script (`eval.sh`) and the test log move
through `upload_file` / `download_file` (`podman cp`, AgentENV's file service). The attempt directory keeps `model.patch`, `test.patch`, `eval.sh`
and `swerebench-tests.log`.

A completed result carries `resolved`, `progress` (the share of `FAIL_TO_PASS`
tests that passed, 0 when a `PASS_TO_PASS` test fails), `grader_revision`, the
`model_patch` path and three facts about the task's hidden tests:
`test_patch_paths` (every path the test patch names, old and new, sorted,
without `/dev/null`), `touched_test_paths` (those the agent's diff names too)
and `fail_to_pass` (the row's list). A grade whose tests finished (exit 0 or 1)
adds `expected_pass_share` (the share of all expected tests), `test_statuses`
and the `log` path. A failure the agent's work can cause is also a completed result,
`resolved: false` and `progress: 0.0` with a `reason` and an `error_output`:
`model_patch_unavailable` (the snapshot's `git diff` exits non-zero, e.g. `.git`
or the base commit removed), `model_patch_apply_failed`,
`test_patch_apply_failed`, `tests_timed_out` (the tests outran `timeout_s`; with
the log so far when it could be fetched) and `test_runner_killed` (an exit code
other than 0 or 1, an OOM kill among them). A grade that cannot be made raises
and the job fails: a sandbox that does not restore or start, a file that does
not move, a dataset or parser whose checksum differs, a parser that raises.

## Validation

`python3 -m pytest tests/runstore -q`. Database tests skip
unless `SPROUT_RUNSTORE_TEST_DSN` names an isolated PostgreSQL database (each test
uses a temporary schema; never point it at production).
