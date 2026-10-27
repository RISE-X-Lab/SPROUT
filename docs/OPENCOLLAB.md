# OpenCollab

Sprout can run OpenCollab's Single2 agent through the same worker, sandbox,
checkpoint, search and training-export interfaces as mini-swe-agent. The default
slot remains `mini-swe-agent`. Select `opencollab-base` to run Single2.

## Install

```bash
pip install -e ".[opencollab,harbor,runstore,rl,dev]"
```

The optional extra pins the public OpenCollab commit
`c69a2076f4b54cc97891df432bf81273e3c5ebf1` (SDK version 0.9.3).
It needs Git and network access at installation. The adapter does not modify the
SDK. Without this extra, mini, oracle, nop and the service clients still import
and run normally; SDK-dependent tests skip. CI installs the extra in a separate job.

Use the normal host setup and services from [README](../README.md). Branching
requires a snapshot backend, such as rootless Podman on a btrfs store.

## Run and evaluate

A direct run uses the worker's existing entry point:

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=local
python -m sprout.worker run --slot opencollab-base --model MODEL \
  --sandbox-image docker.io/library/python:3.11-slim --cwd /tmp \
  --runtime-bin runtime/sprout-runtime --journal runs/opencollab.jsonl \
  "Write and test fizzbuzz.py"
```

Evaluation goes through the Run Store and a running worker, as in
[EVALUATION](EVALUATION.md). Add `--slot opencollab-base` to its existing command.
Use the current upstream's fully prepared task images. For older Harbor bundles,
prepare their declared environment setup and standard logging/artifact directories
in the image before scheduling jobs; this adapter does not restore the removed
Harbor trial runner.

The task's workdir is resolved inside the sandbox before the agent starts, and
recorded with its repository root. A branch restores that physical directory,
not the host directory containing its logs.

Single2 receives its native tools (`bash`, `file_read`, `file_write`, `apply_patch`,
`git_diff`, `grep`). These are SDK tools backed by Sprout's `ToolAccess`; the Go
runtime toolset is unchanged. Every environment operation runs in the owned
sandbox, as the image's configured user. File writes use bounded, verified chunks.

## Checkpoints and branches

A recovery point pairs a completed runtime turn's disk snapshot with a
checksummed native transcript prefix and the corresponding SDK session state.
The worker restores the prefix and SDK state before executing the continuation.
A different slot, protocol, checksum or workspace is rejected.

The SDK uses non-streaming Chat Completions through Sprout's metered gateway.
Native context shaping is recorded as the actual replacement context. Recovery
point indexing only admits closed tool-call groups with matching snapshots.
Cancellation waits for an owned runtime command to settle; unknown execution or
checkpoint failure stops the run and remains an infrastructure failure.

SPROUT still chooses branch points and guidance through its existing search and
review jobs. Flat reviewer commands are serialized as the selected parent's
actual command tool. Nested reviewer calls are validated against that parent's
recorded schemas, with unique ids and no reviewer-authored submission.
Both ordinary-content and reasoning-content continuations are supported.

## RL messages

In the RL driver's existing `miles.run_defaults`, select:

```json
{
  "slot": "opencollab-base",
  "extra": {
    "mini": {"environment": {"cwd": "/workspace"}}
  }
}
```

`mini.environment.cwd` is the existing worker workspace configuration key,
shared by these slots. The rest of the driver, worker profiles and task grader
configuration follows [RL_DRIVER](RL_DRIVER.md). Set resources, endpoints and
paths for your host. Search uses the current upstream request's `search` field
and its documented variable-size group semantics; this adapter does not pad
successful searches or change their stopping policy.

The worker exports the SDK's actual messages and tool schemas, plus provenance:
restored prefix messages, reviewer-authored assistant turns, and SDK-generated
submission receipts. The existing driver excludes these from assistant loss.
The sampled submission action remains a policy target. Real tool replies stay
paired with their calls. Original task verifiers provide rewards; grader or
transport failure is never fabricated as reward zero.

Miles owns tokenization, assistant token masks, advantages, loss, optimization,
checkpoints and serving-weight synchronization. No trainer is included here.
Use a Miles client compatible with the current upstream driver's search request
and result contract; older fixed-size Ash clients need their upstream update.

## Historical tool compatibility

For external trajectories recorded with `run_command`, `read_file`, `edit_file`,
`write_file`, `get_status` and `submit_patch`, explicitly select
`opencollab_tool_protocol: review50-v1` in the slot's `extra`. Evaluation exposes
`--opencollab-tool-protocol review50-v1`. This compatibility mode still runs the
Single2 session loop. It does not require or distribute the review50 dataset.

Configure the actual checkout as the task workdir or `mini.environment.cwd`.
Compatibility mode exposes that checkout at `/workspace` before the first model
action; it refuses an existing alias pointing elsewhere.
`submit_patch` produces a real diff receipt and records the SDK submission fact;
it terminates through native Single2. The optional `/opt/review50-baseline`
marker names the task's prepared Git baseline; otherwise the diff uses `HEAD`.
Official grading remains separate. `get_status` reports the actual working-tree
state, deadline and enforced model-turn count. Tool calls are counted and have
no numeric budget limit; an absent limit is `null`.

## Validation

```bash
python -m pytest tests/worker/agents/opencollab -q
python -m pytest -q -m "not slow"
./gate/run.sh
python ops/contracts/ci_check.py
```

The integration tests exercise real Single2 sessions against a scripted model
and an isolated shell fixture: multi-tool turns, snapshots, native restoration,
reviewer injection, context shaping, submission, training provenance, large
writes, cancellation and infrastructure failures. They do not claim policy
learning or benchmark performance. GPU, task images and snapshot storage are
external prerequisites for live RL; none is required by ordinary unit tests.
