# harbor_plugin: Sprout inside a Harbor Trial

Harbor (`harbor==0.22.0`, the `harbor` extra in `pyproject.toml`; API surface pinned by
`ops/contracts/harbor.yaml`, Python >= 3.12) owns the task: environment start,
agent run, artifact collection, verifier, `result.json`. This package plugs Sprout
into its `BaseEnvironment` and `BaseAgent`; the actor is always mini-swe-agent,
and the sandbox is an AgentENV microVM or a rootless podman container on a btrfs
store -- the same trial either way.

## Modules and import paths

Harbor loads each class from a `TrialConfig` `import_path` (`module:Class`):

| Import path | What it is |
|---|---|
| `harbor_plugin.environment:SandboxEnvironment` | The base: Harbor's `BaseEnvironment` over one `SandboxSession` -- exec, upload/download, the `tests/Dockerfile` overlay replay, receipts, the retained stop. A backend subclass adds how the image is pinned, the backend section, the shape and how egress is toggled. |
| `harbor_plugin.agentenv:AgentENVEnvironment` | `SandboxEnvironment` (`type()` = `agentenv`) on an AgentENV microVM. |
| `harbor_plugin.podman:PodmanEnvironment` | `SandboxEnvironment` (`type()` = `podman`) on a rootless podman container whose rootfs is a btrfs subvolume of a store. |
| `harbor_plugin.agentenv_mini:AgentENVMini` | Host mini-swe-agent (`name()` = `sprout-agentenv-mini-swe`) driving either sandbox. |
| `harbor_plugin.branching:SnapshotEnvironment` | `AgentENVEnvironment` that boots from a recorded snapshot. |
| `harbor_plugin.branching:PodmanSnapshotEnvironment` | `PodmanEnvironment` that boots from a recorded subvolume snapshot. |
| `harbor_plugin.branching:BranchMini` | `AgentENVMini` that resumes an exact mini history prefix (`sprout-agentenv-mini-branch`). |

`agentenv_agent.py` holds the base `AgentENVAgent` (subclasses provide `name()`,
`version()`, `_instruction()`, `_make_spec()`) and `AgentENVOrchestrator`, which
runs Sprout's orchestrator on the sandbox Harbor already started. The agent
classes are written against `SandboxEnvironment` alone -- its `session`,
`backend` dict, pinned `image`, `image_user`/`image_workdir`, `shell_context()`,
`retain_reason`, `checkpoint_mode` and `actor_control` -- so nothing in them
names a backend; `context.metadata["backend"]` records the environment's `type()`.

**Common to both environments** (kwargs `runtime_bin`, required, and
`checkpoint_mode`, `disk_only`|`none`, default `disk_only`). The image's OCI
`User`, `WorkingDir`, `Env` apply to every command: on both backends the
runtime itself runs as root and each command is wrapped (`shell_command`) to
`cd` into the working directory, set the env and, for a non-root `User`, drop
to it with `runuser`; Harbor's own steps (`/logs`, an overlay's `RUN`) run as
root. A `Dockerfile` in the
environment directory whose `FROM` is the task's own `docker_image` is an
**overlay** (Harbor's separate verifier built from `tests/Dockerfile`): the
image is booted and its `COPY` files uploaded and `RUN` steps executed as root
— no registry, no build. `ENV`/`WORKDIR`/`USER` or a second stage are refused
before any sandbox exists. A Dockerfile `FROM` another image is the recipe of
the prebuilt task image and is ignored. A task without a `docker_image` is
accepted only when its Dockerfile is such an overlay (booted from its `FROM`
image); Sprout never builds an image. CPU/memory come from the task. Network is
public or no-network, switchable per phase. Each sandbox writes a
`<session_id>.<type>.json` receipt into the trial directory (`backend`,
`sandbox_id`, the pinned `image` and its `image_config`, `resources`,
`network`, `checkpoint_mode`, `overlay`), which `methods.evaluate` reads back to
tie the graded trial to its sandbox and to give a branch its parent's image
config.

**AgentENVEnvironment.** Kwargs: `server_url` (default `$AENV_SERVER_URL`,
else `http://127.0.0.1:18000`), `api_key_file` (default `$AENV_API_KEY`),
`sandbox_ttl` (default 36000, min 30000). The image is digest-pinned via
`regctl`; disk is at least 64 GiB (`MIN_OCI_DISK_MB`); egress is switched with
`PUT /sandboxes/{id}/network`.

**PodmanEnvironment.** Kwargs: `store` (required: the btrfs directory
`ops/btrfs_store.sh` made), `podman_bin` (default: `$SPROUT_PODMAN_BIN`, else the podman
`ops/setup_podman.sh` installed, else `podman` on PATH). The image is
pinned to the digest podman pulled (`podman image inspect`, pulling first when
absent; no regctl) and its OCI config read from the same record; the session's
`podman` backend extracts the rootfs once and gives the container a writable
snapshot of it, shaped by the task's `cpus`/`memory_mb` (no disk: a subvolume
has none to size). The container runs the runtime as its root -- this user,
under rootless podman -- whatever the image's `User`; the user reaches the
agent's commands through `shell_context`, exactly as on the microVM. A
no-network baseline starts the container disconnected; Harbor's phases toggle a
running one through `SandboxSession.set_network` (`podman network
connect|disconnect` of the sandbox's own network, ~0.15 s) while the runtime's
Unix socket keeps answering. Every sandbox, a verifier's included, has a bridge
network of its own without podman's DNS server: on one bridge shared by all,
each container that came or went stalled the others' packets for seconds.
Snapshots are disk-only, so `final_memory_snapshot` is refused with this
backend; a snapshot is refused (and the turn recorded as not captured) when
the container has made a btrfs subvolume of its own inside its rootfs, which a
snapshot would show empty. The receipt also carries the `store`, and a branch
must run on the store that holds its parent's snapshots. Everything is
rootless.

**AgentENVMini.** Kwargs: `inference_endpoint`, `api_key_env` (name of a set env
var), `actor_timeout_s`, `max_output_tokens`, `max_turns`, `reasoning_effort`.
Prompt = instruction + `mini_tool_primer(workdir)`; a
`ShellContext` interceptor applies the task's cwd, env and user to each call.

## Checkpoints on the live tool path

The agent's calls go through the orchestrator's `ToolAccess` on the trial's
sandbox. Each closed model turn is snapshotted when the next turn starts and
paired with its turn id and the native session in `agent/trajectory.jsonl`.
Per-turn snapshots are disk-only (a restore cold-boots: background processes,
tmpfs and RAM are lost); `none` turns them off (`checkpoint_enabled=False`).

With checkpoints on, a closed turn without a snapshot fails the trial. The
rollout always ends with one more **disk** snapshot, recorded as
`final_snapshot_id` in `agent/snapshot.json` beside `execution.json` and ATIF
`trajectory.json`. If that snapshot fails the VM is **retained**, not
destroyed (`<session_id>.retained-sandbox.json`): it is the only copy of the
rollout's result. `stop()` then releases the sandbox from the session's pool
(`forget`) before closing the pool, because closing a pool destroys every
sandbox it still holds; the same for Harbor's own `stop(delete=False)`. An
actor timeout is a gradable rollout; Harbor's verifier still runs.

`AgentENVMini(final_memory_snapshot=True)` adds a **memory** snapshot after the
disk one (`final_memory_snapshot_id`, journal event `snapshot.memory`). It is
for graders that run inside the agent's VM and need the task's processes alive
on a re-grade; it is never a branch point. `methods.evaluate
--final-memory-snapshot on-failure` takes it for every attempt and releases it
once the attempt grades resolved, appending `snapshot.released` to the journal.

## Branching

`branching.read_branch_context(path)` validates a branch-context JSON up front:
`branch_guidance` (`assistant-turn`|`none`|`user-hint`, the last with a
non-empty `hint`), a snapshot id (AgentENV's UUID or a podman subvolume name;
never a path), `checkpoint_mode` (`disk_only`), a digest-pinned `base_image`
with its `image_config`, `step` >= 1, `native_session_id`, `conversation_cut`,
and a `parent_journal` whose sha256 still matches. `SnapshotRestore` is the
mixin that boots from the snapshot, refuses `force_build`, skips the task-dir
upload and writes `restored-state.json`; `SnapshotEnvironment` and
`PodmanSnapshotEnvironment` are that mixin on each backend's environment class,
so the session creates the branch's sandbox from the snapshot id the way it
would from an image (a podman branch is shaped again: a subvolume carries none).
`BranchMini` re-checks step/snapshot/session against
`available_branch_points(parent)`, loads the exact native prefix and injects a
validated assistant turn, a neutral `resume_without_hint`, or the hint as a
user message (`render_branch_note`, which also asks for a commit when the task
collects its answer from the repository).

## Verifier separation

With `[verifier] environment_mode = "separate"`, Harbor creates the verifier
environment itself from the task's `tests/` context, using the trial's same
environment `import_path` and kwargs, and hands it only the
`[[verifier.collect]]` artifacts. The actor sandbox stays alive for collection;
Harbor tears both down (`delete: true` in `src/sprout/methods/evaluate.py`). In a branch
trial that import path is `SnapshotEnvironment` (or `PodmanSnapshotEnvironment`);
it tells the verifier apart by the directory Harbor hands over (the task's
`tests/`, not its `environment/`) and boots the pristine task image plus the
replayed `tests/Dockerfile`, never the branch snapshot.

## Admission limits

Checked at environment construction, before any sandbox; no Docker fallback.
Rejected: GPU/TPU/Windows (Harbor's own capability checks); compose files; task
MCP servers; network allowlists; host mounts outside
`/logs/{agent,verifier,artifacts,user-agent}`; tasks without a `docker_image`
whose Dockerfile a sandbox cannot replay; on AgentENV, disk not a multiple of
1024 MiB (smaller than 64 GiB is raised); on podman, a store directory that
does not exist (one that is not on btrfs mounted with `user_subvol_rm_allowed`
is refused when the session builds its pool).

## Driver

`python -m sprout.methods.evaluate` runs one Harbor task as a parent trial and then a
method: `sprout` (analyst/reviewer rounds of branch trials; `--rounds`,
`--branches`, `--branch-guidance`), `bpo` or `shepherd` (point-only branches
from a recorded `--parent-output`), or `baseline`. Required: `--task-dir
--dataset-digest --preparation-manifest --output --model --model-endpoint
--model-key-env --runtime-bin`. `--backend {podman,microvm}` (default
`podman`) selects the environment class for every trial of the evaluation --
`podman` takes `--store <btrfs dir>` (default `$SPROUT_PODMAN_STORE`, else
`~/sprout-btrfs`) and leaves `--final-memory-snapshot` at `never` -- and a branch
method's `--parent-output` must have run on the same
backend (and store): its snapshots are the branch points. The same runner
serves DeepSWE, TerminalBench 2.1 and SWE-bench Pro v2 HARD-51;
`examples/qwen/run.py` shows the invocation. `python -m sprout.methods.gate`
runs Harbor's `oracle` and `nop` agents on this environment to prove the
transport grades 1 and 0, on either backend. Run Store jobs take their backend
from the worker profile, podman unless it names microvm (`docs/RUNSTORE.md`).

## Validate without model calls
```bash
python3 -m pytest tests/harbor_plugin -q
```

Every file there `pytest.importorskip("harbor")`: without Harbor they skip, not
fail. They use fakes and start no sandboxes (`test_podman_env.py` drives
`PodmanEnvironment` with a fake session). `tests/methods/test_evaluate.py`
checks the `TrialConfig` wiring against
`tests/harbor_plugin/agentenv_fixtures/shell-task` on both backends.
`tests/harbor_plugin/test_podman_live.py` (marked `slow`; needs podman --
`$SPROUT_PODMAN_BIN`, else the installed one, else PATH -- and a btrfs store at
`SPROUT_PODMAN_STORE`, default `~/sprout-btrfs`) runs the gate
on that fixture through real podman and a btrfs store and asserts the trials
leave no container or subvolume behind, that a retained stop leaves the
container running, and that a `USER nobody` image gets Harbor's root steps as
root and the agent's commands as `nobody`.
