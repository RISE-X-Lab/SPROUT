<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/figures/logo-dark.svg">
    <img src="docs/figures/logo.svg" width="420" alt="sprout: branch an agent rollout from any step"/>
  </picture>
</p>

<p align="center">
  <b>Efficient exploration for coding agents.</b><br>
  Save progress at every turn. Branch from any checkpoint instead of starting over.
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2610.14405"><img src="docs/figures/btn-paper.svg" height="40" alt="Paper on arXiv"></a>&nbsp;
  <a href="https://rise-x-lab.github.io/SPROUT/"><img src="docs/figures/btn-project.svg" height="40" alt="Project page"></a>&nbsp;
  <a href="#citation"><img src="docs/figures/btn-citation.svg" height="40" alt="Citation"></a>
</p>

<p align="center">
  <a href="#license"><img src="https://img.shields.io/badge/License-MIT-4BB85E.svg?style=flat-square&amp;labelColor=1F4A33" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.12%2B-4BB85E.svg?style=flat-square&amp;labelColor=1F4A33" alt="Python 3.12 or later">
  <img src="https://img.shields.io/badge/go-1.25%2B-4BB85E.svg?style=flat-square&amp;labelColor=1F4A33" alt="Go 1.25 or later">
  <img src="https://img.shields.io/badge/sandbox-podman%20%7C%20Firecracker-4BB85E.svg?style=flat-square&amp;labelColor=1F4A33" alt="Sandbox: rootless podman or Firecracker microVM">
  <a href="https://github.com/SWE-agent/mini-swe-agent"><img src="https://img.shields.io/badge/agent-mini--swe--agent-4BB85E.svg?style=flat-square&amp;labelColor=1F4A33" alt="Agent: mini-swe-agent"></a>
</p>

<p align="center">
  <img src="docs/figures/architecture.svg" width="1000" alt="Sprout architecture. A task and its verifier go to a strategy (SPROUT's analyst and reviewer), which queues rollout jobs in the Run Store; a worker claims a job and checkpoints every turn; a verifier grades the finished rollout, and its feedback informs the next attempt. Inside each worker, mini-swe-agent reaches the model through a gateway and the sandbox through ToolAccess and the Sprout runtime, on podman + btrfs or Firecracker / AgentENV. Below, branches grow from saved checkpoints C2 and C4 and pass where the original continuation failed; each checkpoint pairs the agent's context with an environment snapshot."/>
</p>
<p align="center">
  <sub>A checkpoint pairs the agent's context with a snapshot of its sandbox. When a rollout fails,
  SPROUT's analyst and reviewer pick the checkpoints before the mistake and branch from there.</sub>
</p>

## Quick start

On a Linux host with Python 3.12 and Go 1.25:

```bash
git clone https://github.com/dreamyang-liu/sprout.git sprout && cd sprout
ops/setup_podman.sh                # podman 6 for this user, a btrfs store, .venv, the runtime binary
source .venv/bin/activate
export OPENAI_BASE_URL=<url> OPENAI_API_KEY=<key>   # any OpenAI-compatible Chat Completions endpoint
python -m sprout.harness run --model <model> --runtime-bin runtime/sprout-runtime \
    --sandbox-image docker.io/library/python:3.11-slim --cwd /tmp --journal runs/first.jsonl \
    "write fizzbuzz.py, run it, and check that 15 prints FizzBuzz"
python -m sprout.inspect snapshots runs/first.jsonl            # every snapshot the run took
python -m sprout.inspect export runs/first.jsonl -o first.md   # the trajectory as markdown
```

`ops/setup_podman.sh` installs podman 6 under `~/.local/opt/podman`, apart from any
distribution podman, and uses sudo only for packages, the subordinate id range, cgroup
delegation and the btrfs store; everything after that is rootless. Each step is skipped when
already done, and `--check` only reports. Every snapshot id listed there is a sandbox image: a second run
with `--sandbox-image <snapshot id>` and the conversation up to it as `extra.native_prefix`
is a branch. A provider that is not plain Chat Completions (a native messages API, a provider
SDK) goes behind the bundled LiteLLM proxy, `python -m sprout.services init && python -m
sprout.services up proxy`; see [model endpoints](docs/MODEL_ENDPOINTS.md).

To retry a benchmark task with SPROUT, gate the task set first (its oracle must score 1 and
doing nothing 0), then run the method on one task.

```bash
python -m sprout.methods.gate --tasks-dir ~/datasets/deep-swe/tasks \
    --output runs/gate --runtime-bin runtime/sprout-runtime --workers 8
python -m sprout.methods.evaluate --task-dir ~/datasets/deep-swe/tasks/<task> \
    --dataset-digest sha256:... --preparation-manifest runs/prep.json \
    --output runs/<task>-sprout --method sprout \
    --model <model> --model-endpoint <url> --model-key-env OPENAI_API_KEY \
    --runtime-bin runtime/sprout-runtime --rounds 2 --branches 4 3
python -m sprout.inspect task runs/<task>-sprout          # the fork tree
```

`--rounds 2 --branches 4 3` is two review rounds, up to 4 branches and then 3.
`--model-key-env` names the variable that holds the key. `--dataset-digest` and
`--preparation-manifest` pin the task set; `python -m examples.qwen.run` computes both and
runs the whole benchmark × method grid with one shared baseline per task (`--plan` prints
the commands without calling a model). Tasks are the benchmarks' own task directories,
cloned as in the [experiments guide](docs/EXPERIMENTS.md): every attempt on
[DeepSWE](https://github.com/datacurve-ai/deep-swe),
[TerminalBench 2.1](https://github.com/harbor-framework/terminal-bench-2-1) and
[SWE-bench Pro v2 HARD-51](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2) is a
[Harbor](https://github.com/harbor-framework/harbor) `Trial`, graded afterwards by the
task's own verifier, in a separate sandbox for the patch-graded datasets. Leaked sandboxes,
on either backend:
`python -m sprout.harness reap`.

## Checkpoint every turn

An orchestrator owns each run. It starts the sandbox, hands
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) a single `bash` tool, and
snapshots the sandbox's disk at the end of every model turn that ran a command. Each
snapshot is paired with the agent's conversation at that point, and the pair is a
**branch point**: start a run from the snapshot with that conversation and the agent picks
up there, with the same past and a different future.

Every command counts as a change. The runtime's shell is stateless, and a command's effect
is not in its text (`ls` reads, `ls > out` writes), so no turn that ran one goes uncaptured.

- **podman** (the default). A rootless container whose root filesystem is a btrfs
  subvolume; a snapshot is a read-only btrfs snapshot of it, taken while the container
  runs. Stock podman 6 (the setup script installs it), no daemon, and a network of its own per
  sandbox, so egress can be
  switched while it runs.
- **microvm** (`--backend microvm`). A Firecracker microVM served by
  [AgentENV](https://github.com/dreamyang-liu/AgentENV); a snapshot is one layer on the VM
  disk's chain, compacted on the server, and a memory snapshot is available where a grader
  needs the VM's processes alive.

A branch runs on its parent's backend, and on podman in its parent's store: the parent's
snapshots are its branch points.

Per-turn snapshots are disk-only, which keeps them cheap. Over a 113-task DeepSWE run on
podman, with up to 80 sandboxes at once, 22,077 per-turn snapshots took a median of 153 ms
and checkpointing was 2.4% of rollout wall time. On one long attempt, disk-only snapshots
came to 37 MB per episode; with memory they came to 2.2–4.7 GB.
See [snapshot cost](docs/SNAPSHOT_COST.md).

## Branch from where it went wrong

A failed attempt rarely goes wrong at its first step, and the agent usually believes it
succeeded. Sampling again from the start repeats every step it got right. Sprout keeps the
steps before the mistake and branches after them. Four methods spend the same rollout
budget:

| Method | Where it branches |
|---|---|
| `baseline` | nowhere: independent rollouts |
| `sprout` | an analyst reads the failed trajectory and the verdict; a reviewer picks branch points and writes each branch's first turn |
| `bpo` | the checkpoints where the policy's first-token entropy is highest |
| `shepherd` | one restart point chosen by a meta-agent; siblings sample from it |

In SPROUT, an analyst reads each failed attempt on its own: the trajectory, the patch and
the grading verdict. A reviewer reads every analysis, picks branch points, each on its own
attempt and step, and writes each branch's first assistant turn: probe, then fix, then
verify. That turn runs through the normal tool path, so the branch's history and its disk
agree, and the actor continues from it. Rounds repeat until a branch resolves the task or
`--rounds` is spent. The analyst and reviewer can run on a stronger model than the actor
(`--review-model`, `--review-endpoint`), and every analyst, reviewer and Shepherd prompt
lives in [prompts.yaml](src/sprout/methods/prompts.yaml), whose sha256 each run records.

The same branch points feed RL. [Run Store](docs/RUNSTORE.md) queues rollout, grade and
review jobs on PostgreSQL and grades [SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2)
with a port of its official harness. The [RL driver](docs/RL_DRIVER.md) serves
[Miles](https://github.com/radixark/miles) rollout groups from it and branches a failed
rollout at the reviewer's points. Training export marks every message no policy sampled:
the restored prefix and every inserted turn.

## Runtime toolset support

`runtime/` is the Go binary inside every sandbox. It serves these tools over JSON-RPC
(`tools/list`, `tools/call`). `runtime/schema/tools.json` is their checked-in declaration, and
`ToolAccess` refuses any tool or argument it does not list. mini-swe-agent's model sees one
`bash` tool, which runs as `shell`.

| Tool | Arguments (first ones required) | Does | Counts as a change |
| --- | --- | --- | --- |
| `shell` | `command`; `working_dir`, `timeout`, `max_output_bytes` | runs `sh -c`, returns exit code, stdout, stderr | always |
| `text_editor` | `command` (`view`, `str_replace`, `insert`, `write`), `path`; `view_range`, `old_str`, `new_str`, `insert_line`, `insert_text`, `file_text`, `max_output_bytes` | views or edits one file | except `view` |
| `grep_files` | `pattern`; `path`, `include`, `limit`, `max_output_bytes` | searches files with the ripgrep embedded in the runtime | no |
| `web_fetch` | `url`; `format` (`markdown`, `text`, `html`), `headers`, `timeout`, `max_output_bytes` | fetches a page | no |
| `web_search` | `query`; `backend`, `max_results` | searches several engines through ddgs | no |

"Counts as a change" decides whether the turn gets a snapshot of its own (see above). Output
over `max_output_bytes` keeps its first 40% and last 60% and says so. `text_editor` and
`grep_files` work as the runtime's process, so relative paths do not follow a task's working
directory. The ripgrep release is pinned in `runtime/tools/ripgrep/SHA256SUMS` but not kept
in git; `go generate` downloads it before a build:

```bash
cd runtime && go generate ./... && go build -o sprout-runtime .
```

## Layout

```
.
├── src/sprout/harness/               # the agent runtime: one run, end to end
├── src/sprout/harness_agents/        # mini-swe-agent, the one agent
├── src/sprout/harness_checkpoint/    # a snapshot per turn, and rollback to one
├── src/sprout/harness_core/          # the journal, its events, run control, the slot contract
├── src/sprout/harness_gateway/       # the model gateway: routing, budget, the wire-level tap
├── src/sprout/harness_orchestrator/  # one run start to finish, what it holds, reaping
├── src/sprout/harness_sandbox/       # sandbox sessions, backends, image templates
├── src/sprout/harness_tools/         # ToolAccess, the runtime's schema, interceptors
├── src/sprout/harbor_plugin/         # Sprout as a Harbor environment and agent
├── src/sprout/methods/               # evaluate (one task, one method), branching, prompts, gate
├── src/sprout/inspect/               # read-only CLI over run output
├── src/sprout/runstore/              # PostgreSQL job queue, worker, SWE-rebench grader, training export
├── src/sprout/rl_driver/             # the RL trainer's (Miles) rollout adapter over the Run Store
├── src/sprout/services/              # the host's long-running pieces: db, LiteLLM proxy, Run Store, RL driver
├── src/sprout_sandbox/               # the sandbox clients: AgentENV microVMs and rootless podman
├── runtime/                          # the Go binary inside every sandbox: shell, text_editor, grep, web, over JSON-RPC
└── tests/  docs/  examples/  ops/
```

Imports point down only, and `tests/harness/test_layering.py` enforces it: the harness knows
nothing about benchmarks, what counts as an answer lives in the task's verifier, and what to
do when the answer is wrong lives in `methods`.

## Learn more

| Guide | What it covers |
| --- | --- |
| [AGENTS.md](AGENTS.md) | The operational reference: layout and layering, what a run produces, branching in detail, cleanup, the pitfalls |
| [Experiments](docs/EXPERIMENTS.md) | The benchmark × method grid: datasets, task selection, smoke tests, resuming, the comparison export |
| [Harbor plugin](docs/HARBOR_PLUGIN.md) | Sprout inside a Harbor trial: checkpoints on the live tool path, branching, verifier separation |
| [Model endpoints](docs/MODEL_ENDPOINTS.md) | What an endpoint must provide, the LiteLLM proxy, schema-constrained review |
| [Services](docs/SERVICES.md) | PostgreSQL, the LiteLLM proxy, Run Store and the RL driver as one host's services |
| [Run Store](docs/RUNSTORE.md) | The job queue, the worker, the SWE-rebench grader and training export |
| [RL driver](docs/RL_DRIVER.md) | The Miles `/rollout-groups` adapter over Run Store |
| [Leak judge](docs/LEAK_JUDGE.md) | Whether a reviewer-written turn copies hidden-test content |
| [Snapshot cost](docs/SNAPSHOT_COST.md) | Per-turn snapshot latency on each backend, idle and under load |

## Acknowledgments

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is the agent, run unmodified
over Sprout's tool transport. [Harbor](https://github.com/harbor-framework/harbor) loads the
tasks and grades each trial with the task's own verifier. The microVMs run on our
[customized Firecracker](https://github.com/dreamyang-liu/firecracker), served by
[AgentENV](https://github.com/dreamyang-liu/AgentENV); the container backend is stock
rootless [podman](https://podman.io) over [btrfs](https://btrfs.readthedocs.io) snapshots.
BPO and Shepherd follow the branch-point selection of their papers.

## Citation

```bibtex
@misc{liu2026sproutscalingagenticreinforcement,
  title={SPROUT: Scaling Agentic Reinforcement Learning for Long-Horizon Software Engineering},
  author={Mengyang Liu and Weiming Wang and Haixin Nan and Yu Wu and Peize Li and Shihong Huang and Xuning He and Xue Jiang and Yihong Dong},
  year={2026},
  eprint={2610.14405},
  archivePrefix={arXiv},
  primaryClass={cs.SE},
  url={https://arxiv.org/abs/2610.14405},
}
```

## License

[MIT](LICENSE), for everything in this repository including the runtime and the Python SDK.
