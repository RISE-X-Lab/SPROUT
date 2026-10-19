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

Sprout lets a coding agent go back to any turn of a rollout and continue from there, instead of
starting over. At each turn, Sprout saves a checkpoint: a disk snapshot of the agent's sandbox,
paired with the agent's history up to that turn. When a rollout fails, SPROUT, the method of the
paper, reviews it, picks the checkpoints before the mistake and branches from them. Benchmark
evaluation and RL training run as the same jobs, which workers claim from the Run Store.

<p align="center">
  <img src="docs/figures/overview.svg" width="1000" alt="An overview of Sprout. Across the top are the task sets of evaluation and RL, each with its verifier: DeepSWE, TerminalBench 2.1, SWE-bench Pro v2 HARD-51 and SWE-rebench V2. Below, five stages go from left to right. A coding task with its verifier goes to the search, where SPROUT's analyst and reviewer plan attempts and branches. The search runs in the evaluation or in the RL driver. It queues rollout, grade and review jobs in the Run Store, a job queue in PostgreSQL. A worker claims a rollout job and checkpoints every turn. A grade job runs the task's verifier in the kept sandbox or in a new one, with pass or fail, and the result goes back to the search for the next attempt. Inside each worker, a claim loop runs one child process per job. mini-swe-agent reaches the model endpoint through the model gateway, and reaches the sandbox through ToolAccess and the Sprout runtime, on podman with btrfs or on Firecracker with AgentENV. The checkpoints and the journal pair each state with the trace. The rollouts and their rewards go through the RL driver to Miles. At the bottom, a failed rollout of a task saves its progress at checkpoints C2 and C4. New branches grow from them, and two of the branches pass. Each checkpoint pairs the agent context, the messages and tool results, with the sandbox's disk at a turn boundary."/>
</p>

## Quick start

You need a Linux host with Python 3.12 and Go 1.25.

### 1. Set up the host

```bash
git clone https://github.com/dreamyang-liu/sprout.git sprout && cd sprout
ops/setup_podman.sh
source .venv/bin/activate
python -m sprout.worker doctor
```

`ops/setup_podman.sh` installs these items, and skips each one that is already there:

- rootless podman 6, under `~/.local/opt/podman`
- a btrfs store for the sandboxes, at `~/sprout-btrfs`
- `.venv`, with the `mini`, `harbor` and `dev` extras
- the runtime binary, `runtime/sprout-runtime`

The script uses sudo only for the packages, the btrfs store and the cgroup and user id settings.
`doctor` prints one line for each check of the host. If a check fails, it exits 1.

### 2. Run one agent

```bash
export OPENAI_BASE_URL=<url> OPENAI_API_KEY=<key>
python -m sprout.worker run --model <model> --runtime-bin runtime/sprout-runtime \
    --sandbox-image docker.io/library/python:3.11-slim --cwd /tmp --journal runs/first.jsonl \
    "write fizzbuzz.py, run it, and check that 15 prints FizzBuzz"
python -m ops.report snapshots runs/first.jsonl
python -m ops.report export runs/first.jsonl -o runs/first.md
```

The endpoint can be any OpenAI-compatible Chat Completions API. The run prints its outcome, and
the journal `runs/first.jsonl` records each turn. `snapshots` lists the snapshot of each turn,
and `export` writes the trajectory as Markdown. If your provider does not serve plain Chat
Completions, put it behind the bundled
[LiteLLM proxy](docs/MODEL_ENDPOINTS.md#providers-without-such-an-endpoint-the-litellm-proxy).

### 3. Start the Run Store and a worker

Evaluation and RL submit their rollouts, grades and reviews as jobs to the Run Store. A worker
on the same host claims and runs them.

```bash
pip install -e ".[runstore]"
python -m sprout.services init
# In ~/.config/sprout/runstore.json, set each "store" to the absolute path of your btrfs store.
export SPROUT_MODEL_API_KEY=<key>
python -m sprout.services up runstore-api runstore-worker
eval "$(python -m sprout.services env)"
```

`init` writes the configs and the secrets under `~/.config/sprout/`, and keeps each file that
already exists. The worker passes `SPROUT_MODEL_API_KEY` to its jobs. The last command exports
the Run Store token into the shell ([Services](docs/SERVICES.md)).

### 4. Evaluate a task with SPROUT

The tasks are the Harbor task directories of these benchmarks. Clone them as the
[experiments guide](docs/EXPERIMENTS.md) says.

- [DeepSWE](https://github.com/datacurve-ai/deep-swe)
- [TerminalBench 2.1](https://github.com/harbor-framework/terminal-bench-2-1)
- [SWE-bench Pro v2 HARD-51](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2)

```bash
python -m sprout.eval.gate --tasks-dir ~/datasets/deep-swe/tasks --output runs/gate --workers 8
python -m sprout.eval.evaluate --task-dir ~/datasets/deep-swe/tasks/<task> \
    --dataset-digest sha256:... --preparation-manifest runs/prep.json --output runs/<task>-sprout \
    --method sprout --model <model> --model-endpoint <url> --model-key-env SPROUT_MODEL_API_KEY \
    --rounds 2 --branches 4 3
python -m ops.report task runs/<task>-sprout
```

The commands do these steps:

1. The gate checks that the oracle of each task scores 1 and that doing nothing scores 0.
2. The evaluation runs the agent on the task. If the attempt fails, the review picks
   checkpoints, and the agent branches from them. `--rounds 2 --branches 4 3` gives two rounds,
   with up to 4 branches and then up to 3.
3. `ops.report task` prints the tree of the attempts and their rewards.

`--dataset-digest` and `--preparation-manifest` pin the task set. `python -m examples.qwen.run`
writes both, and runs a whole benchmark. The evaluation refuses a checkout with uncommitted
changes, because its summary records the commit. [Evaluation](docs/EVALUATION.md) gives every
option.

### 5. Train with RL

The [RL driver](docs/RL_DRIVER.md) serves [Miles](https://github.com/radixark/miles) rollout
groups from the same jobs. Start it with `python -m sprout.services up rl-driver`, next to the
worker of step 3. The worker grades
[SWE-rebench V2](https://github.com/SWE-rebench/SWE-rebench-V2) tasks with a port of its official
harness.

## Learn more

| Guide | What it covers |
| --- | --- |
| [Architecture](docs/ARCHITECTURE.md) | Checkpoints, SPROUT branching, the processes around the Run Store |
| [Experiments](docs/EXPERIMENTS.md) | Datasets, task selection, smoke tests, resuming, the comparison export |
| [Experiment protocol](examples/qwen/PROTOCOL.md) | The fixed settings, the rollout budget and the metrics |
| [Evaluation](docs/EVALUATION.md) | The options and the output of an evaluation and of the task gate |
| [Model endpoints](docs/MODEL_ENDPOINTS.md) | What an endpoint must provide, and the LiteLLM proxy for other providers |
| [Services](docs/SERVICES.md) | PostgreSQL, the LiteLLM proxy, the Run Store, its worker and the RL driver |
| [Run Store](docs/RUNSTORE.md) | The job queue, the worker, the graders and the training export |
| [RL driver](docs/RL_DRIVER.md) | The Miles `/rollout-groups` adapter over the Run Store |
| [Harbor plugin](docs/HARBOR_PLUGIN.md) | Harbor's own trials on a Sprout sandbox, the reference for the Harbor grade |
| [Runtime](runtime/README.md) | The tools inside each sandbox |
| [Gate](gate/README.md) | The checks that every commit must pass |
| [Decisions](docs/decisions/README.md) | The architecture decisions that one revert cannot undo |

## Acknowledgments

[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) is the agent, run unmodified
over Sprout's tool transport. Sprout reads tasks in the format of
[Harbor](https://github.com/harbor-framework/harbor), and runs each task's verifier as Harbor 0.22
does. The microVMs run on our
[customized Firecracker](https://github.com/dreamyang-liu/firecracker), served by
[AgentENV](https://github.com/dreamyang-liu/AgentENV); the container backend is stock
rootless [podman](https://podman.io) over [btrfs](https://btrfs.readthedocs.io) snapshots.

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
