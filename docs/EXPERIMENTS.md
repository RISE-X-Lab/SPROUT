# The Qwen experiments

Select one of 12 experiments with `--dataset` and `--method`: three datasets and
four methods. Run all commands in Bash from the Sprout repository root unless stated
otherwise.

## 1. Install the runtime and dependencies

Prerequisites: Python 3.12+, Go 1.25+, rootless podman 6 on a btrfs store (the default backend; `ops/setup_podman.sh` installs the podman and sets the host up) or a working microVM snapshot service (the customized Firecracker stack, deployed separately; Section 8), and network access to the task image registries. Clone the complete Sprout repository because the experiment runner imports other Sprout modules.

```bash
git clone https://github.com/dreamyang-liu/sprout.git sprout
cd sprout
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[mini,harbor]"
export LITELLM_LOCAL_MODEL_COST_MAP=True
(cd runtime && go generate ./... && go build -o sprout-runtime .)   # generate fetches the ripgrep it embeds
```

Configure your model endpoint and credentials. Replace the placeholders below with your own settings. Keep credentials outside Git.

```bash
eval "$(python -m sprout.services env)"   # SPROUT_PODMAN_BIN and SPROUT_PODMAN_STORE (the podman 6 and store the
                                          # setup script verified), plus the proxy and Run Store variables
export QWEN_BASE_URL=http://your-private-model-endpoint
read -rs -p 'Qwen API key: ' QWEN_API_KEY; echo
export QWEN_API_KEY
# --backend microvm only (Section 8)
export AENV_SERVER_URL=http://127.0.0.1:18000
export AENV_KEY_FILE=/path/to/aenv-key
```

## 2. Download the official datasets and select tasks

Download the datasets and their environment images separately. This repository includes experiment code and the HARD-51 selection list. Each task directory must retain its `task.toml`, instruction, environment configuration, and official verifier scripts. The runner requires these task files in addition to any tabular dataset metadata.

| Dataset | `--dataset` | Official sources and preparation requirements |
|---|---|---|
| SWE-bench-Pro v2 HARD-51 | `swebenchpro-v2-hard` | [Official Scale benchmark page](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2); [official V2 dataset and setup instructions](https://github.com/scaleapi/SWE-bench_Pro-os/tree/main/v2); [official HARD-51 task IDs](https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/v2/hard51_ids.txt). Clone the official repository and use its `v2/tasks` directory with the included `examples/qwen/swebenchpro-v2-hard/selection.json`. |
| TerminalBench 2.1 | `terminalbench21` | [Official dataset repository](https://github.com/harbor-framework/terminal-bench-2-1); [official Harbor Hub dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1/latest). Use the repository's `tasks` directory and an explicit task selection file. |
| DeepSWE | `deepswe` | [Official benchmark website](https://deepswe.datacurve.ai/); [official dataset repository](https://github.com/datacurve-ai/deep-swe). Use the repository's `tasks` directory and an explicit task selection file. |

The Scale benchmark page links to the official GitHub dataset. That repository publishes V2 tasks under `v2/tasks` and defines HARD-51 in `v2/hard51_ids.txt`. The same dataset is available on [Hugging Face](https://huggingface.co/datasets/ScaleAI/SWE-bench_Pro), with the `hard` configuration selecting HARD-51. For this runner, obtain the task directories and verifiers from the official GitHub repository. The included 51 task IDs were checked against the official list on September 27, 2026.

For a fresh machine, download the task repositories into a directory you own:

```bash
export DATASETS="$HOME/datasets"
mkdir -p "$DATASETS"
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git "$DATASETS/SWE-bench_Pro-os"
git clone https://github.com/harbor-framework/terminal-bench-2-1.git "$DATASETS/terminal-bench-2-1"
git clone https://github.com/datacurve-ai/deep-swe.git "$DATASETS/deep-swe"
mkdir -p "$DATASETS/cohorts"
```

Record the dataset commits used for your evaluation and keep task files unchanged across methods. Follow the upstream release instructions for environment images and verification requirements. Revalidate compatibility when changing dataset versions.

A selection file is a JSON array of task directory names. For example, `["largest-eigenval"]` selects one TerminalBench task. Create `terminalbench21.json` and `deepswe.json` in the `cohorts` directory with the tasks you intend to evaluate. Use the same selection for all four methods. The HARD-51 list is already included in Sprout.

Set the following variables and Bash arrays once per shell session. Adjust paths for your own machine, and keep outputs outside the Sprout checkout:

```bash
export PRO_TASKS="$DATASETS/SWE-bench_Pro-os/v2/tasks"
export TB_TASKS="$DATASETS/terminal-bench-2-1/tasks"
export DEEP_TASKS="$DATASETS/deep-swe/tasks"
export TB_SELECTION="$DATASETS/cohorts/terminalbench21.json"
export DEEP_SELECTION="$DATASETS/cohorts/deepswe.json"
export RUNS="$HOME/sprout-runs/qwen"

COMMON=(--runtime-bin "$PWD/runtime/sprout-runtime" --workers 16)
PRO=(--dataset swebenchpro-v2-hard --tasks-dir "$PRO_TASKS" --selection "$PWD/examples/qwen/swebenchpro-v2-hard/selection.json")
TB=(--dataset terminalbench21 --tasks-dir "$TB_TASKS" --selection "$TB_SELECTION")
DEEP=(--dataset deepswe --tasks-dir "$DEEP_TASKS" --selection "$DEEP_SELECTION")
```

In a new terminal, reactivate the virtual environment and set these environment variables and Bash arrays again.

## 3. Run one of the 12 experiments

For each dataset, finish **baseline** first, then run SPROUT, BPO, and Shepherd. All three branch methods use `--baseline-root` to share the same initial rollout.

Defaults: mini-swe-agent, `qwen3.8-27b`, `high` reasoning effort, assistant-turn branching, up to 300 model turns per rollout, a maximum output-token setting of 64000, a 600-second limit on each agent command (`--command-timeout`, recorded in the manifest; a shared baseline must have used the same value, so a baseline that ran under the earlier 60-second default needs `--command-timeout 60` for its branch methods), and a 400,000-character budget on every analyst and reviewer request (`--prompt-budget-chars`; about 100k tokens -- a reviewer request over it is windowed and clipped in a fixed order and the plan file records what was dropped, a request the endpoint refuses is retried smaller). Baseline runs four independent rollouts regardless of success (`--baseline-rollouts`, recorded in the manifest and matched by a shared baseline; `--baseline-rollouts 1` is pass@1 alone, and `--method sprout` then branches each failed task right after its one rollout). Branch methods skip tasks whose first rollout succeeds. The default total budget is eight attempts, including that shared initial rollout; `--max-rollouts` may be increased up to 128.

For a single wide SPROUT round, pass `--branches N --branch-count-mode fixed
--max-rollouts N+1` (substitute integers, for example `--branches 32
--branch-count-mode fixed --max-rollouts 33`). The runner records the schedule
and passes it through to the evaluator, whose budget limit is the same. To
compare widths 1, 2, 4, 8, 16 and 32, run each width into its own output root
with the same `--baseline-root` and `--baseline-rollouts 1`.

Every analyst, reviewer and Shepherd prompt is in
`resources/review/prompts.yaml`. `--prompts PATH` (passed through to every
evaluator) runs an edited copy; the file's sha256 is recorded in the manifest
(`prompts_sha256`), `summary.json` and each `plan-roundN.json` (`prompts`), so
runs under different prompts are told apart. The reviewer prompt asks for
turns that probe the suspected defect from /tmp or existing tests, then fix,
then re-run the probe and the nearest tests -- the style of the Kimi K3 branch
turns that converted at 0.73 in the fixed 4+3 analysis -- and the validator
rejects a turn that submits (its plan is returned to the reviewer).

The analyst and reviewer may run on another model, or behind another endpoint,
than the actor: `--review-model`, `--review-endpoint`, `--review-key-env` (the
variable holding that endpoint's bearer key) and `--review-effort`, each
defaulting to the actor's (`--model`, `--model-endpoint`, `--model-key-env`,
`high`), and `--review-output-tokens` (default `--max-output-tokens`) is their
output budget apart from the actor's. The four reach the sprout stages alone.
The manifest records them under `review` -- `model`, `own_endpoint` (whether
they had an endpoint of their own; never the URL or the key), `effort`,
`output_tokens`, and `leak_judge` for a judged run -- and none is a
shared-baseline setting, so a branch method may review with another model than
its baseline ran on. `--sprout-rounds R --sprout-branches W` runs exactly R
rounds of W branches each (the budget becomes 1 + R*W, at most 17 rollouts; a
wider schedule is spelled out with `--branches`), and `--leak-judge` checks
every turn the reviewer writes against the task's hidden tests before it is
taken (`docs/LEAK_JUDGE.md`).

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

Each command defaults to 16 task workers. Concurrent commands add to the total worker count. Reduce `--workers` in `COMMON` to limit resource usage.

## 4. Optional: run all four methods for one dataset

Choose either the separate runs above or this combined mode. With `all`, each task runs its four baseline rollouts and then the three branch methods in sequence.

```bash
python -m examples.qwen.run "${PRO[@]}" "${COMMON[@]}" --method all --output "$RUNS/swebenchpro-v2-hard-all"
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" --method all --output "$RUNS/terminalbench21-all"
python -m examples.qwen.run "${DEEP[@]}" "${COMMON[@]}" --method all --output "$RUNS/deepswe-all"
```

## 5. Preflight checks and small smoke tests

Append `--plan` to a command in Section 3 or 4 to check task files and print the execution plan without model calls or sandbox creation. Network access, images, and official verifiers require an actual smoke test.

These commands each select one task and exercise all four methods with a small budget:

```bash
python -m examples.qwen.run --dataset terminalbench21 --method all --tasks-dir "$TB_TASKS" --task largest-eigenval "${COMMON[@]}" --workers 1 --max-turns 2 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/terminalbench21-smoke"
python -m examples.qwen.run --dataset deepswe --method all --tasks-dir "$DEEP_TASKS" --task superjson-error-stack-serialization "${COMMON[@]}" --workers 1 --max-turns 3 --max-output-tokens 4096 --max-rollouts 2 --output "$RUNS/deepswe-smoke"
```

The smoke configuration still runs four baseline rollouts; each branch method has a total attempt budget of two. Smoke results validate integration and are not full benchmark scores.

## 6. Export the five-method, four-metric comparison

After the separate runs from Section 3 finish, select a dataset and combine their results:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m examples.qwen.compare \
  --baseline "$RUNS/$DS-baseline" \
  --sprout "$RUNS/$DS-sprout" \
  --bpo "$RUNS/$DS-bpo" \
  --shepherd "$RUNS/$DS-shepherd" \
  --output "$RUNS/$DS-comparison"
```

This writes `comparison.csv` and `comparison.json`. It refuses roots whose manifests differ on any setting a shared baseline must match (the same set `run.py` enforces at launch: dataset and digest, tasks, model, reasoning effort, turn and output limits, command timeout, backend, store, baseline rollouts). The five rows are pass@1, pass@4 (baseline), BPO, Shepherd, and SPROUT. The four metrics are Resolve Rate, Recovery, Steps, and Tokens (M). `python -m ops.report runs <method root>` prints the `metrics-<method>.json` summary once the runner has written it.

For the combined `all` mode from Section 4, generate a JSON report containing the five-row `comparison` array:

```bash
DS=swebenchpro-v2-hard  # or terminalbench21, deepswe
python -m examples.qwen.report "$RUNS/$DS-all" --method all --save "$RUNS/$DS-all/comparison.json"
```

## 7. Resume runs and retain checkpoints

- Repeating the original command with the same configuration and code commit reuses completed stages: `manifest-<method>.json` records both, `source_commit` included, and is matched whole, so a root written under an older commit is not resumed under a newer one (continue from its baseline in a fresh root with `--baseline-root`, which checks the shared settings alone). Incomplete stages are protected from overwriting: retain their outputs and usage records, then retry in a fresh output directory.
- This entry point requires its own manifests and stage receipts. Outputs from other controllers cannot be passed directly as `--baseline-root`.
- A failed first rollout retains `retained-checkpoints.json`, journals, and exact conversation-prefix references. Recovery also requires the snapshots themselves, in the podman store or the microVM service. Keep both the output directory and that storage.
- The experiment package deletes no disk snapshots. Disk-only snapshots do not preserve RAM, live processes, or tmpfs, so on the microVM backend TerminalBench 2.1 (whose verifier runs inside the agent's sandbox) also takes a memory snapshot at the end of each rollout and releases it once the attempt grades resolved (`--final-memory-snapshot`, default `auto`: `on-failure` there, `never` for patch-graded datasets and on podman, whose snapshots are disk-only).
- BPO and Shepherd use inference-time policy adaptations. See the [experiment protocol](examples/qwen/PROTOCOL.md) for budget, accounting, and recovery details.
- Progress of a running or finished experiment, per method, with each live attempt's current turn: `python -m ops.report runs "$RUNS/deepswe-baseline"` (also works on a root whose stages errored or whose runner was killed: unfinished stages are reported as stale, and a stage receipt with no log beside it, which no runner ever launched, as unlaunched). One task's baseline attempts, analyst/reviewer rounds and branches: `python -m ops.report task "$RUNS/deepswe-sprout" <task> --format mermaid`. Whether a retained parent's snapshots are still in the store: `python -m ops.report snapshots "$RUNS/deepswe-baseline/tasks/<task>/baseline/attempt-1"`.

## 8. The two backends: podman (the default) and the microVM service

The commands above run on this machine's rootless podman 6 (the one
`ops/setup_podman.sh` installs, named by `SPROUT_PODMAN_BIN`; with `btrfs-progs`). Each trial's sandbox is a container whose rootfs is a btrfs
subvolume of the store (`--store`, default `$SPROUT_PODMAN_STORE`, else
`~/sprout-btrfs`); a checkpoint is a read-only btrfs snapshot of it
(milliseconds), and a branch is a container started from one. Root is needed
once, to make the store (`ops/setup_podman.sh` does this and the rest of a
host's setup):

```bash
sudo ops/btrfs_store.sh "$HOME/sprout-btrfs" /var/lib/sprout-btrfs-rootless.img 200G
```

What podman implies: the first trial on each task image extracts the image's
rootfs into the store (`podman pull` plus `podman export | tar`, once per image;
about a minute for a 3 GB DeepSWE image, and every later container from it
starts in under a second; `ops/prepull_images.py` does it ahead of a batch);
snapshots are disk-only, so `--final-memory-snapshot` stays `never` (`auto`
resolves to it) and TerminalBench 2.1 re-grades cannot rely on live processes;
a branch method's `--baseline-root` must have run on podman and on the same
store, since its snapshots are subvolumes there; and the runtime binary has to
run inside the image (a musl image needs the static `CGO_ENABLED=0 go build`).
Everything runs as your user.

On the microVM service, add `--backend microvm --api-key-file "$AENV_KEY_FILE"`
to every command, with `AENV_SERVER_URL` exported. Each sandbox is then a
Firecracker microVM, a checkpoint is an incremental disk layer, and TerminalBench
2.1 takes its memory snapshot (`auto` resolves to `on-failure` there). A shared
baseline and its branch methods run on the same backend:

```bash
VM=(--backend microvm --api-key-file "$AENV_KEY_FILE")
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" "${VM[@]}" --method baseline --output "$RUNS/terminalbench21-baseline"
python -m examples.qwen.run "${TB[@]}" "${COMMON[@]}" "${VM[@]}" --method sprout --baseline-root "$RUNS/terminalbench21-baseline" --output "$RUNS/terminalbench21-sprout"
```

Cleaning up is the same two commands on either backend: `python -m
sprout.worker sweep <journal> --yes` deletes the snapshots a journal names, and
`python -m sprout.worker reap` removes sandboxes a killed run left. Run Store
jobs take their backend from the worker profile, podman unless it names microvm
(`docs/RUNSTORE.md`).
