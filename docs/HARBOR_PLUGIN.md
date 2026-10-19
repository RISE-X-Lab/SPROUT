# Harbor's own trials on a Sprout sandbox

`src/sprout/eval/harbor/` gives the reference rewards that the worker's `harbor` grade must
equal. It is a Harbor environment on a rootless podman sandbox of Sprout's. Harbor runs the
trials of its own `oracle` and `nop` agents on it. A test compares their rewards with the
worker's grades of the same agents. Sprout keeps this adapter and the `harbor==0.22.0`
dependency until the equivalence check of
[ADR-0003](decisions/0003-sprout-runs-harbor-verifier.md) has passed.

Nothing else in Sprout runs an agent or a verifier through Harbor. Evaluations and the task
gate submit Run Store jobs ([evaluation](EVALUATION.md)), and the worker runs each task's
verifier itself. The `harbor` extra in `pyproject.toml` installs Harbor, and
`ops/contracts/harbor.yaml` pins the part of its API that the adapter and the tests call.

Read this document if you change the worker's `harbor` grade, the podman sandbox or the
Harbor pin.

## Harbor's trials are the reference for the harbor grade

The worker's `harbor` grade is correct when it gives the reward that Harbor's own trial gives
for the same agent on the same task. Both sides start their sandboxes through the same
`SandboxSession` class and the same podman pool. So the test compares two ways to run an
agent and a verifier on one sandbox stack. The figure shows the two paths and the comparison.

<p align="center">
  <img src="figures/harbor-plugin-reference.svg" width="900" alt="On the left, a Harbor 0.22.0 Trial runs Harbor's oracle or nop agent and Harbor's verifier. It calls PodmanEnvironment, from sprout.eval.harbor, to start, run commands in and stop the sandbox. PodmanEnvironment creates a SandboxSession and runs commands and network changes through it. On the right, the Sprout worker runs a rollout job of the same agent, then a grade job with the benchmark harbor, from the rollout's snapshot or in the sandbox that the rollout kept. Each job of the worker has a SandboxSession of its own. Each SandboxSession reaches a podman sandbox through the podman pool: a container whose rootfs is a btrfs subvolume, with sprout-runtime inside. The reward of the Harbor trial and the reward of the worker's grade go to the test test_harbor_grade_live.py, which requires them to be equal."/>
</p>
<p align="center"><sub>Harbor's trial and the worker's grade run on the same SandboxSession and podman sandbox, and a test requires their rewards to be equal.</sub></p>

`tests/sprout_sandbox/test_harbor_grade_live.py` does these steps on the fixture task
`tests/fixtures/harbor/shell-task`:

1. Harbor runs the trials of its `oracle` and `nop` agents on `PodmanEnvironment`. The
   helper `harbor_trial` in `tests/sprout_sandbox/harbor_trials.py` makes each trial.
2. The worker runs the same two agents as rollouts, through its attempt entry point.
3. The worker grades each rollout with the `harbor` grade, from the rollout's final snapshot.
4. The worker runs the two agents again, keeps each sandbox, and grades each rollout in the
   sandbox that it kept.
5. The test requires each grade's reward to equal Harbor's reward: 1 for `oracle` and 0 for
   `nop`.

ADR-0003 decides that Sprout deletes the adapter and the `harbor` dependency only after a
larger equivalence check:

- the gate on every task of the three datasets: DeepSWE, TerminalBench 2.1 and HARD-51
- a re-grade of recorded snapshots against the recorded Harbor rewards

Keep the adapter and its tests until that check passes. The live test is the check for one
change, on one fixture task.

## Modules and how Harbor loads them

The package has two modules:

| Import path | What it is |
|---|---|
| `sprout.eval.harbor.environment:SandboxEnvironment` | Harbor's `BaseEnvironment` over one `SandboxSession`. It answers each call that Harbor makes. A backend subclass adds the image pin, the backend section, the sandbox's shape and the egress switch. |
| `sprout.eval.harbor.podman:PodmanEnvironment` | the `SandboxEnvironment` whose `type()` is `podman`: a rootless podman container whose rootfs is a btrfs subvolume of a store |

Harbor loads the environment from the `import_path` of a `TrialConfig`, with these keyword
arguments in `kwargs`:

| Keyword | Required | What it gives |
|---|---|---|
| `runtime_bin` | yes | the `sprout-runtime` binary. Each container gets it at `/usr/local/bin/sprout-runtime`. |
| `store` | yes | the btrfs directory that `ops/btrfs_store.sh` made |
| `podman_bin` | no | the podman binary. The default is the first of these: `$SPROUT_PODMAN_BIN` when it is not empty, the podman that `ops/setup_podman.sh` installed when it is executable, `podman` on `PATH`. |

`tests/sprout_sandbox/harbor_trials.py` makes each trial with a config like this one:

```python
TrialConfig.model_validate(
    {
        "task": {"path": "<task directory>"},
        "trial_name": "<task>-oracle",
        "trials_dir": "<output directory>",
        "agent_timeout_multiplier": 1.0,
        "agent": {"name": "oracle"},
        "environment": {
            "import_path": "sprout.eval.harbor.podman:PodmanEnvironment",
            "delete": True,
            "kwargs": {"runtime_bin": "<sprout-runtime>", "store": "<btrfs store>"},
        },
    }
)
```

## What the environment does

When Harbor starts the environment, the environment does these steps:

1. If Harbor asks for a build (`force_build`), it raises `ValueError`. Sprout starts the
   task's prebuilt image and never builds one.
2. It selects the image: the task's `docker_image`, or the `FROM` image of the overlay when
   the task has no `docker_image`.
3. It runs `podman image inspect` on the image. If the image is not local, the environment
   pulls it first.
4. It pins the image to the digest of the repository that the task names. If the record has
   no such digest, for example for a local build, it raises `PodmanError`.
5. It reads the image's OCI `User`, `WorkingDir` and `Env` from the same record. If the
   record has no `User` or no `WorkingDir`, the environment uses `root` or `/`.
6. It starts a container from the pinned image, with the task's CPUs and memory, on the
   baseline network policy that Harbor gives. If the container does not start, it raises
   `RuntimeError` with the reason.
7. As root, it makes `/logs/agent`, `/logs/user-agent`, `/logs/verifier` and
   `/logs/artifacts`, with mode 777.
8. It writes the receipt `<session_id>.podman.json` into the trial directory.
9. It puts the task's files into the sandbox, as the table below says.

The default shape of the container is 2 CPUs and 1024 MB. A `public` baseline policy gives
the container egress, and a `no-network` policy gives it none. If a step from 6 on fails, or
Harbor cancels the start, the environment destroys the sandbox and closes the pool. Then it
raises the error again.

The receipt holds these fields:

- `backend`
- `sandbox_id`
- `image`, the pinned image, and `image_config`, its OCI config
- `requested_disk_size_mb`
- `resources`, the CPUs and memory of the sandbox
- `network`, the baseline network policy
- `overlay`, the `COPY` and `RUN` steps of the overlay, or null
- `store`

In step 9, the environment looks at the `Dockerfile` of the environment directory that
Harbor gives. That directory is `environment/` for the agent's sandbox and `tests/` for the
sandbox of a separate verifier.

| Condition | What the environment does |
|---|---|
| The `Dockerfile` is `FROM` the task's own image. This is an **overlay**, such as the `tests/Dockerfile` that Harbor's separate verifier builds from. The same applies to a task without a `docker_image` whose `Dockerfile` a sandbox can replay. | It uploads the overlay's `COPY` sources and runs its `RUN` steps as root. They run in the image's `WorkingDir`, with the image's `Env` and Harbor's variables, within the task's `build_timeout_sec`. |
| The `Dockerfile` is `FROM` another image. | Nothing. That `Dockerfile` is the recipe of the prebuilt image. |
| There is no `Dockerfile`, and the directory holds files. | It uploads the directory, as Harbor's base class does. The files go into the task's `workdir`, or into the directory that `pwd` prints in the sandbox. |

After the start, the environment answers Harbor's calls as follows:

| Harbor's call | What the environment does |
|---|---|
| `exec` | It runs the command through the session's `shell` tool. The command runs in Harbor's `cwd`, else the task's `workdir`, else the image's `WorkingDir`. It gets the image's `Env` with Harbor's variables on top. It runs as Harbor's user, else as the image's `User`. |
| `exec`, the limits | The timeout is Harbor's, else 36000 s: the 8-hour actor budget and time for setup and collection. The output limit is 16 MiB for each stream. |
| `exec`, the result | It returns the command's exit status and output, stdout first to Harbor's output callback. A timeout raises `TimeoutError`. A command without an exit status, or a transport failure, raises `RuntimeError`. |
| `upload_file`, `upload_dir` | As root, it makes the parent directory, then copies the file. A directory goes as one `tar.gz` archive through `/tmp` in the sandbox, and the environment removes the archive. |
| `download_file`, `download_dir` | It copies the file out. A directory comes as one `tar.gz` archive, and the environment removes the archive in the sandbox. |
| a phase network policy | It switches the egress of the running container through `SandboxSession.set_network`. That runs `podman network connect` or `podman network disconnect` on the sandbox's own network. It accepts only `public` and `no-network`. Before the start, it raises `RuntimeError`. |
| `stop(delete=True)` | It destroys the container, then closes the pool. |
| `stop(delete=False)` | It releases the container from the pool first, so that the pool's close leaves the container running. `python -m sprout.worker reap` or a person removes it later. |

A second `stop` does nothing. If the destroy fails, the environment still closes the pool,
then raises an error that names what is left.

## Each sandbox has its own network

Each podman sandbox has a bridge network of its own, because a shared bridge lost traffic
whenever a container joined or left it. The figure shows two sandboxes, their networks and
the route that does not exist between them.

<p align="center">
  <img src="figures/harbor-plugin-networks.svg" width="900" alt="Two sandboxes on one host. The container of sandbox A connects to the network named after sandbox A, and the container of sandbox B connects to the network named after sandbox B. Each network is a bridge that podman creates with --disable-dns and isolate=true. A container is connected to its network only while its sandbox is online: podman network connect and podman network disconnect switch it. Each network carries its container's traffic, DNS included, to the host, whose upstream resolvers the container's resolv.conf names. No aardvark-dns runs. The two networks have no route to each other, because of isolate=true."/>
</p>
<p align="center"><sub>Each container has a bridge network of its own, with no DNS server of podman's and no route to the network of another sandbox.</sub></p>

A Linux bridge whose address nobody set takes the lowest MAC among its ports. On one bridge
for every sandbox, a container that joined or left changed the lowest MAC when its own MAC
was the lowest. Then the gateway's MAC moved under all the other containers. Their packets
went nowhere until their neighbour entries refreshed.

On netavark 1.4, the version in Ubuntu 24.04, we measured 47 MAC changes in 20 s of churn.
Every resolver timed out while it lasted. 42 of 176 TerminalBench gate trials at 32 workers
logged failed lookups.

The sandbox networks also run without aardvark-dns (`--disable-dns`). aardvark-dns is one
process for every network, and it reloads on each change of membership. Without it,
`resolv.conf` names the host's upstream resolvers. With `isolate=true`, no sandbox reaches
the network of another.

The pool names each network after its sandbox. Each network and each container carries the
labels `sprout.managed=1` and `sprout.store=<store>`. The pool creates an offline sandbox,
disconnects it and only then starts it, so that its first command already runs offline.

## Admission limits

The environment refuses a task when Harbor constructs it, before any container exists. It
never falls back to Docker. Sprout's own checks come first, in this order:

1. a `runtime_bin` that is not a file
2. a compose file in the environment directory
3. task-side MCP servers
4. a task with neither a `docker_image` nor a `Dockerfile`
5. a `Dockerfile` `FROM` the task's own image that a sandbox cannot replay
6. a task without a `docker_image` whose `Dockerfile` a sandbox cannot replay, because Sprout
   never builds an image
7. a host mount outside `/logs/agent`, `/logs/verifier`, `/logs/artifacts` and
   `/logs/user-agent`
8. a `store` that is not a directory

Each of these checks raises `ValueError`. Harbor's base class then makes its own checks:

| Refused | Error |
|---|---|
| a CPU or memory enforcement policy that asks for a resource request (`request` or `guarantee`) | `ValueError` |
| a task that needs a GPU or a TPU | `RuntimeError` |
| a network allowlist, as the baseline policy or as a phase policy | `ValueError` |
| a task whose `[environment].os` is `windows` | `RuntimeError` |

A sandbox can replay a `Dockerfile` that has only these instructions:

- one `FROM`
- `COPY <src> <dst>`, without options, from a source that exists
- `RUN`

Each other instruction, such as `ENV`, `WORKDIR` or `USER`, and a second `FROM`, makes the
`Dockerfile` one that a sandbox cannot replay.

## Validate

Do these steps:

1. Run the tests that need no podman:

   ```bash
   python3 -m pytest tests/eval/test_harbor_environment.py tests/contract/test_harbor_task.py -q -m "not slow"
   ```

   `tests/eval/test_harbor_environment.py` drives `PodmanEnvironment` with a fake session
   and the fake podman of `tests/support/tools.py`.
2. Check the pinned API surface of Harbor:

   ```bash
   python ops/contracts/ci_check.py --slot harbor
   ```

   The command prints `0 failures` when Harbor still has each pinned class, method and
   field.
3. Run each large test file alone, while no experiment uses the store:

   ```bash
   python3 -m pytest tests/sprout_sandbox/test_harbor_live.py -m slow
   python3 -m pytest tests/sprout_sandbox/test_harbor_grade_live.py -m slow
   ```

4. Run the dataset test:

   ```bash
   python3 -m pytest tests/contract/test_harbor_task_datasets.py -m slow
   ```

The large tests of step 3 are marked `slow`. They need these things:

- Harbor, from the `harbor` extra
- podman
- `btrfs-progs`
- the runtime binary `runtime/sprout-runtime`
- a btrfs store at `SPROUT_PODMAN_STORE` (default `~/sprout-btrfs`)

Each of them skips with a reason when one is missing. They also skip when the store is in
use: when it has a live sandbox, or when a container with the label `sprout.managed=1`
exists. A large test removes each container, network and subvolume that appears while it
runs.

The tests check these behaviors:

- `tests/sprout_sandbox/test_harbor_live.py` runs Harbor's own `oracle` and `nop` trials on
  the fixture task (`harbor_trial`). They must score 1 and 0, and leave no container,
  network, sandbox, snapshot or socket behind. The file also checks the receipt, the phase
  policy and an image whose `USER` is not root.
- `tests/sprout_sandbox/test_harbor_grade_live.py` grades the worker's own `oracle` and
  `nop` rollouts of the same task with the `harbor` grade. It grades from the snapshot and
  in the kept sandbox, and compares the rewards with Harbor's.
- `tests/contract/test_harbor_task_datasets.py` checks that
  `sprout.contract.harbor_task.load_task` reads every task of the three datasets as Harbor's
  own task model reads it. It needs `harbor==0.22.0` and the datasets under
  `$SPROUT_HARBOR_DATASETS` (default `~/ash-data/datasets`), and it starts no sandbox.
