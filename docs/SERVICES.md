# Services

`python -m sprout.services` starts the persistent services of one experiment host and keeps them
up, so that you start nothing by hand before a batch. One file, `~/.config/sprout/services.yaml`,
lists the services. The code knows how to start a process or a container and how to probe it, but
nothing about a particular service. Read this document if you set up a host, if a service on a
host is down, or if you change `src/sprout/services/`.

The package imports no module that needs an optional extra, and no code of the services that it
starts. So the CLI runs before you install the extras. `tests/services/test_package_dependencies.py`
and `tests/repo/test_layering.py` give the allowed modules and enforce the rule.

## Start the services of a host

On a new host, run the guided setup. It checks the host and writes the config files, and it asks
before each change ([Guided setup](#guided-setup)):

```bash
python -m sprout.services setup
```

To do the same work by hand, do these steps:

1. Write the config files and the secrets with `python -m sprout.services init`.
2. Edit the files that `init` names. The table after these steps lists them.
3. Start each enabled service and its dependencies with `python -m sprout.services up`.
4. Export the variables of the services into your shell with
   `eval "$(python -m sprout.services env)"`.
5. Check the services with `python -m sprout.services doctor`. If a check or a probe fails,
   `doctor` exits 1.

| File in `~/.config/sprout/` | What to edit |
|---|---|
| `litellm.yaml` | The route of each model alias ([Model endpoints](MODEL_ENDPOINTS.md)) |
| `runstore.json` | The profiles: `snapshot_repository`, `sandbox_resources` and `worker_env` ([Run Store](RUNSTORE.md)) |
| `rl_driver.json` | `miles.tasks`, if you use the RL driver ([RL driver](RL_DRIVER.md)) |

## The services and their dependencies

The CLI starts and stops five services and only checks three. Each process receives the exports
of the services that it depends on. The figure shows each `depends_on` edge of the template with
the export that passes along it. It also shows the facts that the worker's doctor gives the CLI.

<p align="center">
  <img src="figures/services-dependencies.svg" width="1000" alt="The eight services of the packaged services.yaml template and the worker's doctor. db is a podman container on port 15432. runstore-api is a process on port 18110 that depends on db and receives SPROUT_RUNSTORE_DSN from it. runstore-worker is a process that depends on runstore-api, podman and store, and receives SPROUT_RUNSTORE_TOKEN, SPROUT_PODMAN_BIN and SPROUT_PODMAN_STORE from them. rl-driver is a process on port 11001 that depends on runstore-api and receives SPROUT_RUNSTORE_TOKEN from it. proxy is a process on port 4000, and aenv is a disabled external service. Neither has a dependency. podman and store are checks. The worker's doctor, which the CLI runs as {python} -m sprout.worker doctor --json, gives the verdict of the podman check and of the store check, and the values of the {podman_bin} and {store} placeholders. The db container runs on {podman_bin}. Exports also pass on transitively: runstore-worker and rl-driver also receive SPROUT_RUNSTORE_DSN from db through runstore-api. The CLI starts and stops db, proxy, runstore-api, runstore-worker and rl-driver. It only checks or probes podman, store and aenv."/>
</p>
<p align="center"><sub>The services of the template, each <code>depends_on</code> edge with the export that passes along it, and the facts from the worker's doctor.</sub></p>

| Service | Kind | Health probe | Command or image |
|---|---|---|---|
| `db` | `podman` | `pg_isready` on `127.0.0.1:15432` | `docker.io/library/postgres:16-alpine` |
| `proxy` | `process` | `GET http://127.0.0.1:4000/health/liveliness` | `litellm` on `127.0.0.1:4000` |
| `runstore-api` | `process` | `GET http://127.0.0.1:18110/v1/jobs` with the bearer token | `sprout.runstore serve` on `127.0.0.1:18110` |
| `runstore-worker` | `process` | `process` | `sprout.worker start --concurrency 1` |
| `rl-driver` | `process` | `GET http://127.0.0.1:11001/health` with the bearer token | `sprout.rl_driver` on `127.0.0.1:11001` |
| `podman`, `store` | `check` | the doctor's `podman` or `store` check | none |
| `aenv` | `external`, `enabled: false` | `GET $AENV_SERVER_URL/sandboxes` with `X-API-Key` | none |

The CLI builds the environment of a process from these parts, in this order. A later part
replaces a variable of an earlier part:

1. the environment of the CLI, without the variables in `unset_env`
2. the `exports` of each service that the process depends on, transitively, dependencies first
3. the `exports` of the process itself
4. the `env` of the process

A container receives only its own `env`. The CLI gives `podman run` each name as `-e NAME`, and
the value in the environment of `podman run`. So no value appears in an argv.

## Commands

The CLI has nine commands. Each command prints plain text without colour, one line for each
action. `--help` lists the flags. `env` is the only command that prints a secret.

Each command exits 0 if each action succeeded or each report was printed. It exits 1 on any
failure, for example an unknown name or Ctrl-C. An invalid command line exits 2. `logs` of a
container exits with the status of `podman logs`, and `logs -f` exits 0 on Ctrl-C. Without
`services.yaml`, each command except `init` and `setup` tells you to run `init`.

| Command | What it does |
|---|---|
| `setup` | Guides you through the host checks, the config files, the model endpoint and the start ([Guided setup](#guided-setup)). |
| `init` | Writes each missing config file and secret ([`init`](#init)). It never overwrites one. |
| `up [NAME...]` | Starts the named services and their dependencies, dependencies first. Without names, it starts each `enabled` service. It leaves a healthy service alone ([`up`](#up)). |
| `down [NAME...]` | Stops the named services and their dependents, dependents first. Without names, it stops each `process` and `podman` service ([`down` and `restart`](#down-and-restart)). |
| `restart NAME...` | Runs `down`, then `up`, for the named services and their dependents. |
| `status` | Prints `name \| kind \| pid \| health \| port` for each service, disabled ones included. A `check` shows the doctor's verdict. |
| `logs NAME [-f] [-n N]` | Prints the last `N` lines of `<state>/<name>/log`, 50 by default. With `-f`, it follows the log. For a container, it runs `podman logs [-f]` instead. |
| `env` | Prints an `export NAME=value` line for each export of each service, enabled or not, quoted for `eval`. A later export of a name replaces an earlier one. |
| `doctor` | Prints each host check, each problem in `runstore.json` and the probe of each enabled service ([`doctor`](#doctor)). It starts nothing. |

### `up`

`up` goes through the chosen services in dependency order. Independent services keep the order
of the file. The figure shows what `up` does for one service.

<p align="center">
  <img src="figures/services-up.svg" width="1000" alt="The tests that python -m sprout.services up makes for each chosen service, in dependency order, from the top. First: did a dependency fail? If yes, the service is skipped and counts as failed. Second: is the kind check? If yes, the result is the verdict of the worker's doctor, and the service is up only if the verdict is ok. Third: does the health probe pass? If yes, the service is already healthy and up does nothing more. Otherwise the kind decides. An external service: up prints its start_hint, and the service counts as failed, because the server runs as root. A podman service: up starts the container. If the container runs, up does nothing to it. If it is stopped, up runs podman start. If it is absent, up runs podman run -d. Then up asks whether the probe passes within timeout_s: yes, the service is up; no, it failed. A process service: if its process runs although the probe failed, the service failed, because up starts no second instance. Otherwise up runs the prepare steps, then starts the process detached, and asks whether the probe passes within timeout_s: yes, the service is up; no, it failed. A failed service makes up skip each service that depends on it."/>
</p>
<p align="center"><sub>The tests that <code>up</code> makes for one service, and the start of each kind.</sub></p>

A service also fails in these cases:

- A `podman` command fails, or the container stops before its probe passes.
- A `prepare` step exits with a non-zero status. `up` skips a step whose `unless` path exists.
- A placeholder has no value ([Validation](#validation)).

A proxy that you started by hand on port 4000 passes the probe, so `up` leaves it alone. `up`
makes a secret that does not exist yet. For a failed start, `up` prints
`<name>: FAILED: <reason>`. After it has gone through every chosen service, `up` exits 1 if one
failed.

### `down` and `restart`

A service whose dependency stops does not work. So `down` also stops each service that depends on
a named service, transitively, in reverse dependency order. For example, `down db` also stops
`runstore-api` and the two services that depend on it. `down` acts on each kind in this way:

- A process gets SIGTERM, then SIGKILL ([How a process is supervised](#how-a-process-is-supervised)).
- A running container gets `podman stop --time 20`. The CLI never removes a container.
- A named `check` or `external` service gets the line `nothing to stop`.

If a stop fails, `down` prints a `FAILED` line, goes on, and exits 1 at the end. Then `restart`
starts nothing. Otherwise `restart` runs `up` for the same services and their dependencies.

### `doctor`

`doctor` prints one line for each check. Each line starts with a status word, such as `ok`. The
lines come in this order:

1. `uv`: whether `uv` is on PATH or in `~/.local/bin`.
2. Each check of the worker's doctor that no `check` service in `services.yaml` shows. A disabled
   `check` service also counts. With the template, these are `runtime` and `keyring`. Without a
   report, this is one failed `worker` line.
3. Each problem in `runstore.json` ([Config checks](#config-checks)), or one `ok` line. Without
   the file, there is no line.
4. One line for each service, in the order of the file:
   - `skip`, for a disabled service
   - the doctor's verdict, for a `check`
   - the probe and the pid, for each other service

If a line is `fail`, `doctor` exits 1. A `warn` or a `skip` does not change the exit code.

### `init`

`init` writes each of these files that does not exist, then makes each secret that
`services.yaml` names. It prints one line for each file and secret, then the files to edit:

1. `services.yaml`, from `resources/services/services.yaml`. If the file is invalid, `init`
   exits 1.
2. `litellm.yaml`, a copy of `resources/services/litellm.yaml`.
3. `runstore.json`, from `resources/runstore/config.example.json`, with these values of the host:
   - `artifact_root`: `<state>/runstore-artifacts`
   - each `python`: `{python}`
   - each `runtime_bin`: `runtime/sprout-runtime` of the checkout
   - the `store` of the `podman` backend in `run_defaults` and `grade_defaults`:
     `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`
4. `rl_driver.json`, from `resources/rl_driver/config.example.json`, with the `ledger`
   `<state>/rl-driver/groups.sqlite3` and the `driver_token_env` `SPROUT_RL_DRIVER_TOKEN`.

### Config checks

`doctor` and `setup` check `runstore.json` for the problems that make jobs fail. The file that
`init` writes has none of them. `doctor` ends each problem line with `Fix: <what to do>`. `setup`
prints the fix on the next line.

| Problem | Status |
|---|---|
| The file is not valid JSON, or it has no `profiles` object. | `fail` |
| A string value starts with `/path/to/`, the prefix of the example paths of the template. | `fail` |
| A `{"$env": NAME}` reference in a `worker_env` names a variable that this shell does not set. Each job of the profile fails with `KeyError`, unless the shell of the worker sets it. | `warn` |

### Guided setup

`python -m sprout.services setup` sets up a new host in four steps, and asks before each change.
Each question shows its default in brackets, and Enter takes it:

1. **Host checks.** The setup prints each check of the worker's doctor
   ([The worker's host checks](#the-workers-host-checks)), with the fix under each check that did
   not pass. If a check failed, the setup asks whether to go on, by default yes. If you answer
   no, the setup stops and writes no file.
2. **Config files.** The setup writes each missing config file and secret, as `init` does. Then
   it prints each problem of [Config checks](#config-checks) with its fix.
3. **Model endpoint.** The setup asks for these values:
   - the endpoint URL, by default `$QWEN_BASE_URL`, else `$OPENAI_BASE_URL`. An empty URL skips
     the rest of this step. The setup refuses a URL that `sprout.contract.sampling.endpoint`
     refuses for a rollout request.
   - the model name, by default `qwen3.8-27b`
   - the variable that holds the key, by default `SPROUT_MODEL_API_KEY`. The setup says whether
     this shell sets it, and never prints the value.

   Then the setup offers to add `{"NAME": {"$env": "NAME"}}` to the `worker_env` of `mini`,
   `mini-microvm` and `review`, the profiles whose jobs call the model, by default yes. For each
   other `worker_env` variable that this shell does not set, it offers to remove the reference,
   by default no. Last, it offers one chat completion with `max_tokens` 16 to
   `<endpoint>/v1/chat/completions`, by default no, because it costs a request. The setup prints
   the HTTP status, with the key replaced by `***`.
4. **Start.** The setup offers to run `python -m sprout.services up runstore-api runstore-worker`,
   by default no. Then it prints the `env` line and an example run of `examples.qwen.run` with
   `--plan` for this host.

Only step 3 changes a file that exists, and only for a change that you accept. Before it rewrites
`runstore.json`, the setup copies the old file to `<config>/backup-<YYYYmmdd-HHMMSS>/`. The
worker takes the environment of the shell that starts it, so start it in a shell that sets the
key variable. A worker that runs already reads changed profiles only after
`python -m sprout.services restart runstore-worker`.

A flag answers each question, and `--help` lists the flags. With `--yes`, each question that no
flag answers takes its default, so a script can run the setup without a terminal. The setup exits 1
in these cases:

- You answer no after a failed host check, or `services.yaml` is invalid.
- A flag, or a default under `--yes`, gives an answer that the question refuses.
- Without `--yes`, the input ends before an answer.
- A service that you chose to start does not come up.

## Where things live

The CLI keeps its files under two roots. An override variable comes before the XDG variable. A
leading `~` in an override is the `HOME` of the environment.

| Root | Default | Override |
|---|---|---|
| Config | `$XDG_CONFIG_HOME/sprout`, else `~/.config/sprout` | `SPROUT_SERVICES_CONFIG_DIR` |
| State | `$XDG_STATE_HOME/sprout/services`, else `~/.local/state/sprout/services` | `SPROUT_SERVICES_STATE_DIR` |

The config root holds `services.yaml` and the three files that `init` names. The state root holds
what the programs make:

| Path under the state root | What it holds |
|---|---|
| `<name>/` | The working directory of the service, with its `pid` file and its `log` |
| `db/data` | The data directory of PostgreSQL |
| `litellm-env/` | The venv of the proxy |
| `runstore-artifacts/` | The `artifact_root` of `runstore.json` |
| `rl-driver/groups.sqlite3` | The ledger of the RL driver |
| `secrets/<name>` | One secret |

The log of a service holds a timestamped argv and the output of each command that the CLI runs to
start or stop the service. A process writes its own output to the same log. The log holds no
probe. The output of a container is in `podman logs`.

The `paths` fixture of `tests/services/conftest.py` moves `HOME` and both roots into the directory
of the test. To run a second instance of every service, set both override variables, and change
the ports in the `services.yaml` of that instance.

### Secrets

Each secret is one file `<state>/secrets/<name>` with mode 0600, in a directory with mode 0700.
Its value is `secrets.token_urlsafe(32)`. A command makes it one time, and later commands read it:

| Commands | A secret that does not exist yet |
|---|---|
| `init`, `setup`, `up`, `restart`, `env` | The command makes it. Each time it reads a secret, it sets both modes again. |
| `down`, `status`, `doctor`, `logs` | The command makes no secret. It reports an error for each service that needs one. |

## How experiments consume it

An experiment shell runs `eval "$(python -m sprout.services env)"`, then passes the variables to
its commands. [The key of each request](MODEL_ENDPOINTS.md#the-key-of-each-request) says how the
model key reaches the worker. With the template, `env` exports these variables:

| Variable | From | Value |
|---|---|---|
| `SPROUT_PODMAN_BIN`, `SPROUT_PODMAN_STORE` | `podman`, `store` | The podman and the store of the worker's doctor report |
| `SPROUT_RUNSTORE_DSN` | `db` | `host=127.0.0.1 port=15432 dbname=sprout_runstore user=sprout_runstore password=<db-password>` |
| `SPROUT_RUNSTORE_TOKEN` | `runstore-api` | The `runstore-token` secret. A client sends it as `Authorization: Bearer <token>`. |
| `SPROUT_RL_DRIVER_TOKEN` | `rl-driver` | The `rl-driver-token` secret |
| `OPENAI_BASE_URL` | `proxy` | `http://127.0.0.1:4000/v1` |
| `OPENAI_API_KEY`, `SPROUT_LITELLM_KEY` | `proxy` | The `litellm-master-key` secret |

The worker also receives the podman and the store, so it runs its sandboxes where the experiment
shell expects them. A service that is not in `services.yaml` exports nothing.

## The services

This section gives the facts that the tables do not give. [Run Store](RUNSTORE.md) and
[RL driver](RL_DRIVER.md) describe the programs.

### `db`

`db` is PostgreSQL 16 in the container `sprout-services-db` on `{podman_bin}`, the podman of the
sandboxes. It publishes `127.0.0.1:15432`, because the PostgreSQL of the host owns port 5432. The
probe uses the `pg_isready` of the host, which proves the published port that the Run Store
connects through. The data directory `<state>/db/data` is a bind mount, so the data survives
`down` and a reboot.

A container belongs to the podman that made it. To move `db` to another podman, do these steps:

1. Stop and remove `sprout-services-db` with the old podman. The data directory stays.
2. In a shell where `SPROUT_PODMAN_BIN` names the new podman, run `python -m sprout.services up db`.
3. In the same shell, run `python -m sprout.services restart runstore-worker`.

### `proxy`

`proxy` is the LiteLLM proxy. [Model endpoints](MODEL_ENDPOINTS.md) says when to use it and how
to set its aliases. The proxy has its own venv, `<state>/litellm-env`, because litellm pins other
httpx and pydantic versions than Sprout. Before a start, two `prepare` steps make the venv with
`uv` and install `litellm[proxy]==1.103.0`. The proxy inherits the provider credentials from the
shell that runs `up`. `up` waits up to 120 s, because the proxy imports for a while before it
listens.

To add a provider SDK that litellm does not bundle, do these steps:

1. Add the SDK to the install step of the `proxy` entry in `services.yaml`.
2. Remove `<state>/litellm-env`. The install step does not run while
   `<state>/litellm-env/bin/litellm` exists.
3. Run `python -m sprout.services restart proxy`.

### `runstore-api`, `runstore-worker` and `rl-driver`

These three processes run Sprout programs on `{python}`. Their entries add these facts:

- Before each start of `runstore-api`, `up` runs `{python} -m sprout.runstore init`. That command
  is idempotent, and it refuses a volatile `artifact_root`. The API has no `/health` route, so the
  probe sends `GET /v1/jobs` with the token. If the probe gets 401, the API has another token.
- Before it registers, `runstore-worker` checks the profiles of `runstore.json`
  ([Run Store](RUNSTORE.md#checks-at-the-start)). If a check fails, the worker exits within the
  first second, before its probe passes at 2 s. The service log gives the reason.
- `rl-driver` checks its token on each route, `/health` included. So the probe sends the token,
  and a probe without it reads a healthy driver as down.

### `aenv` (external, disabled)

`aenv` is the AgentENV microVM server for `--backend microvm`. The server runs as root, so the CLI
never starts it. If the probe fails, `up aenv` prints the `start_hint` and exits 1. The hint
rebuilds the `ublk_drv` module if a reboot removed it, then starts the server. Replace
`<ublk_drv source directory>` in the hint with the directory of the module source on your host.

## The worker's host checks

`python -m sprout.worker doctor` checks the machine that a worker runs on, and changes nothing. It
prints one line for each check. If a check is `fail`, it exits 1. A `warn` does not change the
exit code. The `podman` and `store` services show its first two checks, and nothing starts them.

| Check | Passes when | Otherwise |
|---|---|---|
| `podman` | `<podman> --version` exits 0 within 15 s. The podman is `--podman-bin`, else `$SPROUT_PODMAN_BIN`, else `~/.local/opt/podman/bin/podman` if it is executable, else `podman` on PATH. | `fail`. `ops/setup_podman.sh` installs a podman. |
| `store` | The store is a directory, `btrfs` is on PATH, and `findmnt` reports a btrfs mount with `user_subvol_rm_allowed`. The store is `--store`, else `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`. `PodmanPool` accepts the same stores. | `fail`, in the words of the pool. You make a store one time, as root, with `sudo ops/btrfs_store.sh <mount point> <image file> <size>`. |
| `runtime` | The runtime binary is a file: `--runtime-bin`, else `runtime/sprout-runtime` of the checkout. | `fail`, with the build command |
| `keyring` | `kernel.keys.maxkeys` is 20000 or more. Below that, rootless podman starts fail at about 180 concurrent containers. | `warn`, with the `sysctl` fix. Also `warn` if the value is not readable. |

With `--json`, the doctor prints one object instead:
`{"podman_bin": ..., "store": ..., "checks": {"<check>": {"status": ..., "detail": ...}}}`. The
podman and the store are the ones that it checked. The services CLI reads the object by these rules:

- When a command first needs the report, the CLI runs `{python} -m sprout.worker doctor --json`
  in its own environment. It waits up to 120 s.
- In one command, the CLI runs the doctor at most one time for each set of flags. A `check` with
  a `path` adds `--podman-bin <path>` or `--store <path>`. A `path` does not change `{podman_bin}`
  or `{store}`, which come from the report without flags.
- If the doctor prints no report, each check fails with the exit status and the last stderr line
  of the doctor. For example, a `{python}` that cannot import `sprout.worker` gives no report.

## services.yaml

`resources/services/services.yaml` is the template, and its comments explain each entry. The file
has two top-level keys:

- `python`: the interpreter that `{python}` names. If it is null, `{python}` is the
  `.venv/bin/python` of the checkout if that exists, else the interpreter that runs the CLI.
- `services`: a mapping from each service name to its entry. A name matches `^[a-z][a-z0-9-]*$`.

### Kinds

Each service has one of these kinds:

| Kind | What the CLI does | Keys |
|---|---|---|
| `process` | Starts the command detached, with its pid and log under `<state>/<name>`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `command`, `env`, `unset_env`, `prepare`, `timeout_s` |
| `podman` | Runs a container on `{podman_bin}`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `image`, `container`, `publish`, `volumes`, `env`, `timeout_s` |
| `external` | Starts nothing. If the probe fails, `up` prints `start_hint`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `start_hint` |
| `check` | Starts nothing. Shows a check of the worker's doctor. | `kind`, `enabled`, `check` (`podman` or `btrfs_store`), `path`, `exports` |

The template's comments explain each key. `enabled` is `true` by default. If it is `false`, a bare
`up` leaves the service out. `status` shows `disabled`, and `doctor` prints `skip`. A command that
names the service still acts on it. `timeout_s` is 60 s by default. `port` is informational.

### Health kinds

Each `health` mapping has one of these kinds. A service without `health` has the kind `none`.

| Kind | Passes when | Keys |
|---|---|---|
| `http` | A GET of `url` answers 200 within 3 s. | `url` (required), `headers` |
| `pg_isready` | The `pg_isready -h <host> -p <port>` of the host exits 0 within 15 s, with `-U <user>` if `user` is set. If the host has no `pg_isready`, the CLI runs it in `container` with `podman exec`. | `port` (required), `host` (default `127.0.0.1`), `user`, `container` (default: the `container` of the service) |
| `process` | For a process: the pid in the pid file is alive and runs the service's command. During `up`, the process must also stay alive for 2 s. For a container: the container runs. | none |
| `none` | For a process or a container: the same as `process`. For an `external` service: always. | none |

### Placeholders

The CLI resolves these placeholders when a command runs:

| Placeholder | Value |
|---|---|
| `{python}` | The interpreter of the `python` key |
| `{repo}` | The checkout that the package was imported from |
| `{config}`, `{state}` | The config root and the state root |
| `{podman_bin}`, `{store}` | The podman and the store of the worker's doctor report |
| `{secret:NAME}` | The secret `NAME`. A name matches `^[a-z0-9][a-z0-9-]*$`. |
| `{env:NAME}` | The variable `NAME` of the CLI's environment |

The field decides whether a placeholder or a secret can go in it:

| Fields | Placeholders | `{secret:NAME}` |
|---|---|---|
| `env`, `exports`, `health.headers` | expanded | allowed |
| `command`, `prepare` (`run` and `unless`), `volumes`, `publish`, `health.url`, `path` | expanded | refused |
| `unset_env` | used as written, but checked | refused |
| `image`, `container` | used as written, and a brace is refused | refused |
| `start_hint`, `health.host`, `health.user`, `health.container` | used as written | not checked |

A secret can go only where its value becomes the environment of a process, or where `env` prints
it. Each other field can reach the log or the terminal as written, and each user on the host can
read an argv in `/proc/<pid>/cmdline`. After the expansion, a leading `~` is the home directory.

### Validation

The loader checks the whole file each time a command loads it, so a bad entry fails before `up`
starts anything. At the first problem, it names the service and the field. For example, it refuses
an unknown key or a dependency cycle.

A placeholder can be valid in the file but have no value in a run, in these cases:

- `{env:NAME}`, when the variable is not set
- `{secret:NAME}`, when the secret does not exist and the command makes no secret
- `{podman_bin}` or `{store}`, when the worker's doctor gives no report

Then `up` and `down` print one `FAILED` line for the service, and `up` skips its dependents.
`status` shows a `down: ...` cell. `env` prints the reason on stderr and exits 1.

## How a process is supervised

The CLI starts a process detached from the shell, so a closed terminal does not stop it. The
figure shows the states of a process service and what changes each one.

<p align="center">
  <img src="figures/services-process.svg" width="1000" alt="The states of one process service under python -m sprout.services, with the trigger and the result of each transition. not running: there is no pid file, or its pid is not the service's process. From not running, up runs the prepare steps and starts the process detached, and the service is starting: the CLI polls the probe every 0.5 s. From starting: if the probe passes, the service is up. If the process exits first, the start failed, the CLI removes the pid file, and the service is not running. If timeout_s runs out, the start failed, and the process is left running with a failing probe. From up: if the probe fails while the process lives, the service runs with a failing probe. If the process exits on its own, the service is not running, and the next up starts it again. In the state runs with a failing probe, up fails and starts no second instance. From up, or from runs with a failing probe, down sends SIGTERM to the process group, and the service is stopping. From stopping: if the process is gone within 20 s of SIGTERM, down prints stopped. Else down sends SIGKILL, and if the process is gone within 5 s, down prints killed. Both go to not running, and down removes the pid file. If the process survives SIGKILL, down prints FAILED and exits 1."/>
</p>
<p align="center"><sub>The states of a process service, with the trigger and the result of each transition.</sub></p>

`src/sprout/services/supervisor.py` follows these rules:

| Subject | Rule |
|---|---|
| Program | For `command` and each `prepare` step, a path is used as written. A bare name is looked up on PATH, then in `~/.local/bin`, where the uv installer puts uv and which a cron shell lacks on PATH. If no program is found, the start fails. |
| Start | The process runs in a session of its own (`setsid`), with stdin from `/dev/null` and its output appended to `<state>/<name>/log`. The CLI writes its pid to `<state>/<name>/pid`. |
| Working directory | `<state>/<name>`, never the checkout. `conmon` can write an `oom` marker into its working directory, and it once dirtied the checkout that way. |
| A failed start | The CLI prints the last 20 lines of the log. After a timeout, the process keeps running, so that you can read `logs <name>`, then run `down <name>`. |
| Ownership | A pid counts as the service's only while the process exists, is not a zombie, and its `/proc/<pid>/cmdline` holds each argument of the expanded command. For a command without arguments, the base names of the programs must match. The program itself is not compared, because the `#!` line of a venv script runs it under python. |
| Stop | `down` signals the process group. If the pid file does not name the service's process, `down` removes the pid file and signals nothing. If a placeholder in `command` has no value, `down` signals nothing and prints `FAILED`. |

By the ownership rule, the CLI never signals a recycled pid. After a reboot, a stale pid file reads
as not running.
