# Services

`python -m sprout.services` keeps the persistent services of one experiment host up, so that
you start nothing by hand before a batch. One file, `~/.config/sprout/services.yaml`, lists the
services. The code knows how to start a process or a container and how to probe it. It knows
nothing about a particular service. Read this document if you set up an experiment host, or if a
service on a host is down.

The CLI starts and stops these services:

- `db`: the PostgreSQL of the Run Store, in a rootless podman container
- `proxy`: the LiteLLM proxy in front of the providers that Sprout cannot call directly
- `runstore-api`: the Run Store API
- `runstore-worker`: the Run Store worker
- `rl-driver`: the RL driver

The CLI only checks these:

- `podman`: the podman that the sandboxes run on
- `store`: the btrfs store of that podman
- `aenv`: the AgentENV microVM server, if you enable it

The worker owns the checks of its host. So the CLI asks `python -m sprout.worker doctor` about
the podman and the store ([The worker's host checks](#the-workers-host-checks)).

The package imports only these modules:

- the standard library
- PyYAML
- `sprout.resource`
- its own modules

`tests/services/test_package_dependencies.py` and `tests/repo/test_layering.py` enforce this. So
the CLI runs before you install the optional extras, and it imports no code of the services that
it starts.

## Start the services of a host

Do these steps on a new host:

1. Write the config files and the secrets:

   ```bash
   python -m sprout.services init
   ```

   The command prints each file and secret that it wrote or kept, then the files to edit.
2. Edit each file that `init` names. The table after these steps lists them.
3. Start each enabled service, dependencies first:

   ```bash
   python -m sprout.services up
   ```

   The command prints one line for each action.
4. Export the variables of the services into your shell:

   ```bash
   eval "$(python -m sprout.services env)"
   ```

   The shell now has the variables that
   [How experiments consume it](#how-experiments-consume-it) lists.
5. Look at the services:

   ```bash
   python -m sprout.services status    # name | kind | pid | health | port
   python -m sprout.services doctor    # every check and probe; starts nothing
   ```

   If a check or a probe fails, `doctor` exits 1.

`init` tells you to edit these files:

| File | What to edit |
|---|---|
| `~/.config/sprout/litellm.yaml` | The route of each model alias ([Model endpoints](MODEL_ENDPOINTS.md)) |
| `~/.config/sprout/runstore.json` | The profiles: the podman store, `snapshot_repository`, `sandbox_resources` and `worker_env` |
| `~/.config/sprout/rl_driver.json` | `miles.tasks`, if you use the RL driver |

## The services and their dependencies

Three services depend on others, and each dependent receives the exports of its dependencies.
The figure shows each `depends_on` edge of the template with the export that passes along it. It
also shows the facts that the worker's doctor gives the CLI. The table after the figure lists
each service.

<p align="center">
  <img src="figures/services-dependencies.svg" width="1000" alt="The eight services of the packaged services.yaml template and the worker's doctor. db is a podman container on port 15432. runstore-api is a process on port 18110 that depends on db and receives SPROUT_RUNSTORE_DSN from it. runstore-worker is a process that depends on runstore-api, podman and store, and receives SPROUT_RUNSTORE_TOKEN, SPROUT_PODMAN_BIN and SPROUT_PODMAN_STORE from them. rl-driver is a process on port 11001 that depends on runstore-api and receives SPROUT_RUNSTORE_TOKEN from it. proxy is a process on port 4000, and aenv is a disabled external service. Neither has a dependency. podman and store are checks. The worker's doctor, which the CLI runs as {python} -m sprout.worker doctor --json, gives the verdict of the podman check and of the store check, and the values of the {podman_bin} and {store} placeholders. The db container runs on {podman_bin}. Exports also pass on transitively: runstore-worker and rl-driver also receive SPROUT_RUNSTORE_DSN from db through runstore-api. The CLI starts and stops db, proxy, runstore-api, runstore-worker and rl-driver. It only checks or probes podman, store and aenv."/>
</p>
<p align="center"><sub>The services of the template, each <code>depends_on</code> edge with the export that passes along it, and the facts from the worker's doctor.</sub></p>

The template defines these services:

| Service | Kind | `depends_on` | Health probe | Exports |
|---|---|---|---|---|
| `db` | `podman` | none | `pg_isready` on `127.0.0.1:15432` | `SPROUT_RUNSTORE_DSN` |
| `proxy` | `process` | none | `GET http://127.0.0.1:4000/health/liveliness` | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `SPROUT_LITELLM_KEY` |
| `runstore-api` | `process` | `db` | `GET http://127.0.0.1:18110/v1/jobs` with the bearer token | `SPROUT_RUNSTORE_TOKEN` |
| `runstore-worker` | `process` | `runstore-api`, `podman`, `store` | `process` | none |
| `rl-driver` | `process` | `runstore-api` | `GET http://127.0.0.1:11001/health` with the bearer token | `SPROUT_RL_DRIVER_TOKEN` |
| `podman` | `check` | none | the doctor's `podman` check | `SPROUT_PODMAN_BIN` |
| `store` | `check` | none | the doctor's `store` check | `SPROUT_PODMAN_STORE` |
| `aenv` | `external`, `enabled: false` | none | `GET $AENV_SERVER_URL/sandboxes` with `X-API-Key` | none |

These rules connect the services:

- `up` starts each dependency before its dependents. `down` stops each dependent before its
  dependencies.
- The CLI builds the environment of a process from these parts, in this order:
  1. the environment of the shell that runs the CLI, without the variables in `unset_env`
  2. the `exports` of each service that the process depends on, transitively, dependencies first
  3. the `exports` of the process itself
  4. the `env` of the process

  A later part replaces a variable of an earlier part. So `runstore-api` receives
  `SPROUT_RUNSTORE_DSN` from `db`. `runstore-worker` and `rl-driver` receive the DSN and
  `SPROUT_RUNSTORE_TOKEN` through `runstore-api`. `runstore-worker` also receives
  `SPROUT_PODMAN_BIN` and `SPROUT_PODMAN_STORE` from the two checks.
- A container receives only its own `env`. The CLI gives `podman run` each name as `-e NAME`,
  and the value in the environment of `podman run`.
- The worker's doctor gives the verdict of each `check` service and the values of `{podman_bin}`
  and `{store}`. The `db` container runs on `{podman_bin}`.

## Commands

The CLI has eight commands. Each command prints plain text without colour, one line for each
action.

| Command | What it does |
|---|---|
| `init` | Writes `services.yaml` and the config files beside it from the templates, and makes the secrets. It never overwrites a file or a secret. |
| `up [NAME...]` | Starts the named services and their dependencies, dependencies first. Without names, it starts each `enabled` service and its dependencies. It leaves a healthy service alone. |
| `down [NAME...]` | Stops the named services and each service that depends on them, dependents first. Without names, it stops each `process` and `podman` service. |
| `restart NAME...` | Runs `down`, then `up`, for the named services and their dependents. |
| `status` | Prints one row for each service: `name \| kind \| pid \| health \| port`. |
| `logs NAME [-f] [-n N]` | Prints the last lines of the service's log. With `-f`, it follows the log. |
| `env` | Prints an `export NAME=value` line for each export of each service, quoted for `eval`. |
| `doctor` | Prints each host check and the probe of each enabled service. It starts nothing. |

### `up`

`up` goes through the chosen services in dependency order. Independent services keep the order
of the file. For each service, `up` does these steps:

1. If a dependency of the service failed, `up` prints `<name>: skipped (<dependency> failed)`.
   The skipped service counts as failed, and `up` goes on with the next service.
2. If the service is a `check`, `up` prints the verdict of the worker's doctor. The service is up
   only if the verdict is `ok`.
3. `up` runs the health probe. If the probe passes, `up` prints `<name>: already healthy` and
   does nothing more. A proxy that you started by hand on port 4000 also counts as healthy.
4. If the service is `external`, `up` prints its `start_hint`. The service counts as failed.
5. If the service is a `podman` container, `up` acts on the state of the container:
   - If the container runs, `up` waits for its probe.
   - If the container exists but is stopped, `up` runs `podman start`, then waits.
   - If the container does not exist, `up` runs `podman run -d`, then waits.

   `up` waits up to `timeout_s`. If the container stops first, or the time runs out, the service
   fails.
6. If the service is a process that runs but failed its probe, the service fails. `up` does not
   start a second instance. The message tells you to run `logs <name>`, then `restart <name>`.
7. `up` runs each `prepare` step, except a step whose `unless` path exists. If a step exits with
   a non-zero status, the service fails. `up` prints the last lines of the log.
8. `up` starts the process and waits for its probe
   ([How a process is supervised](#how-a-process-is-supervised)).

For a failed start, `up` prints one line: `<name>: FAILED: <reason>`. For a failed check, `up`
prints the verdict. For an external service, it prints the hint. If a service failed, `up` exits
1 after it has gone through every chosen service.

When `up` needs a secret that does not exist yet, it makes the secret.

### `down` and `restart`

A service whose dependency stops does not work, so `down` also stops each service that depends
on a named service, transitively. For example, `down db` also stops `runstore-api` and the two
services that depend on it.

`down` goes through the chosen services in reverse dependency order:

- For a process, `down` stops it as [How a process is supervised](#how-a-process-is-supervised)
  says.
- For a running container, `down` runs `podman stop --time 20 <container>`. For a container that
  does not run, `down` prints its state and does nothing. The CLI never removes a container.
- For a named `check` or `external` service, `down` prints `nothing to stop`.

If a stop fails, `down` prints a `FAILED` line and goes on with the other services. At the end,
`down` exits 1.

`restart` stops the named services and their dependents with `down`. If a stop fails, `restart`
exits 1 and starts nothing. Otherwise `restart` runs `up` for the same services. `up` also goes
through their dependencies, and it prints `already healthy` for each healthy one.

### `status`

`status` prints one row for each service in the file, disabled services included. The `pid`
column shows the pid of a process that this CLI started, while the process still runs the
service's command. For a running container, it shows the pid that podman reports. Otherwise it
shows `-`. The `port` column shows the `port` of the service, or `-` for a check.

The `health` column depends on the service:

| Service | `health` |
|---|---|
| A `process`, `podman` or `external` service | `ok: <detail>` or `down: <detail>`, for example `ok: HTTP 200` or `down: Connection refused` |
| A `check` service | The verdict of the worker's doctor: `ok: <detail>`, `warn: <detail>` or `fail: <detail>` |
| A disabled service | `disabled` |

`status` never prints a secret and never makes one. If a secret does not exist yet, the row of
the service that needs it says so. `status` exits 0 if `services.yaml` loads.

### `logs`

`logs` prints the log of one service:

- For a process, `logs` prints the last `N` lines of `<state>/<name>/log`. The default is 50.
  With `-f`, `logs` then prints each new line until you press Ctrl-C. If the service has no log
  yet, `logs` exits 1.
- For a container, `logs` runs `podman logs [-f] <container>` on `{podman_bin}` and exits with
  its status. `-n` has no effect. The output of the container is in podman. The service log holds
  only what the CLI did.

### `env`

`env` prints one `export NAME=value` line for each export of each service in the file, enabled
or not, in the order of the file. The values are quoted for `eval "$(...)"`. A later export of
the same name replaces an earlier one.

`env` is the only command that prints a secret. It makes a secret that does not exist yet. If a
placeholder cannot be resolved, `env` prints no line. It writes the reason on stderr and exits 1.

### `doctor`

`doctor` prints these lines, in this order:

1. `uv`: whether `uv` is on PATH or in `~/.local/bin`.
2. Each check of the worker's doctor that no `check` service in `services.yaml` shows. A
   disabled `check` service also counts as one that shows its check. With the template, these
   lines are `runtime` and `keyring`. If the doctor gives no report, this is one failed `worker`
   line that gives the reason.
3. One line for each service, in the order of the file:
   - `skip`, for a disabled service
   - the verdict of the worker's doctor, for a `check` service
   - the health probe and the pid, for each other service

Each line has these parts:

| Part | Content |
|---|---|
| Status word | `ok`, `warn`, `fail` or `skip` |
| Subject | The check or the service |
| Detail | The detail of the check or the probe |

`doctor` starts nothing and makes no secret. If a line is `fail`, `doctor`
exits 1. A `warn` does not change the exit code.

### `init`

`init` does these steps:

1. It writes `services.yaml` from `resources/services/services.yaml`, if the file does not
   exist.
2. It loads and validates `services.yaml`. If the file is invalid, `init` exits 1.
3. It writes these files beside `services.yaml`, each only if it does not exist:
   - `litellm.yaml`, a copy of `resources/services/litellm.yaml`
   - `runstore.json`, from the Run Store template, with the values of this host
     ([`runstore-api`](#runstore-api))
   - `rl_driver.json`, from the RL driver template, with the values of this host
     ([`rl-driver`](#rl-driver))
4. It makes each secret that a `{secret:NAME}` in `services.yaml` names, if the secret does not
   exist.
5. It prints one line for each file and for each secret: `wrote` or `kept` for a file, and
   `made` or `kept` for a secret. Then it prints the files to edit. It prints no secret value.

### Exit codes and errors

Each command exits with one of these codes:

- 0, if each action succeeded or each report was printed.
- 1, in each of these cases:
  - A start or a stop failed.
  - A doctor check failed.
  - A name is unknown.
  - A placeholder cannot be resolved.
  - `services.yaml` is missing or invalid.
- the status of `podman logs`, for `logs` of a container.
- 2, if the command line itself is invalid. `argparse` prints the usage.

If `services.yaml` does not exist, each command except `init` prints this line and exits 1:
``no <path>: run `python -m sprout.services init` first``. If a command names an unknown service,
it lists the services of the file and exits 1. If you press Ctrl-C, the command prints
`interrupted` and exits 1. `logs -f` exits 0 on Ctrl-C.

## Where things live

The CLI keeps its files under two roots:

| Root | Default | Override |
|---|---|---|
| Config | `$XDG_CONFIG_HOME/sprout`, else `~/.config/sprout` | `SPROUT_SERVICES_CONFIG_DIR` |
| State | `$XDG_STATE_HOME/sprout/services`, else `~/.local/state/sprout/services` | `SPROUT_SERVICES_STATE_DIR` |

If an override variable is set, it comes before the XDG variable. A leading `~` in an override
is the `HOME` of the environment.

The config root holds the files that you edit:

- `services.yaml`
- `litellm.yaml`
- `runstore.json`
- `rl_driver.json`

The state root holds what the programs make:

| Path under the state root | What it holds |
|---|---|
| `<name>/` | The working directory of each command that the CLI runs for the service |
| `<name>/pid` | The pid of a process that the CLI started |
| `<name>/log` | Each command that the CLI ran for the service, and the output of the process |
| `db/data` | The data directory of PostgreSQL |
| `litellm-env/` | The venv of the proxy |
| `runstore-artifacts/` | The `artifact_root` of `runstore.json` |
| `rl-driver/groups.sqlite3` | The ledger of the RL driver |
| `secrets/<name>` | One secret |

### Secrets

Each secret is one file under `<state>/secrets/`. The directory has mode 0700, and each file has
mode 0600. The value is `secrets.token_urlsafe(32)`. A command makes it one time, and each later
command reads it back. The template names these secrets:

- `db-password`
- `litellm-master-key`
- `runstore-token`
- `rl-driver-token`

The commands treat a secret that does not exist yet in two ways:

| Commands | A secret that does not exist yet |
|---|---|
| `init`, `up`, `env` | The command makes it. Each time the command reads a secret, it also sets both modes again. |
| `down`, `status`, `doctor`, `logs` | The command reports an error for each service that needs the secret. These commands only read secrets. |

### The service log

The CLI writes each command that it runs for a service to `<state>/<name>/log`. A timestamped
line gives the argv, and the output of the command follows. These commands go to the log:

- the start of a process
- each `prepare` step
- `podman run`
- `podman start`
- `podman stop`

The CLI does not log `podman inspect` or `podman logs`. A process writes its stdout and stderr
to the same log. An argv holds variable names only, because podman reads the value of each
`-e NAME` from the environment. So the CLI writes no secret value to the log.

### Tests and a second instance

The `paths` fixture of `tests/services/conftest.py` sets `HOME` and both override variables to
directories under the `tmp_path` of the test. So a services test cannot touch your home. To run
a second instance of every service, set both override variables to other directories. Then edit
the ports in the `services.yaml` of that instance.

## How experiments consume it

An experiment shell evaluates the output of `env`, then passes the variables to its commands:

```bash
eval "$(python -m sprout.services env)"
python -m examples.qwen.run ... --model luna \
    --model-endpoint "$OPENAI_BASE_URL" --model-key-env SPROUT_LITELLM_KEY --structured-review
```

The worker also reads the model key. [The key](MODEL_ENDPOINTS.md#the-key-of-each-request) says
how the key reaches the worker.

`env` exports these variables with the template:

| Variable | From | Value |
|---|---|---|
| `SPROUT_PODMAN_BIN` | `podman` | The `podman_bin` of the doctor report: `$SPROUT_PODMAN_BIN`, else `~/.local/opt/podman/bin/podman` from `ops/setup_podman.sh` if it is executable, else `podman` |
| `SPROUT_PODMAN_STORE` | `store` | The `store` of the doctor report: `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs` |
| `SPROUT_RUNSTORE_DSN` | `db` | `host=127.0.0.1 port=15432 dbname=sprout_runstore user=sprout_runstore password=<db-password>` |
| `SPROUT_RUNSTORE_TOKEN` | `runstore-api` | The `runstore-token` secret. A client sends it as `Authorization: Bearer <token>`. |
| `SPROUT_RL_DRIVER_TOKEN` | `rl-driver` | The `rl-driver-token` secret |
| `OPENAI_BASE_URL` | `proxy` | `http://127.0.0.1:4000/v1` |
| `OPENAI_API_KEY`, `SPROUT_LITELLM_KEY` | `proxy` | The `litellm-master-key` secret |

Only the services in `services.yaml` contribute. So a host without `rl-driver` in its file
exports no driver token.

## The services

This section gives the facts of each service in the template. `docs/RUNSTORE.md` and
`docs/RL_DRIVER.md` describe the Run Store and the RL driver themselves.

### `db`

`db` is PostgreSQL 16 in a rootless container on `{podman_bin}`, the podman that the sandboxes
run on. The template gives it these values:

| Setting | Value |
|---|---|
| `image` | `docker.io/library/postgres:16-alpine` |
| `container` | `sprout-services-db` |
| `publish` | `127.0.0.1:15432:5432`. The host's own PostgreSQL owns port 5432, so do not use it. |
| `volumes` | `<state>/db/data` bind-mounted at `/var/lib/postgresql/data:Z` |
| `env` | `POSTGRES_USER=sprout_runstore`, `POSTGRES_DB=sprout_runstore`, and `POSTGRES_PASSWORD` from the `db-password` secret |

The probe runs `pg_isready -h 127.0.0.1 -p 15432 -U sprout_runstore` from the client tools of
the host. That proves the published port, which the Run Store connects through. If the host has
no `pg_isready`, the probe runs `podman exec sprout-services-db pg_isready -U sprout_runstore`.

After a reboot, the container exists but is stopped, so `up` runs `podman start`. The data
directory is a bind mount, so it survives `down` and a reboot.

A container belongs to the podman that made it, and the CLI never removes a container. So to
move `db` to another podman, do these steps:

1. Stop the container under the old podman: `<old podman> stop sprout-services-db`.
2. Remove it under the old podman: `<old podman> rm sprout-services-db`.
3. Run `python -m sprout.services up db` in a shell where the worker's doctor finds the new
   podman. For example, set `SPROUT_PODMAN_BIN` to it. The new container mounts the same data
   directory.
4. In the same shell, run `python -m sprout.services restart runstore-worker`. The worker then
   gets the new `SPROUT_PODMAN_BIN`.

### `proxy`

`proxy` is LiteLLM. It runs this command:

```text
<state>/litellm-env/bin/litellm --config <config>/litellm.yaml --host 127.0.0.1 --port 4000
```

The proxy has a venv of its own at `<state>/litellm-env`, because litellm pins other httpx and
pydantic versions than Sprout. Before a start, `up` runs these `prepare` steps:

1. `uv venv --python 3.12 <state>/litellm-env`. The step does not run while
   `<state>/litellm-env/bin/python` exists.
2. `uv pip install --python <state>/litellm-env/bin/python 'litellm[proxy]==1.103.0'`. The step
   does not run while `<state>/litellm-env/bin/litellm` exists.

The 2026-10-09 smoke verified version 1.103.0 ([Model endpoints](MODEL_ENDPOINTS.md)). To add
a provider SDK that litellm does not bundle, do these steps:

1. Add the SDK to the install line of the `proxy` entry in `services.yaml`.
2. If `<state>/litellm-env` exists, remove it. The install step does not run while
   `<state>/litellm-env/bin/litellm` exists.
3. Run `python -m sprout.services restart proxy`. `up` makes the venv again with the SDK.

The proxy gets its keys this way:

- `LITELLM_MASTER_KEY` is the `litellm-master-key` secret, from the `env` of the service.
- The proxy inherits the provider credentials from the shell that runs `up`.
- `unset_env` removes `FORCE_COLOR` from the environment of the proxy, so the log stays plain
  text.

The probe is `GET http://127.0.0.1:4000/health/liveliness`, which must answer 200. `up` waits up
to 120 s (`timeout_s: 120`), because the proxy imports for a while before it listens. The model
aliases are in `~/.config/sprout/litellm.yaml` ([Model endpoints](MODEL_ENDPOINTS.md)). `init`
copies that file from `resources/services/litellm.yaml`.

### `runstore-api`

`runstore-api` serves the Run Store API. Before each start, `up` runs this `prepare` step:

```text
{python} -m sprout.runstore init --config <config>/runstore.json
```

`sprout.runstore init` is idempotent. It refuses a volatile `artifact_root`, so
`python -m sprout.services init` puts the artifact root under the state root. Then `up` runs
this command:

```text
{python} -m sprout.runstore serve --config <config>/runstore.json --host 127.0.0.1 --port 18110
```

`runstore-api` depends on `db`. The probe is `GET http://127.0.0.1:18110/v1/jobs` with
`Authorization: Bearer <runstore-token>`, and it must answer 200. The API has no `/health`
route. If the probe gets 401, the API runs with a different token.

`python -m sprout.services init` writes `runstore.json` from
`resources/runstore/config.example.json`, with these values of the host:

- `artifact_root`: `<state>/runstore-artifacts`
- each `runtime_bin`: `runtime/sprout-runtime` of the checkout
- the `python` of each profile: the interpreter of `{python}`

### `runstore-worker`

`runstore-worker` runs this command:

```text
{python} -m sprout.worker start --config <config>/runstore.json --concurrency 1
```

To change the concurrency, edit the command in `services.yaml`. The worker depends on three
services ([The services and their dependencies](#the-services-and-their-dependencies)). Its probe
is `process`: the worker is up after it has
stayed alive for 2 s. After that, the probe passes while the pid runs the worker's command.

The worker checks the profiles of `runstore.json` before it registers
([Run Store](RUNSTORE.md#checks-at-the-start)). If a check fails, the worker exits at once with
`Profile <name>: <reason>`, and `up` reports the service down. The service log has the reason.

### `rl-driver`

`rl-driver` runs this command:

```text
{python} -m sprout.rl_driver --config <config>/rl_driver.json --host 127.0.0.1 --port 11001
```

The driver depends on `runstore-api`. The probe is `GET http://127.0.0.1:11001/health` with
`Authorization: Bearer <rl-driver-token>`, and it must answer 200. The driver checks the token on
each route, `/health` included. So a probe without the token reads a healthy driver as down.

`python -m sprout.services init` writes `rl_driver.json` from
`resources/rl_driver/config.example.json`, with these values:

- `ledger`: `<state>/rl-driver/groups.sqlite3`
- `driver_token_env`: `SPROUT_RL_DRIVER_TOKEN`, the variable that the driver receives from its
  own `exports`

### `podman` (check only)

`podman` shows the `podman` check of the worker's doctor: the podman of the sandboxes is
executable and answers `--version`. Nothing starts it. Without a `path`, the doctor checks the
first of these that applies:

1. `$SPROUT_PODMAN_BIN`, if it is set
2. the podman 6 that `ops/setup_podman.sh` installs at `~/.local/opt/podman/bin/podman`, if it is
   executable
3. `podman` on PATH

A `path` names another podman to check. The value of `{podman_bin}` stays the podman of the
default report.

`podman` exports `SPROUT_PODMAN_BIN` as `{podman_bin}`. So these two run the podman that the
setup script verified:

- an experiment shell that ran `eval "$(python -m sprout.services env)"`
- the Run Store worker, which depends on this check in the template

A `services.yaml` from before this check existed does not have it, because `init` never
overwrites the file. Until you add it, `doctor` shows the doctor's `podman` check among its host
lines. To add the check, do these steps:

1. Copy the `podman` entry of `resources/services/services.yaml` into your `services.yaml`.
2. Copy the `exports` of the `store` entry of the template into your `store` entry.
3. Add `podman` and `store` to the `depends_on` of `runstore-worker`.
4. Run `python -m sprout.services restart runstore-worker`. The worker then gets
   `SPROUT_PODMAN_BIN` and `SPROUT_PODMAN_STORE`.

### `store` (check only)

`store` shows the `store` check of the worker's doctor. Nothing starts it. The store is
`$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`. A `path` names another store to check.

The check asks the same questions as `BtrfsStore`, which `PodmanPool` opens. It reports a problem
in the same words. You make the store one time, as root:
`sudo ops/btrfs_store.sh <mount point> <image file> <size>`.

`store` exports `SPROUT_PODMAN_STORE` as `{store}`.

### `aenv` (external, disabled)

`aenv` is the AgentENV microVM server for `--backend microvm`. The template disables it with
`enabled: false`. The probe is `GET $AENV_SERVER_URL/sandboxes` with
`X-API-Key: $AENV_API_KEY`, and it must answer 200.

The server runs as root, so the CLI never starts it. If the probe fails, `up aenv` prints the
`start_hint` and exits 1. The hint of the template gives these steps:

1. If the `ublk_drv` module is gone after a reboot, rebuild it against the running kernel with
   `make -C /usr/src/kernels/$(uname -r) M=$PWD modules` in its source directory. Then install it
   and load it with `modprobe ublk_drv`.
2. Start the server: `sudo setsid env ... /opt/aenv-bench/bin/server`.
3. Ask its `/sandboxes` with `curl` and the API key.

Replace `<ublk_drv source directory>` in the hint with the directory of the module source on your host.

## The worker's host checks

`python -m sprout.worker doctor` checks the machine that a worker runs on. It starts nothing and
changes nothing. It runs these checks, in this order:

| Check | Passes when | Otherwise |
|---|---|---|
| `podman` | The podman is executable, and `<podman> --version` exits 0 within 15 s. The podman is `--podman-bin`, else `$SPROUT_PODMAN_BIN`, else `~/.local/opt/podman/bin/podman` if it is executable, else `podman` on PATH. | `fail` |
| `store` | The store is a directory, `btrfs` is on PATH, and `findmnt` reports a btrfs mount with `user_subvol_rm_allowed`. The store is `--store`, else `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs`. `PodmanPool` accepts the same stores. | `fail`, in the words of the pool |
| `runtime` | The runtime binary is a file. It is `--runtime-bin`, else `runtime/sprout-runtime` of the checkout. | `fail`, with the build command |
| `keyring` | `kernel.keys.maxkeys` is 20000 or more. | `warn`. Below 20000, rootless podman starts fail at about 180 concurrent containers. The fix is `sudo sysctl kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000`. If the value is not readable, the check also gives `warn`. |

The doctor prints one line for each check, with these parts:

| Part | Content |
|---|---|
| Status | `ok`, `warn` or `fail` |
| Check | `podman`, `store`, `runtime` or `keyring` |
| Detail | What the check found |

With `--json`, the doctor prints one JSON object instead:

```json
{
  "podman_bin": "/path/to/.local/opt/podman/bin/podman",
  "store": "/path/to/sprout-btrfs",
  "checks": {
    "podman": {"status": "ok", "detail": "podman 6.1.3 (/path/to/.local/opt/podman/bin/podman)"},
    "store": {"status": "ok", "detail": "/path/to/sprout-btrfs: btrfs, user_subvol_rm_allowed"},
    "runtime": {"status": "ok", "detail": "/path/to/sprout/runtime/sprout-runtime"},
    "keyring": {"status": "warn", "detail": "kernel.keys.maxkeys=200 < 20000: ..."}
  }
}
```

`podman_bin` and `store` are the podman and the store that the checks used. If a check is
`fail`, the doctor exits 1. A `warn` does not change the exit code.

The services CLI reads the doctor by these rules:

- When a command first needs a check or the value of `{podman_bin}` or `{store}`, the CLI runs
  `{python} -m sprout.worker doctor --json` as a child process. The child gets the environment of
  the CLI. The CLI waits up to 120 s for it.
- In one command, the CLI runs the doctor at most one time for each set of flags.
- A `check` service shows the doctor's check of its kind: `check: podman` shows `podman`, and
  `check: btrfs_store` shows `store`. For a service with a `path`, the CLI runs the doctor again
  with `--podman-bin <path>` or `--store <path>`.
- `{podman_bin}` and `{store}` are the `podman_bin` and `store` of the report without flags. A
  `path` does not change them. `env` exports them, and the `db` container runs on that podman.
- If the doctor prints no report, each check fails. An example cause is a `{python}` that cannot
  import `sprout.worker`. The detail gives the exit status of the doctor and its last line on
  stderr. Then `env`, and each command that needs a placeholder, fail with the same message.

## services.yaml

`resources/services/services.yaml` is the template, and its comments explain each entry. The
loader checks the whole file each time a command loads it.

### The top level

The file is a mapping with two keys:

- `python`: the interpreter that `{python}` names. If it is null, `{python}` is the checkout's
  `.venv/bin/python` if that exists, else the interpreter that runs the CLI.
- `services`: a mapping from each service name to its entry. A name must match
  `^[a-z][a-z0-9-]*$`.

### Kinds

Each service has one of these kinds:

| Kind | What the CLI does | Keys |
|---|---|---|
| `process` | Starts the command detached, with its pid and log under `<state>/<name>`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `command`, `env`, `unset_env`, `prepare`, `timeout_s` |
| `podman` | Runs a container on `{podman_bin}`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `image`, `container`, `publish`, `volumes`, `env`, `timeout_s` |
| `external` | Starts nothing. If the probe fails, `up` prints `start_hint`. | `kind`, `enabled`, `depends_on`, `health`, `port`, `exports`, `start_hint` |
| `check` | Starts nothing. Shows a check of the worker's doctor. | `kind`, `enabled`, `check` (`podman` or `btrfs_store`), `path`, `exports` |

### Keys

| Key | Meaning |
|---|---|
| `enabled` | Default `true`. If `false`, a bare `up` leaves the service out, `status` shows `disabled`, and `doctor` prints `skip`. A command that names the service still acts on it. |
| `depends_on` | The services that `up` starts first. The service receives their exports. |
| `port` | Informational. `status` shows it. |
| `timeout_s` | How long `up` waits for the probe, in seconds. The default is 60. |
| `exports` | Variables that `env` prints. The service and its dependents also receive them. |
| `env` | Variables of this process or container only. |
| `unset_env` | Variables that the CLI removes from the environment of the process. |
| `prepare` | Steps of the form `{run: [argv...], unless: <path>}`. Before a start, `up` runs each step whose `unless` path does not exist. |
| `command` | The argv of the process. |
| `image`, `container` | The image reference and the container name. Every podman verb finds the container by this name. |
| `publish`, `volumes` | The `-p` and `-v` values of `podman run`. Before `podman run`, the CLI makes the host side of each `volumes` entry that starts with `/`. |
| `start_hint` | The text that `up` prints for an `external` service. |
| `check`, `path` | The doctor check to show, and the podman or store to check instead of the default. |
| `health` | The probe. [Health kinds](#health-kinds) gives its keys. |

### Health kinds

Each `health` mapping has one of these kinds. A service without `health` has the kind `none`.

| Kind | Passes when | Keys |
|---|---|---|
| `http` | A GET of `url` answers 200 within 3 s. | `url` (required), `headers` |
| `pg_isready` | The `pg_isready -h <host> -p <port>` of the host exits 0 within 15 s. It gets `-U <user>` if `user` is set. If the host has no `pg_isready`, the CLI runs `pg_isready` in `container` with `podman exec`. | `port` (required), `host` (default `127.0.0.1`), `user`, `container` (default: the `container` of the service) |
| `process` | For a process: the pid in the pid file is alive and runs the service's command. During `up`, the process must also stay alive for 2 s. For a container: the container runs. | none |
| `none` | For a process or a container: the same as `process`. For an `external` service: always. | none |

### Placeholders

The CLI resolves these placeholders when a command runs:

| Placeholder | Value |
|---|---|
| `{python}` | The interpreter of the `python` key |
| `{repo}` | The checkout that the package was imported from |
| `{config}` | The config root |
| `{state}` | The state root |
| `{podman_bin}` | The podman of the worker's doctor report |
| `{store}` | The store of the worker's doctor report |
| `{secret:NAME}` | The secret `NAME`. A name matches `^[a-z0-9][a-z0-9-]*$`. |
| `{env:NAME}` | The variable `NAME` of the CLI's environment |

The CLI expands placeholders in these fields only:

- `command`
- `env`
- `exports`
- `prepare`: `run` and `unless`
- `volumes`
- `publish`
- `health.url` and `health.headers`
- `path`

The CLI uses each other string as written. These fields are literal:

- `image`
- `container`
- `start_hint`
- `unset_env`
- `health.host`
- `health.user`
- `health.container`

The loader still refuses an unknown placeholder and a secret in `unset_env`. It refuses a brace in
`image`. A `container` must match the name rule of podman, so it cannot hold a placeholder.

After the expansion, a leading `~` is the home directory. For the first word of `command` or of
a `prepare` step, the CLI looks up a bare program name on PATH, then in `~/.local/bin`. The uv
installer puts uv in `~/.local/bin`, and a cron shell does not have that directory on PATH.

A `{secret:NAME}` can appear only in these fields:

- `env`
- `exports`
- `health.headers`

These values become the environment of a process, or `env` prints them when you ask. The loader
refuses a secret in each of these fields, and it names the field:

- `command`
- `unset_env`
- `prepare`
- `volumes`
- `publish`
- `health.url`
- `path`

Most of these fields reach the log or the terminal as written. Also, each user on the host can
read an argv in `/proc/<pid>/cmdline`.

### Validation

The loader checks the whole file before a command acts, so a bad entry fails before `up` starts
anything. The loader refuses each of these problems, and it names the service and the field:

- YAML that does not parse
- a top-level key other than `python` and `services`
- a service name that breaks the name rule
- an unknown kind
- a key that the kind does not take
- an unknown `check`
- an unknown dependency, or a service that depends on itself
- a dependency cycle. The error shows the whole cycle.
- a `port` that is not an integer from 1 to 65535
- a `timeout_s` that is not a positive number
- a `process` without `command`
- a `podman` service without `image`, or without a valid `container`
- an unknown health kind or health key
- an `http` probe without `url`, or a `pg_isready` probe without `port`
- a `prepare` step that is not `{run: [argv...]}` with an optional `unless`
- a value of the wrong type, for example an `enabled` that is not `true` or `false`
- an unknown placeholder
- a `{secret:NAME}` in a field that cannot hold one

A placeholder can be valid in the file but have no value in a run, in these cases:

- `{env:NAME}`, when the variable is not set
- `{secret:NAME}`, when the secret does not exist and the command does not make secrets
- `{podman_bin}` or `{store}`, when the worker's doctor gives no report

In these cases, the commands react in these ways. None of them prints a traceback.

- `up` and `down` print one `FAILED` line. `up` skips the dependents of the failed service.
- `status` shows a `down: ...` cell.
- `env` prints an error line and exits 1.

## How a process is supervised

The CLI starts a process detached from the shell, so a closed terminal does not stop it. `up`
starts a process in these steps:

1. The CLI finds the program. It uses a path as written. It looks up a bare name on PATH, then in
   `~/.local/bin`. If it finds no program, the start fails.
2. The CLI starts the process in a session of its own (`setsid`), with these settings:
   - the working directory `<state>/<name>`, never the checkout. `conmon` can write an `oom`
     marker into its working directory, and it once dirtied the checkout that way.
   - stdin from `/dev/null`
   - stdout and stderr appended to `<state>/<name>/log`
3. The CLI writes the pid to `<state>/<name>/pid`.
4. The CLI runs the probe every 0.5 s. The first of these events ends the wait:
   - The probe passes. `up` prints the pid and the log.
   - The process exits. The start fails at once. The CLI removes the pid file and prints the
     last 20 lines of the log.
   - `timeout_s` runs out. The start fails, and the CLI prints the last 20 lines of the log. The
     process keeps running, so that you can look at it with `logs <name>`, then stop it with
     `down <name>`.

For a probe of kind `process` or `none`, the probe passes after the process has stayed alive for
2 s. A worker that fails on its config exits within the first second.

A later command trusts the pid file only while both of these are true:

- The process exists and is not a zombie.
- `/proc/<pid>/cmdline` holds each argument of the service's expanded command. If the command
  has no arguments, the base names of the two programs must match.

The CLI does not compare the program itself, because the `#!` line of a venv script runs it under
python. So the CLI never signals a recycled pid. After a reboot, a stale pid file reads as
`down`, and the next `down` removes it without a signal.

`down` stops a process in these steps:

1. If the pid file does not name the service's process, `down` removes the pid file and prints
   `not running`. It signals nothing.
2. `down` sends SIGTERM to the process group of the process.
3. `down` waits up to 20 s. If the process is gone, `down` removes the pid file and prints
   `stopped`.
4. `down` sends SIGKILL and waits up to 5 s. If the process is gone, `down` removes the pid file
   and prints `killed`.
5. If the process survives SIGKILL, `down` prints a `FAILED` line and exits 1.

`down` needs the expanded command to prove that a pid belongs to the service. If a placeholder in
`command` has no value, `down` signals nothing and prints a `FAILED` line.
