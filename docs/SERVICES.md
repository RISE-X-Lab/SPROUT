# Services

`python -m sprout.services` manages the persistent resources an experiment host
needs, so nothing is hand-started before a batch again: the Run Store's
PostgreSQL (a rootless podman container), the LiteLLM proxy in front of the
providers Sprout cannot talk to directly, the Run Store API and its worker, the
RL driver -- plus three things it only checks: the podman the sandboxes run on, its btrfs store and, optionally, the AgentENV microVM server. One file, `~/.config/sprout/services.yaml`, says what
they are; the code knows how to start a process or a container and how to probe
it, nothing about any service in particular. The package is stdlib + PyYAML and
imports nothing else from Sprout (`tests/harness/test_layering.py`), so it
works before the extras are installed and never inherits a fault from what it
starts.

```bash
python -m sprout.services init             # services.yaml, litellm.yaml, runstore.json, rl_driver.json; the secrets
python -m sprout.services up               # db, proxy, runstore-api, runstore-worker, rl-driver -- dependencies first
eval "$(python -m sprout.services env)"    # the sandbox podman and store, the DSN, the tokens and the proxy URL + key, in this shell
python -m sprout.services status           # name | kind | pid | health | port
python -m sprout.services doctor           # every check and probe, starting nothing; exit 1 when something is wrong
```

## Commands

| Command | What it does |
|---|---|
| `init` | Writes `services.yaml` from the packaged template, copies `litellm.yaml`, `runstore.json` (artifact root under state, the checkout's runtime binary and interpreter filled in) and `rl_driver.json` (ledger under state, the generated token in force) beside it when absent -- never overwriting -- and generates every `{secret:NAME}` the spec names. Prints what it wrote and what to edit. |
| `up [NAME...]` | Resolves dependencies in topological order and starts only what is not already healthy: a service whose probe passes is reported `already healthy` and left alone (a proxy started by hand on :4000 included). A `prepare` step runs first (the proxy's venv on the first `up`, `sprout.runstore init` before every API start). A dependent of a failed service is skipped, not tried. Without names: every `enabled` service. |
| `down [NAME...]` | Stops in reverse order. A named service takes its dependents with it (`down db` also stops the API, the worker and the driver -- they are broken without it). Without names: everything. |
| `restart NAME...` | `down` then `up` for the named services and their dependents. |
| `status` | One row per service: `name \| kind \| pid \| health \| port`. `ok: HTTP 200`, `down: connection refused`, `disabled`; a pid is shown only for a process this CLI started (or a running container). Never prints a secret. |
| `logs NAME [-f] [-n N]` | The service's log (the last 50 lines; `-f` follows). For the container, `podman logs`. |
| `env` | `export NAME=value` lines for every service's `exports`, quoted for `eval`. The only command that prints a secret. |
| `doctor` | The host checks (the podman the sandboxes run on, uv, `runtime/sprout-runtime`, `kernel.keys.maxkeys`), the `check` services (podman, the store) and the health of every enabled service, starting nothing. Exit 1 when anything fails; a keyring quota below 20000 is a `warn` (rootless podman starts fail at about 180 concurrent containers; `sudo sysctl kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000`). |

Exit codes are 0 and 1; output is plain text, one line per action.

## Where things live

| | Path | Moved by |
|---|---|---|
| Config | `~/.config/sprout/` (`$XDG_CONFIG_HOME/sprout`): `services.yaml`, `litellm.yaml`, `runstore.json`, `rl_driver.json` | `SPROUT_SERVICES_CONFIG_DIR` |
| State | `~/.local/state/sprout/services/` (`$XDG_STATE_HOME/sprout/services`): `<name>/pid`, `<name>/log`, `db/data` (PostgreSQL's data directory), `litellm-env/` (the proxy's venv), `runstore-artifacts/`, `rl-driver/groups.sqlite3` | `SPROUT_SERVICES_STATE_DIR` |
| Secrets | `<state>/secrets/<name>`, directory 0700, files 0600: `db-password`, `litellm-master-key`, `runstore-token`, `rl-driver-token`, each `secrets.token_urlsafe(32)` generated once by `init` | with the state |

Every test sets both variables to a `tmp_path`, so no test touches the real
home; a second instance of everything on free ports is the same two variables
plus edited ports.

Every command the CLI runs on a service's behalf -- the start itself, a
`prepare` step, a podman verb -- is logged to `<state>/<name>/log` with its
argv (names only, never a secret value: podman reads `-e NAME` from the CLI's
environment) and its output; a process's stdout and stderr go to the same file.

## How experiments consume it

```bash
eval "$(python -m sprout.services env)"
python -m examples.qwen.run ... --model luna \
    --model-endpoint "$OPENAI_BASE_URL" --model-key-env SPROUT_LITELLM_KEY --structured-review
```

| Variable | From | Value |
|---|---|---|
| `SPROUT_PODMAN_BIN` | podman | the podman the sandboxes run on: `$SPROUT_PODMAN_BIN`, else `~/.local/opt/podman/bin/podman` (`ops/setup_podman.sh`), else `podman` |
| `SPROUT_PODMAN_STORE` | store | the btrfs store: `$SPROUT_PODMAN_STORE`, else `~/sprout-btrfs` |
| `SPROUT_RUNSTORE_DSN` | db | `host=127.0.0.1 port=15432 dbname=sprout_runstore user=sprout_runstore password=<db-password>` |
| `SPROUT_RUNSTORE_TOKEN` | runstore-api | the `runstore-token` secret (`Authorization: Bearer`) |
| `SPROUT_RL_DRIVER_TOKEN` | rl-driver | the `rl-driver-token` secret |
| `OPENAI_BASE_URL` | proxy | `http://127.0.0.1:4000/v1` |
| `OPENAI_API_KEY`, `SPROUT_LITELLM_KEY` | proxy | the `litellm-master-key` secret |

Only services present in `services.yaml` contribute, so a host without the RL
driver exports no driver token. Inside the CLI the same rule feeds the
services themselves: a process receives its own `exports` and those of every
service it depends on, transitively, plus its `env` -- the API gets the DSN
from db, the worker and the driver get the DSN and the token through the API.

## The services

**db** -- PostgreSQL 16 in a rootless container on the same podman the sandboxes run on
(the `podman` check's resolution; a host switching podmans moves this container by hand:
`podman stop` and `podman rm` it under the old one, then `up db` under the new, the data
directory being a bind mount): image `docker.io/library/postgres:16-alpine`,
container `sprout-services-db`, published on `127.0.0.1:15432` (the host's own
PostgreSQL owns 5432 -- stay off it), data in `<state>/db/data` bind-mounted
at `/var/lib/postgresql/data:Z`, `POSTGRES_USER=sprout_runstore`,
`POSTGRES_DB=sprout_runstore`, `POSTGRES_PASSWORD` from the secret. `up` is
`podman run -d` when the container does not exist and `podman start` when it
exists but is stopped (after a reboot); `down` is `podman stop`; the container
is never removed by the CLI. Health: `pg_isready -h 127.0.0.1 -p 15432 -U
sprout_runstore` from the host's client tools (which proves the published port),
falling back to `podman exec sprout-services-db pg_isready`.

**proxy** -- LiteLLM, `litellm --config ~/.config/sprout/litellm.yaml --host
127.0.0.1 --port 4000`, from a venv of its own at `<state>/litellm-env` because
it pins another httpx and pydantic than Sprout; the first `up` makes it with
`uv venv --python 3.12` and `uv pip install 'litellm[proxy]==1.103.0'` (the
version the 2026-10-09 smoke verified; a provider SDK litellm does not bundle is
appended to that line in `services.yaml`; `docs/MODEL_ENDPOINTS.md`).
`LITELLM_MASTER_KEY` is the generated secret; provider credentials are inherited
from the shell that runs `up`, and `FORCE_COLOR` is removed from the proxy's
environment (`unset_env`) so the log stays plain text. Health: `GET /health/liveliness` is 200; `up` waits up to
120 s for it (the proxy imports for a while before it listens). The aliases
(`luna`, `sonnet5`, `kimi-k3`) are in `~/.config/sprout/litellm.yaml`, seeded
from `resources/services/litellm.yaml`.

**runstore-api** -- `python -m sprout.runstore serve --config
~/.config/sprout/runstore.json --host 127.0.0.1 --port 18110`, after
`python -m sprout.runstore init --config ...` (idempotent; refuses a volatile
`artifact_root`, so `init` puts it under state). Depends on db. Health: `GET
/v1/jobs` with the bearer token is 200 -- there is no `/health` route, and a
401 says the token is not the one the API was started with.
**runstore-worker** -- `python -m sprout.runstore worker --config ...
--concurrency 1` (edit the concurrency in `services.yaml`); depends on the
API; health is the process being alive. **rl-driver** -- `python -m
sprout.rl_driver --config ~/.config/sprout/rl_driver.json --host 127.0.0.1
--port 11001`; depends on the API; health `GET /health`. `docs/RUNSTORE.md` and
`docs/RL_DRIVER.md` describe the three; their config files are the packaged
`config.example.json` of each package with this host's paths filled in.

**podman** (check only) -- the podman the sandboxes run on answers `--version`:
`$SPROUT_PODMAN_BIN`, else the podman 6 `ops/setup_podman.sh` installs under
`~/.local/opt/podman`, else `podman` on PATH. It exports the resolved path as
`SPROUT_PODMAN_BIN` (the `{podman_bin}` placeholder), so an experiment shell that
ran `eval "$(python -m sprout.services env)"` and the Run Store worker (which
depends on this check in the template) run the same podman as the setup script
verified. A `services.yaml` written before this check existed must gain it by hand
(`init` never overwrites): copy the `podman` entry and the `exports` of `store` from
`resources/services/services.yaml`, add `podman, store` to the worker's
`depends_on`, then `restart runstore-worker` so the worker inherits the two exports.

**store** (check only) -- the podman backend's store, `$SPROUT_PODMAN_STORE`
else `~/sprout-btrfs`, must be a btrfs mount with `user_subvol_rm_allowed`
(`findmnt -no FSTYPE,OPTIONS --target <store>`, the same question
`PodmanPool` asks); made once as root with `sudo ops/btrfs_store.sh`. Exports
the resolved path as `SPROUT_PODMAN_STORE` (the `{store}` placeholder).

**aenv** (external, `enabled: false` in the template) -- the AgentENV microVM
server for `--backend microvm`. Health: `GET $AENV_SERVER_URL/sandboxes` with
`X-API-Key: $AENV_API_KEY` is 200. It runs as root, so `up aenv` prints the
start command from `AGENTS.md` ("Recovering the AgentENV server after a
reboot": rebuild `ublk_drv` against the running kernel if the module is gone,
then `sudo setsid env ... /opt/aenv-bench/bin/server`) and exits 1 until it
answers; nothing is run.

## services.yaml

The template (`resources/services/services.yaml`) is commented in
full. Kinds: `process` (started detached), `podman` (a container), `external`
(started by hand; `up` prints `start_hint`), `check` (a host fact). Health
kinds: `http` (a GET must answer 200, optional `headers`), `pg_isready`,
`process` (the pid is alive and is ours), `none`. Placeholders: `{python}`
(the `python:` key; null is the checkout's `.venv/bin/python`), `{repo}`,
`{config}`, `{state}`, `{podman_bin}`, `{store}`, `{secret:NAME}`, `{env:NAME}`, in every string but a
container's `image` and `container` (the name is what every podman verb looks
the container up by, so it is one literal); a leading `~` is the home
directory; a bare program name is looked up on PATH, then in `~/.local/bin`.
A `{secret:NAME}` may stand only in `env`, `exports` and `health.headers` --
values handed to a process as its environment or printed by `env` on request
-- never in `command`, `prepare`, `volumes`, `publish`, `health.url` or
`path`, which are logged and echoed as typed (and an argv is readable by every
user in `/proc/<pid>/cmdline`); the loader refuses it, naming the field. Per
service: `depends_on`, `port` (informational), `timeout_s`
(how long `up` waits for health; default 60), `exports`, `env`, `unset_env`,
`prepare` (steps `{run: [...], unless: <path>}`, skipped while the path
exists), and the kind's own fields (`command`; `image`, `container`,
`publish`, `volumes`; `start_hint`; `check`, `path`). `enabled: false` leaves
a service out of a bare `up`, `status` and `doctor`; naming it still acts on
it. The file is validated when loaded -- an unknown dependency, a cycle, an
unknown kind or key, a non-integer port, a malformed or misplaced placeholder
-- with the service and field named. A placeholder the file allows but a run
cannot resolve (an unset `{env:NAME}`, a secret `init` has not made) is one
`FAILED` line from `up`/`down` and a `down: ...` cell in `status`, never a
traceback.

## How a process is supervised

A process starts in a session of its own (fork + `setsid`), from its state
directory (never from the checkout: conmon's `oom` marker once dirtied the
tree), with stdin from `/dev/null` and both streams appended to its log; `up`
then polls the health probe every 0.5 s until it passes, the process exits
(reported at once, with the log's last lines) or `timeout_s` runs out (the
process is left running for you to look at). A `health: process` service is
up once it has stayed alive two seconds. The pid file is believed only while
`/proc/<pid>/cmdline` still carries the arguments the service was started
with, so a recycled pid is never mistaken for ours and a stale file after a
reboot reads as `down`, is removed on the next `down`, and is never signalled.
`down` sends SIGTERM to the process group, waits up to 20 s, then SIGKILL.
