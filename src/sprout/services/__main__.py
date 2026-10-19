"""``python -m sprout.services``: argument parsing, dispatch and exit codes.

A command exits 0 when every action succeeded or every report was printed. It
exits 1 when a start or a stop failed, a doctor check failed, a name is unknown,
a placeholder cannot be resolved, or ``services.yaml`` is missing or invalid.
``logs`` of a container exits with the code of ``podman logs``. The output is one
line per action, without colour. Only ``env`` prints a secret.

:class:`Manager` acts on each kind of service: a process through the supervisor,
a container through podman, an external service as a printed hint, and a check
as a verdict of the worker's doctor. docs/SERVICES.md describes the commands.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import sprout.services
from sprout.resource import resource
from sprout.services import health, secrets
from sprout.services.podman import Podman
from sprout.services.render import StatusRow, doctor_line, env_lines, status_table
from sprout.services.spec import (
    Context,
    ExpandError,
    Service,
    Spec,
    SpecError,
    expand,
    expand_all,
    load_spec,
    template_path,
)
from sprout.services.state import Paths
from sprout.services.supervisor import (
    StartError,
    process_state,
    resolve_program,
    run_logged,
    start,
    stop,
    tail,
    wait_for,
)

Say = Callable[[str], None]


def _one_line(text: str, limit: int = 160) -> str:
    """``text`` as one line of at most ``limit`` characters. The service log keeps the whole argv."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


#: A service of health kind ``process`` or ``none`` is up once it has stayed alive this long, in
#: seconds. A worker that dies on its config does so within the first second.
SETTLE_S = 2.0
#: The templates of the per-service config files that ``init`` writes beside services.yaml.
RUNSTORE_TEMPLATE = resource("runstore/config.example.json")
RL_DRIVER_TEMPLATE = resource("rl_driver/config.example.json")


class Manager:
    """Acts on the services of one spec, on the host that ``paths`` describes.

    ``ctx`` creates a missing secret, and ``up`` uses it. ``read_ctx`` only reads secrets. ``status_row``,
    ``doctor``, ``down`` and ``podman`` use ``read_ctx``, so they never write a secret.
    """

    def __init__(self, spec: Spec, paths: Paths, environ: Mapping[str, str] | None = None) -> None:
        self.spec = spec
        self.paths = paths
        self.environ = dict(os.environ if environ is None else environ)
        python = spec.python or paths.default_python()
        self._worker_doctor = health.WorkerDoctor(python=python, environ=self.environ)
        self.ctx = Context(
            paths=paths,
            python=python,
            secret=lambda name: secrets.ensure(paths, name),
            host=self._worker_doctor.fact,
            environ=self.environ,
        )
        self.read_ctx = Context(
            paths=paths,
            python=python,
            secret=lambda name: secrets.require(paths, name),
            host=self._worker_doctor.fact,
            environ=self.environ,
        )

    # --- what a service is made of -----------------------------------------
    def command(self, service: Service, ctx: Context) -> list[str]:
        """The service's ``command`` with its placeholders expanded. Raises :class:`ExpandError`."""
        return [expand(token, ctx) for token in service.command]

    def environment(self, service: Service, ctx: Context) -> dict[str, str]:
        """The CLI's environment minus ``unset_env``, plus the exports it consumes, plus ``env``.

        Raises :class:`ExpandError` when a placeholder cannot be resolved.
        """
        env = {name: value for name, value in self.environ.items() if name not in service.unset_env}
        env.update(expand_all(self.spec.exports_for(service), ctx))
        env.update(expand_all(service.env, ctx))
        return env

    def podman(self, service: Service) -> Podman:
        """The container's verbs on the podman the sandboxes run on.

        Raises :class:`ExpandError` when the worker's doctor names no podman.
        """
        return Podman(
            self.paths.log_file(service.name),
            self.paths.service(service.name),
            self.environ,
            podman_bin=expand("{podman_bin}", self.read_ctx),
        )

    def check(self, service: Service) -> health.Probe:
        """The worker doctor's verdict on a ``check`` service, never an exception."""
        try:
            path = expand(service.path, self.read_ctx) if service.path else ""
        except ExpandError as error:  # an unset {env:...} in `path`
            return health.Probe(False, str(error))
        return self._worker_doctor.check(kind=service.check, path=path)

    # --- what is running ---------------------------------------------------
    def liveness(self, service: Service, ctx: Context) -> tuple[bool | None, str, str]:
        """``(alive, why, pid text)`` of a process or a container. Other kinds give ``(None, "", "-")``."""
        if service.kind == "process":
            state = process_state(self.paths, service.name, self.command(service, ctx))
            return state.alive, state.detail, str(state.pid) if state.alive else "-"
        if service.kind == "podman":
            container = self.podman(service).inspect(service.container)
            if not container.exists:
                return False, "no container %s" % service.container, "-"
            return (
                container.running,
                "container %s %s" % (service.container, container.status),
                str(container.pid or "-"),
            )
        return None, "", "-"

    def probe(self, service: Service, ctx: Context) -> tuple[health.Probe, str]:
        """``(health, pid text)`` of the service. Never raises :class:`ExpandError`.

        A placeholder that this run cannot resolve, such as an unset ``{env:NAME}`` in the command,
        gives a failed probe that names it. So one broken entry does not hide the rest of a report.
        """
        try:
            alive, detail, pid = self.liveness(service, ctx)
        except ExpandError as error:
            return health.Probe(False, str(error)), "-"
        return health.probe(service.health, ctx, alive=alive, alive_detail=detail), pid

    # --- up ----------------------------------------------------------------
    def up(self, service: Service, say: Say) -> bool:
        """Bring one service up, and return True when it is healthy at the end."""
        name = service.name
        if service.kind == "check":
            result = self.check(service)
            say("%s: %s %s" % (name, result.word, result.detail))
            return result.ok
        probe, _ = self.probe(service, self.ctx)
        if probe.ok:
            say("%s: already healthy (%s)" % (name, probe.detail))
            return True
        if service.kind == "external":
            say("%s: not running (%s); it needs root, start it by hand:" % (name, probe.detail))
            say(service.start_hint.rstrip("\n") or "(no start_hint in services.yaml)")
            return False
        try:
            if service.kind == "podman":
                self._up_container(service, say)
            else:
                self._up_process(service, say, probe.detail)
        except (StartError, ExpandError) as error:
            # An unset {env:NAME} is the same kind of fault as a failed start:
            # one line naming the service, and its dependents are skipped.
            say("%s: FAILED: %s" % (name, error))
            return False
        return True

    def _up_process(self, service: Service, say: Say, unhealthy: str) -> None:
        name = service.name
        alive, detail, _ = self.liveness(service, self.ctx)
        if alive:
            # A second instance would fail on the port and hide the real fault.
            raise StartError(
                "%s is running but not healthy (%s); `logs %s`, then `restart %s`"
                % (detail, unhealthy, name, name)
            )
        env = self.environment(service, self.ctx)
        directory = self.paths.service(name)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = self.paths.log_file(name)
        for step in service.prepare:
            if step.unless and Path(expand(step.unless, self.ctx)).exists():
                continue
            argv = [expand(token, self.ctx) for token in step.run]
            argv[0] = resolve_program(argv[0])
            say("%s: preparing: %s" % (name, _one_line(shlex.join(argv))))
            status = run_logged(argv, env, log_path, directory)
            if status != 0:
                raise StartError(
                    "prepare step exited %d; last lines of %s:\n%s" % (status, log_path, tail(log_path))
                )
        command = self.command(service, self.ctx)
        say("%s: starting (argv and output in %s)" % (name, log_path))
        if service.health.kind in ("process", "none"):
            started = time.monotonic()

            def probe() -> health.Probe:
                settled = time.monotonic() - started
                return health.Probe(settled >= SETTLE_S, "alive for %.1f s" % settled)
        else:

            def probe() -> health.Probe:
                return health.probe(service.health, self.ctx, alive=True, alive_detail="")

        pid = start(name, command, env, self.paths, probe=probe, timeout_s=service.timeout_s)
        say("%s: up (pid %d, log %s)" % (name, pid, log_path))

    def _up_container(self, service: Service, say: Say) -> None:
        name, container = service.name, service.container
        podman = self.podman(service)
        state = podman.inspect(container)
        if state.running:
            say("%s: container %s is running, waiting for it to answer" % (name, container))
        elif state.exists:
            say("%s: starting container %s (podman start)" % (name, container))
            podman.start(container)
        else:
            say("%s: creating container %s from %s (podman run)" % (name, container, service.image))
            podman.run(service, self.ctx)

        def running() -> str | None:
            current = podman.inspect(container)
            return None if current.running else "container %s is %s" % (container, current.status)

        result, elapsed = wait_for(
            lambda: health.probe(service.health, self.ctx, alive=True, alive_detail=""),
            service.timeout_s,
            still_running=running,
        )
        if not result.ok:
            raise StartError(
                "container %s is not healthy after %.0f s (%s); see `podman logs %s`"
                % (container, elapsed, result.detail, container)
            )
        say("%s: up (container %s, %s)" % (name, container, result.detail))

    # --- down --------------------------------------------------------------
    def down(self, service: Service, say: Say) -> bool:
        """Stop one managed service, and return False when the stop failed. Other kinds only print a line."""
        name = service.name
        if not service.managed:
            say("%s: %s, nothing to stop" % (name, service.kind))
            return True
        if service.kind == "podman":
            try:
                podman = self.podman(service)
            except ExpandError as error:
                say("%s: FAILED: %s" % (name, error))
                return False
            state = podman.inspect(service.container)
            if not state.running:
                say(
                    "%s: container %s %s"
                    % (name, service.container, "is %s" % state.status if state.exists else "does not exist")
                )
                return True
            say("%s: stopping container %s (podman stop)" % (name, service.container))
            try:
                podman.stop(service.container)
            except StartError as error:
                say("%s: FAILED: %s" % (name, error))
                return False
            say("%s: stopped" % name)
            return True
        try:
            # The command proves that a pid is ours. Without the command, nothing is signalled.
            outcome = stop(self.paths, name, self.command(service, self.read_ctx))
        except (RuntimeError, ExpandError) as error:
            say("%s: FAILED: %s" % (name, error))
            return False
        say("%s: %s" % (name, outcome))
        return True

    # --- reports -----------------------------------------------------------
    def status_row(self, service: Service) -> StatusRow:
        """The service's row of the ``status`` table."""
        port = str(service.port) if service.port else "-"
        if not service.enabled:
            return StatusRow(service.name, service.kind, "-", "disabled", port)
        if service.kind == "check":
            result = self.check(service)
            return StatusRow(service.name, service.kind, "-", "%s: %s" % (result.word, result.detail), "-")
        probe, pid = self.probe(service, self.read_ctx)
        text = ("ok" if probe.ok else "down") + (": %s" % probe.detail if probe.detail else "")
        return StatusRow(service.name, service.kind, pid, text, port)

    def doctor(self, say: Say) -> bool:
        """Print every host check and the probe of every enabled service, and start nothing.

        A disabled service gets a ``skip`` line. True when nothing failed.
        """
        fine = True
        # The worker's checks that no `check` service names, enabled or not, such as the podman
        # check for a services.yaml written before that service existed.
        kinds = [service.check for service in self.spec.services if service.kind == "check"]
        host = [("uv", health.program("uv")), *self._worker_doctor.other_checks(kinds)]
        for subject, result in host:
            say(doctor_line(result.word, subject, result.detail))
            fine = fine and (result.ok or result.warning)
        for service in self.spec.services:
            if not service.enabled:
                say(doctor_line("skip", service.name, "disabled in services.yaml"))
                continue
            if service.kind == "check":
                result = self.check(service)
            else:
                result, pid = self.probe(service, self.read_ctx)
                if pid != "-":
                    result = health.Probe(result.ok, "%s (pid %s)" % (result.detail, pid), result.warning)
            say(doctor_line(result.word, service.name, result.detail))
            fine = fine and (result.ok or result.warning)
        return fine


# --- init ---------------------------------------------------------------------


def _seed(path: Path, text: str, say: Say) -> bool:
    """Write ``text`` to ``path`` when no file is there, and return True. An existing file is kept."""
    if path.exists():
        say("kept   %s (exists)" % path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    say("wrote  %s" % path)
    return True


def _rewrite(value: Any, key: str, replacement: Any) -> Any:
    """``value`` with each ``key`` at any depth set to ``replacement``.

    The JSON template holds the example paths of another host.
    """
    if isinstance(value, dict):
        return {k: (replacement if k == key else _rewrite(v, key, replacement)) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, key, replacement) for item in value]
    return value


def _runstore_config(paths: Paths, python: str) -> str:
    """The Run Store template with the values of this host: the artifact root under state, the
    checkout's runtime binary, and ``python`` for every profile.

    ``runstore init`` refuses a volatile artifact root. The worker runs each job of a profile as
    ``<python> -m sprout.worker.jobs.rollout``, so that interpreter must import sprout."""
    config = json.loads(RUNSTORE_TEMPLATE.read_text(encoding="utf-8"))
    config["artifact_root"] = str(paths.state / "runstore-artifacts")
    config = _rewrite(config, "runtime_bin", str(paths.runtime_binary()))
    config["profiles"] = _rewrite(config.get("profiles", {}), "python", python)
    return json.dumps(config, indent=2) + "\n"


def _rl_driver_config(paths: Paths) -> str:
    """The RL driver template with its ledger under state and its token from ``SPROUT_RL_DRIVER_TOKEN``."""
    config = json.loads(RL_DRIVER_TEMPLATE.read_text(encoding="utf-8"))
    config["ledger"] = str(paths.state / "rl-driver" / "groups.sqlite3")
    config["driver_token_env"] = "SPROUT_RL_DRIVER_TOKEN"
    return json.dumps(config, indent=2) + "\n"


def _cmd_init(args: argparse.Namespace, paths: Paths) -> int:
    say = print
    paths.config.mkdir(parents=True, exist_ok=True)
    paths.state.mkdir(parents=True, exist_ok=True)
    _seed(paths.spec, template_path("services.yaml").read_text(encoding="utf-8"), say)
    spec = _load(paths)
    if spec is None:
        return 1
    python = spec.python or paths.default_python()
    _seed(paths.config / "litellm.yaml", template_path("litellm.yaml").read_text(encoding="utf-8"), say)
    _seed(paths.config / "runstore.json", _runstore_config(paths, python), say)
    _seed(paths.config / "rl_driver.json", _rl_driver_config(paths), say)
    for name in spec.secret_names():
        existed = secrets.exists(paths, name)
        secrets.ensure(paths, name)
        say("%s secret %s" % ("kept  " if existed else "made  ", paths.secrets / name))
    say("")
    say("edit   %s: the model aliases the experiments use" % (paths.config / "litellm.yaml"))
    say(
        "edit   %s: profiles (the podman store, snapshot_repository, sandbox_resources, worker_env)"
        % (paths.config / "runstore.json")
    )
    say("edit   %s: miles.tasks, when the RL driver is used" % (paths.config / "rl_driver.json"))
    say("then   python -m sprout.services up        # dependencies first; `status` and `doctor` to look")
    say(
        '       eval "$(python -m sprout.services env)"   # the podman and store the sandboxes use, '
        "the DSN, tokens and proxy URL for an experiment shell"
    )
    return 0


# --- the other commands ------------------------------------------------------


def _load(paths: Paths) -> Spec | None:
    """The spec in ``paths.spec``, or None after an error on stderr."""
    if not paths.spec.is_file():
        print("no %s: run `python -m sprout.services init` first" % paths.spec, file=sys.stderr)
        return None
    try:
        return load_spec(paths.spec)
    except SpecError as error:
        print(error, file=sys.stderr)
        return None


def _known(spec: Spec, names: Sequence[str]) -> bool:
    """Whether the spec has every name. Prints the unknown ones on stderr."""
    unknown = [name for name in names if name not in spec.names]
    if unknown:
        print(
            "unknown service %s (services.yaml has: %s)" % (", ".join(unknown), ", ".join(spec.names)),
            file=sys.stderr,
        )
    return not unknown


def _up(manager: Manager, services: Sequence[Service]) -> bool:
    """Start ``services`` in order, and return True when every one is up.

    A service whose dependency failed is skipped, so one fault prints one failure, not a cascade.
    """
    failed: set[str] = set()
    for service in services:
        blocked = sorted(set(service.depends_on) & failed)
        if blocked:
            print("%s: skipped (%s failed)" % (service.name, ", ".join(blocked)))
            failed.add(service.name)
            continue
        if not manager.up(service, print):
            failed.add(service.name)
    return not failed


def _down(manager: Manager, spec: Spec, names: Sequence[str]) -> tuple[bool, tuple[Service, ...]]:
    """Stop the named services and their dependents, dependents first. Returns ``(ok, the chosen)``.

    No names choose every managed service. A named external or check service still gets its
    "nothing to stop" line, so the command never looks ignored.
    """
    chosen = spec.with_dependents(names) if names else tuple(s for s in spec.services if s.managed)
    chosen_names = {service.name for service in chosen}
    fine = True
    for service in reversed(spec.order(spec.names)):
        if service.name in chosen_names and not manager.down(service, print):
            fine = False
    return fine, chosen


def _cmd_up(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None or not _known(spec, args.names):
        return 1
    return 0 if _up(Manager(spec, paths), spec.order(args.names or None)) else 1


def _cmd_down(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None or not _known(spec, args.names):
        return 1
    fine, _ = _down(Manager(spec, paths), spec, args.names)
    return 0 if fine else 1


def _cmd_restart(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None or not _known(spec, args.names):
        return 1
    manager = Manager(spec, paths)
    fine, chosen = _down(manager, spec, args.names)
    if not fine:
        return 1
    return 0 if _up(manager, spec.order([service.name for service in chosen])) else 1


def _cmd_status(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None:
        return 1
    manager = Manager(spec, paths)
    sys.stdout.write(status_table([manager.status_row(service) for service in spec.services]))
    return 0


def _cmd_logs(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None or not _known(spec, [args.name]):
        return 1
    service = spec.get(args.name)
    if service.kind == "podman":
        try:
            podman = Manager(spec, paths).podman(service)
        except ExpandError as error:
            print(error, file=sys.stderr)
            return 1
        return podman.logs(service.container, args.follow)
    log_path = paths.log_file(service.name)
    if not log_path.is_file():
        print("no log yet at %s" % log_path, file=sys.stderr)
        return 1
    with log_path.open("r", encoding="utf-8", errors="replace") as fh:
        # The last lines, then whatever arrives, from the same handle: nothing
        # written between the two is lost.
        sys.stdout.write("".join(deque(fh, maxlen=args.lines)))
        sys.stdout.flush()
        if not args.follow:
            return 0
        try:
            while True:
                chunk = fh.read()
                if chunk:
                    sys.stdout.write(chunk)
                    sys.stdout.flush()
                else:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            return 0


def _cmd_env(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None:
        return 1
    manager = Manager(spec, paths)
    exports: dict[str, str] = {}
    try:
        for service in spec.services:
            exports.update(expand_all(service.exports, manager.ctx))
    except ExpandError as error:
        print(error, file=sys.stderr)
        return 1
    sys.stdout.write(env_lines(exports))
    return 0


def _cmd_doctor(args: argparse.Namespace, paths: Paths) -> int:
    spec = _load(paths)
    if spec is None:
        return 1
    return 0 if Manager(spec, paths).doctor(print) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sprout.services",
        description=sprout.services.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser(
        "init",
        help="write services.yaml and the per-service configs from the templates; "
        "make the secrets (never overwrites)",
    )
    init.set_defaults(func=_cmd_init)

    up = sub.add_parser("up", help="start what is not already healthy, dependencies first")
    up.add_argument("names", nargs="*", metavar="NAME", help="default: every enabled service")
    up.set_defaults(func=_cmd_up)

    down = sub.add_parser("down", help="stop, dependents first")
    down.add_argument(
        "names",
        nargs="*",
        metavar="NAME",
        help="default: everything (dependents of a named service are stopped too)",
    )
    down.set_defaults(func=_cmd_down)

    restart = sub.add_parser("restart", help="down then up for the named services and their dependents")
    restart.add_argument("names", nargs="+", metavar="NAME")
    restart.set_defaults(func=_cmd_restart)

    status = sub.add_parser("status", help="one line per service: name | kind | pid | health | port")
    status.set_defaults(func=_cmd_status)

    logs = sub.add_parser("logs", help="a service's log (a container's through `podman logs`)")
    logs.add_argument("name", metavar="NAME")
    logs.add_argument(
        "-f", "--follow", action="store_true", help="keep printing as lines arrive (Ctrl-C to stop)"
    )
    logs.add_argument(
        "-n", "--lines", type=int, default=50, help="how many trailing lines to start with (default 50)"
    )
    logs.set_defaults(func=_cmd_logs)

    env = sub.add_parser(
        "env", help='shell export lines for every service\'s exports: eval "$(python -m sprout.services env)"'
    )
    env.set_defaults(func=_cmd_env)

    doctor = sub.add_parser(
        "doctor", help="every host check and health probe, starting nothing; exit 1 when something is wrong"
    )
    doctor.set_defaults(func=_cmd_doctor)

    args = parser.parse_args(argv)
    command: Callable[[argparse.Namespace, Paths], int] = args.func
    try:
        return command(args, Paths.from_env())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
