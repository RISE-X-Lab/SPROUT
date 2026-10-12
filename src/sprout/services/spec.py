"""``services.yaml``: the data model, its loader and the placeholder language.

The file names each service, its kind and how to tell it is alive; the code
knows how to start a process or a container and how to probe, nothing about any
particular service. So adding a service, moving a port or pointing the proxy
at another venv is an edit to the file, never to the code, and the packaged
template (``templates/services.yaml``) is one example of the file rather than
a special case.

Everything is validated at load time with the service and field named in the
error, because the alternative is a traceback from ``up`` half way through a
start sequence: an unknown dependency, a cycle, a kind or health kind outside
the known sets, a port that is not an int, a malformed placeholder. Only the
*values* placeholders stand for -- secrets, paths, the environment -- are
resolved later, by :func:`expand` with a :class:`Context`, so loading never
creates a secret or reads one.

Kinds: ``process`` (a command this CLI starts detached and supervises),
``podman`` (a container, through the podman CLI), ``external`` (something
started by hand; ``up`` prints its ``start_hint``) and ``check`` (a host fact
``doctor`` verifies; nothing to start). Health kinds: ``http`` (a GET must
answer 200), ``pg_isready``, ``process`` (the pid is alive and is ours),
``none``.

Placeholders: ``{python}`` the interpreter ``python:`` names (default: the
checkout's ``.venv``), ``{repo}`` the checkout, ``{config}`` the directory of
``services.yaml``, ``{state}`` the state directory, ``{secret:NAME}`` a
generated secret, ``{env:NAME}`` a variable of the shell running the CLI,
``{podman_bin}`` the podman the sandboxes run on and ``{store}`` their btrfs
store (``$SPROUT_PODMAN_BIN`` / ``$SPROUT_PODMAN_STORE``, else the defaults). A
leading ``~`` is the home directory. They work in every string but a
container's ``image`` and ``container``: the name is the key ``inspect``,
``start``, ``stop`` and ``podman logs`` all look the container up by, so it is
one literal, the same in the file and in ``podman ps`` (podman allows no brace
in a name anyway). A ``{secret:NAME}`` may stand only where its value travels
as a process's environment or is printed on request -- ``env``, ``exports``,
``health.headers`` -- never in ``command``, ``prepare``, ``volumes``,
``publish``, ``health.url`` or ``path``: those are written to the service log
as typed, echoed by ``up``, and an argv is readable by every user in
``/proc/<pid>/cmdline``. The loader refuses it, naming the field, so a one-line
slip in the file cannot leak a secret silently.

A process receives its own ``exports`` and those of every service it depends
on, transitively, plus its ``env``: ``exports`` are what a *consumer* of the
service needs (the DSN, a token, a base URL), and a dependent is a consumer.
``env`` prints every service's exports as shell export lines.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from sprout.services.state import Paths

KINDS = ("process", "podman", "external", "check")
HEALTH_KINDS = ("http", "pg_isready", "process", "none")
CHECKS = ("btrfs_store", "podman")
NAME = re.compile(r"^[a-z][a-z0-9-]*$")
#: podman's own rule for a container name; a placeholder cannot pass it.
CONTAINER_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")

#: What a service of each kind may say. A key outside the set is an error, so
#: `depends-on` or `commands` fails at load instead of being ignored.
_COMMON = {"kind", "enabled", "depends_on", "health", "port", "exports"}
_KEYS = {
    "process": _COMMON | {"command", "env", "unset_env", "prepare", "timeout_s"},
    "podman": _COMMON | {"image", "container", "publish", "volumes", "env", "timeout_s"},
    "external": _COMMON | {"start_hint"},
    "check": {"kind", "enabled", "check", "path", "exports"},
}
_HEALTH_KEYS = {"kind", "url", "headers", "host", "port", "user", "container"}
_STEP_KEYS = {"run", "unless"}

#: The placeholders :func:`expand` knows ...
PLACEHOLDER = re.compile(r"\{(python|repo|config|state|podman_bin|store|secret:[a-z0-9][a-z0-9-]*|env:[A-Za-z_][A-Za-z0-9_]*)\}")
#: ... and the shape of anything that was probably meant as one. A JSON object
#: in an argument (`{"mini": ...}`) does not match, so it is left alone.
_BRACED_WORD = re.compile(r"\{[A-Za-z_][\w:.-]*\}")


class SpecError(ValueError):
    """A problem with services.yaml, naming the service and field."""


class ExpandError(ValueError):
    """A placeholder the file allows but this run cannot resolve: an unset
    ``{env:NAME}``, or a ``{secret:NAME}`` that ``init`` has not made yet when
    the command must not create one. The callers turn it into one line about
    the service; it is never a traceback."""


@dataclass(frozen=True)
class Health:
    kind: str = "none"
    url: str = ""                                   # http
    headers: tuple[tuple[str, str], ...] = ()       # http; values may carry placeholders
    host: str = "127.0.0.1"                         # pg_isready
    port: int | None = None                         # pg_isready
    user: str = ""                                  # pg_isready: -U, when given
    #: pg_isready: the container to `podman exec` the probe in when the host
    #: has no pg_isready. Filled from the service's own container when unset.
    container: str = ""


@dataclass(frozen=True)
class Step:
    """One command run before a start; skipped while ``unless`` names an existing path."""

    run: tuple[str, ...]
    unless: str = ""


@dataclass(frozen=True)
class Service:
    name: str
    kind: str
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    health: Health = field(default_factory=Health)
    port: int | None = None
    exports: tuple[tuple[str, str], ...] = ()
    #: How long `up` waits for the health probe (default 60 s; the proxy needs more).
    timeout_s: float = 60.0
    # process
    command: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    unset_env: tuple[str, ...] = ()
    prepare: tuple[Step, ...] = ()
    # podman
    image: str = ""
    container: str = ""
    publish: str = ""
    volumes: tuple[str, ...] = ()
    # external
    start_hint: str = ""
    # check
    check: str = ""
    path: str = ""

    @property
    def managed(self) -> bool:
        """Whether this CLI starts and stops it."""
        return self.kind in ("process", "podman")


@dataclass(frozen=True)
class Spec:
    source: str
    python: str | None
    services: tuple[Service, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(service.name for service in self.services)

    def get(self, name: str) -> Service:
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(name)

    def order(self, names: Iterable[str] | None = None) -> tuple[Service, ...]:
        """The named services and their dependencies, dependencies first.

        Declaration order among independents, so the start sequence is the
        order the file reads in and the same on every run. ``None`` is every
        enabled service.
        """
        wanted = list(names) if names is not None else [s.name for s in self.services if s.enabled]
        ordered: list[Service] = []
        seen: set[str] = set()

        def visit(name: str) -> None:
            if name in seen:
                return
            seen.add(name)
            service = self.get(name)
            for dependency in service.depends_on:
                visit(dependency)
            ordered.append(service)

        for name in wanted:
            visit(name)
        return tuple(ordered)

    def with_dependents(self, names: Iterable[str]) -> tuple[Service, ...]:
        """The named services plus everything that depends on them, in start order.

        What ``down`` stops: a service whose dependency goes away is broken
        anyway, and stopping it first is what the reverse of this order does.
        """
        closure = set(names)
        changed = True
        while changed:
            changed = False
            for service in self.services:
                if service.name not in closure and closure & set(service.depends_on):
                    closure.add(service.name)
                    changed = True
        return tuple(service for service in self.services if service.name in closure)

    def exports_for(self, service: Service) -> tuple[tuple[str, str], ...]:
        """The export templates a process receives: its dependencies' (transitively,
        dependencies first) and then its own, a later one replacing an earlier."""
        merged: dict[str, str] = {}
        for provider in self.order([service.name]):
            merged.update(provider.exports)
        return tuple(merged.items())

    def secret_names(self) -> tuple[str, ...]:
        """Every ``{secret:NAME}`` the file mentions, in first-use order; what ``init`` creates."""
        names: list[str] = []
        for service in self.services:
            for _, text, _ in _placeholder_fields(service):
                for found in PLACEHOLDER.findall(text):
                    if found.startswith("secret:") and found[7:] not in names:
                        names.append(found[7:])
        return tuple(names)


# --- expansion ----------------------------------------------------------------

def installed_podman() -> Path:
    """Where ``ops/setup_podman.sh`` installs podman: a static build, off PATH so
    the distribution's podman, if any, is left alone. The same path as
    ``sprout_sandbox.podman.installed_podman`` (this package imports no layer it
    starts, so the path is spelled twice and a test keeps the two in step)."""
    return Path.home() / ".local" / "opt" / "podman" / "bin" / "podman"


def podman_bin_for(environ: Mapping[str, str]) -> str:
    """The podman the sandboxes run on: ``$SPROUT_PODMAN_BIN``, else the one
    ``ops/setup_podman.sh`` installed, else ``podman`` on PATH -- the rule of
    ``sprout_sandbox.podman.default_podman_bin``. What the ``{podman_bin}``
    placeholder, the ``podman`` check and the database container resolve to."""
    chosen = environ.get("SPROUT_PODMAN_BIN")
    if chosen:
        return os.path.expanduser(chosen)
    installed = installed_podman()
    if installed.is_file() and os.access(installed, os.X_OK):
        return str(installed)
    return "podman"


def store_for(environ: Mapping[str, str]) -> Path:
    """The btrfs store: ``$SPROUT_PODMAN_STORE``, else ``~/sprout-btrfs`` (every entry point's default)."""
    return Path(environ.get("SPROUT_PODMAN_STORE") or Path.home() / "sprout-btrfs")


@dataclass(frozen=True)
class Context:
    """What placeholders stand for at run time; the spec itself holds none of it."""

    paths: Paths
    python: str
    secret: Callable[[str], str]
    environ: Mapping[str, str]


def expand(text: str, ctx: Context) -> str:
    """``text`` with every placeholder resolved against ``ctx``; :class:`ExpandError` when one cannot be."""
    def value(match: "re.Match[str]") -> str:
        key = match.group(1)
        if key == "python":
            return ctx.python
        if key == "repo":
            return str(ctx.paths.repo)
        if key == "config":
            return str(ctx.paths.config)
        if key == "state":
            return str(ctx.paths.state)
        if key == "podman_bin":
            return podman_bin_for(ctx.environ)
        if key == "store":
            return str(store_for(ctx.environ))
        if key.startswith("secret:"):
            try:
                return ctx.secret(key[7:])
            except ValueError as error:          # `secrets.require`: not made yet
                raise ExpandError(str(error)) from None
        name = key[4:]
        if name not in ctx.environ:
            raise ExpandError("environment variable %s is not set" % name)
        return ctx.environ[name]

    expanded = PLACEHOLDER.sub(value, text)
    return os.path.expanduser(expanded) if expanded.startswith("~") else expanded


def expand_all(pairs: Iterable[tuple[str, str]], ctx: Context) -> dict[str, str]:
    return {key: expand(value, ctx) for key, value in pairs}


# --- loading ------------------------------------------------------------------

def template_path(name: str) -> Path:
    """A file shipped in ``templates/`` beside this module (hatchling packages them)."""
    return Path(__file__).with_name("templates") / name


def load_spec(path: Path) -> Spec:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SpecError("%s: not valid YAML: %s" % (path, error)) from None
    return parse_spec(data, str(path))


def parse_spec(data: Any, source: str = "services.yaml") -> Spec:
    if not isinstance(data, dict):
        raise SpecError("%s: the file must be a mapping with a `services` key" % source)
    unknown = set(data) - {"python", "services"}
    if unknown:
        raise SpecError("%s: unknown top-level keys %s (known: python, services)" % (source, sorted(unknown)))
    python = data.get("python")
    if python is not None and not isinstance(python, str):
        raise SpecError("%s: `python` must be a string path or null" % source)
    raw_services = data.get("services")
    if not isinstance(raw_services, dict) or not raw_services:
        raise SpecError("%s: `services` must be a non-empty mapping of name -> service" % source)

    services = tuple(_parse_service(source, name, body) for name, body in raw_services.items())
    names = {service.name for service in services}
    for service in services:
        for dependency in service.depends_on:
            if dependency not in names:
                raise SpecError("%s: service %r, field 'depends_on': unknown service %r"
                                % (source, service.name, dependency))
            if dependency == service.name:
                raise SpecError("%s: service %r, field 'depends_on': depends on itself" % (source, service.name))
    spec = Spec(source=source, python=python, services=services)
    _check_cycles(spec)
    for service in services:
        for field_name, text, secret_ok in _placeholder_fields(service):
            for braced in _BRACED_WORD.findall(text):
                if not PLACEHOLDER.fullmatch(braced):
                    raise SpecError("%s: service %r, field %r: unknown placeholder %s (known: {python} {repo} "
                                    "{config} {state} {podman_bin} {store} {secret:name} {env:NAME})"
                                    % (source, service.name, field_name, braced))
                if braced.startswith("{secret:") and not secret_ok:
                    raise SpecError("%s: service %r, field %r: a secret may not appear here: an argv, a mount, a URL "
                                    "or a path is logged and printed as typed; pass it through `env` or `exports` "
                                    "(or `health.headers`) instead" % (source, service.name, field_name))
    return spec


def _parse_service(source: str, name: Any, body: Any) -> Service:
    if not isinstance(name, str) or not NAME.match(name):
        raise SpecError("%s: service name %r: lower-case letters, digits and dashes, starting with a letter"
                        % (source, name))

    def fail(field_name: str, problem: str) -> SpecError:
        return SpecError("%s: service %r, field %r: %s" % (source, name, field_name, problem))

    if not isinstance(body, dict):
        raise fail("kind", "the service must be a mapping")
    kind = body.get("kind")
    if kind not in KINDS:
        raise fail("kind", "%r is not one of %s" % (kind, ", ".join(KINDS)))
    unknown = set(body) - _KEYS[kind]
    if unknown:
        raise fail(sorted(unknown)[0], "not a field of a %s service (known: %s)" % (kind, ", ".join(sorted(_KEYS[kind]))))

    enabled = body.get("enabled", True)
    if not isinstance(enabled, bool):
        raise fail("enabled", "must be true or false")
    depends_on = _strings_list(body.get("depends_on", []), fail, "depends_on")
    port = _port(body.get("port"), fail, "port")
    health = _parse_health(body.get("health"), fail, body.get("container", ""))
    timeout_s = body.get("timeout_s", 60)
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise fail("timeout_s", "must be a positive number of seconds")
    exports = _string_pairs(body.get("exports", {}), fail, "exports")
    env = _string_pairs(body.get("env", {}), fail, "env")

    command = tuple(_strings_list(body.get("command", []), fail, "command"))
    if kind == "process" and not command:
        raise fail("command", "a process needs a non-empty argv list")
    image, container = body.get("image", ""), body.get("container", "")
    if kind == "podman":
        # Neither is expanded: the name is what every verb looks the container
        # up by, and a reference has no brace. Refusing here beats podman's
        # exit 125 half way through `up`.
        if not isinstance(image, str) or not image:
            raise fail("image", "a podman service needs an image reference")
        if "{" in image:
            raise fail("image", "an image reference is used as written; placeholders are not expanded in it")
        if not isinstance(container, str) or not CONTAINER_NAME.match(container):
            raise fail("container", "a podman service needs a container name: letters, digits, `_`, `.`, `-` "
                                    "(podman's rule); placeholders are not expanded in it")
    check = body.get("check", "")
    if kind == "check" and check not in CHECKS:
        raise fail("check", "%r is not one of %s" % (check, ", ".join(CHECKS)))
    for key in ("publish", "start_hint", "path"):
        if body.get(key) is not None and not isinstance(body.get(key), str):
            raise fail(key, "must be a string")
    steps = body.get("prepare", [])
    if not isinstance(steps, list):
        raise fail("prepare", "must be a list of steps ({run: [...], unless: path})")
    prepare = tuple(_parse_step(step, fail, index) for index, step in enumerate(steps))

    return Service(
        name=name, kind=kind, enabled=enabled, depends_on=tuple(depends_on), health=health, port=port,
        exports=exports, timeout_s=float(timeout_s), command=command, env=env,
        unset_env=tuple(_strings_list(body.get("unset_env", []), fail, "unset_env")), prepare=prepare,
        image=image, container=container, publish=body.get("publish") or "",
        volumes=tuple(_strings_list(body.get("volumes", []), fail, "volumes")),
        start_hint=body.get("start_hint") or "", check=check, path=body.get("path") or "",
    )


def _parse_health(body: Any, fail: Callable[[str, str], SpecError], container: str) -> Health:
    if body is None:
        return Health()
    if not isinstance(body, dict):
        raise fail("health", "must be a mapping with a `kind`")
    unknown = set(body) - _HEALTH_KEYS
    if unknown:
        raise fail("health.%s" % sorted(unknown)[0], "not a health field (known: %s)" % ", ".join(sorted(_HEALTH_KEYS)))
    kind = body.get("kind")
    if kind not in HEALTH_KINDS:
        raise fail("health.kind", "%r is not one of %s" % (kind, ", ".join(HEALTH_KINDS)))
    url = body.get("url", "")
    if kind == "http" and (not isinstance(url, str) or not url):
        raise fail("health.url", "an http probe needs a URL")
    port = _port(body.get("port"), fail, "health.port")
    if kind == "pg_isready" and port is None:
        raise fail("health.port", "a pg_isready probe needs the port")
    for key in ("host", "user", "container"):
        if body.get(key) is not None and not isinstance(body.get(key), str):
            raise fail("health.%s" % key, "must be a string")
    return Health(kind=kind, url=url, headers=_string_pairs(body.get("headers", {}), fail, "health.headers"),
                  host=body.get("host") or "127.0.0.1", port=port, user=body.get("user") or "",
                  container=body.get("container") or container)


def _parse_step(body: Any, fail: Callable[[str, str], SpecError], index: int) -> Step:
    where = "prepare[%d]" % index
    if not isinstance(body, dict) or set(body) - _STEP_KEYS or "run" not in body:
        raise fail(where, "a step is {run: [argv...]} with an optional `unless: path`")
    run = _strings_list(body["run"], fail, where + ".run")
    if not run:
        raise fail(where + ".run", "needs a non-empty argv list")
    unless = body.get("unless", "")
    if not isinstance(unless, str):
        raise fail(where + ".unless", "must be a path")
    return Step(run=tuple(run), unless=unless)


def _port(value: Any, fail: Callable[[str, str], SpecError], where: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
        raise fail(where, "must be an integer port (1-65535), got %r" % (value,))
    return value


def _strings_list(value: Any, fail: Callable[[str, str], SpecError], where: str) -> list[str]:
    if not isinstance(value, list):
        raise fail(where, "must be a list of strings")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise fail(where, "every item must be a string, got %r" % (item,))
        out.append(str(item))
    return out


def _string_pairs(value: Any, fail: Callable[[str, str], SpecError], where: str) -> tuple[tuple[str, str], ...]:
    """A mapping of names to strings; a number is accepted and spelled out, since
    YAML reads `--port 4000`'s 4000 or a bare `user: 70` as numbers."""
    if not isinstance(value, dict):
        raise fail(where, "must be a mapping of name -> string")
    pairs = []
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise fail(where, "keys must be strings, got %r" % (key,))
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise fail("%s.%s" % (where, key), "must be a string, got %r" % (item,))
        pairs.append((key, str(item)))
    return tuple(pairs)


def _check_cycles(spec: Spec) -> None:
    """Depth-first with a path, so the error shows the loop rather than naming one member."""
    done: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in path:
            loop = path[path.index(name):] + [name]
            raise SpecError("%s: service %r, field 'depends_on': dependency cycle %s"
                            % (spec.source, name, " -> ".join(loop)))
        if name in done:
            return
        for dependency in spec.get(name).depends_on:
            visit(dependency, path + [name])
        done.add(name)

    for service in spec.services:
        visit(service.name, [])


def _placeholder_fields(service: Service) -> Iterable[tuple[str, str, bool]]:
    """Every ``(field, text, secret allowed)`` a placeholder may appear in.

    This is the one list the loader validates and the code expands, so a field
    is never in one and not the other (``volumes`` once was validated and then
    handed to podman raw). ``start_hint`` is printed verbatim and ``image`` /
    ``container`` are literal, so none of the three is here. A secret is
    allowed only where the value becomes a process's environment or is printed
    on request; everywhere else the text is logged and echoed as typed.
    """
    for text in service.command:
        yield "command", text, False
    for text in service.unset_env:
        yield "unset_env", text, False
    for field_name, pairs in (("env", service.env), ("exports", service.exports),
                              ("health.headers", service.health.headers)):
        for key, value in pairs:
            yield "%s.%s" % (field_name, key), value, True
    for index, step in enumerate(service.prepare):
        for text in step.run:
            yield "prepare[%d].run" % index, text, False
        yield "prepare[%d].unless" % index, step.unless, False
    for text in service.volumes:
        yield "volumes", text, False
    yield "publish", service.publish, False
    yield "health.url", service.health.url, False
    yield "path", service.path, False
