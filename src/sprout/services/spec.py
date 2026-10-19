"""``services.yaml``: the data model, its loader and the placeholder language.

The file names each service, its kind and how to tell that it is alive. The code
knows how to start a process or a container and how to probe it, and nothing
about a particular service. The packaged template
(``resources/services/services.yaml``) is one example of the file, not a special case.

``load_spec`` and ``parse_spec`` validate the whole file at load time, and raise
``SpecError`` with the service and the field named. Loading never creates or reads
a secret. :func:`expand` resolves the values of the placeholders later, against a
:class:`Context`. A ``{secret:NAME}`` may appear only where its value becomes a
process's environment or is printed on request.

docs/SERVICES.md ("services.yaml") lists the kinds, the health kinds, the
placeholders and the fields that may hold a secret.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sprout.resource import resource
from sprout.services.state import Paths

KINDS = ("process", "podman", "external", "check")
HEALTH_KINDS = ("http", "pg_isready", "process", "none")
CHECKS = ("btrfs_store", "podman")
NAME = re.compile(r"^[a-z][a-z0-9-]*$")
#: podman's rule for a container name. A placeholder cannot match it.
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
PLACEHOLDER = re.compile(
    r"\{(python|repo|config|state|podman_bin|store|secret:[a-z0-9][a-z0-9-]*|env:[A-Za-z_][A-Za-z0-9_]*)\}"
)
#: ... and the shape of anything that was probably meant as one. A JSON object
#: in an argument (`{"mini": ...}`) does not match, so it is left alone.
_BRACED_WORD = re.compile(r"\{[A-Za-z_][\w:.-]*\}")


class SpecError(ValueError):
    """A problem with services.yaml, naming the service and field."""


class ExpandError(ValueError):
    """A placeholder that the file allows but this run cannot resolve.

    Examples are an unset ``{env:NAME}``, a ``{secret:NAME}`` that ``init`` has not made when the
    command must not create one, and a ``{podman_bin}`` or ``{store}`` that the worker's doctor did not
    report."""


@dataclass(frozen=True)
class Health:
    """How to tell that a service is alive. The comment on a field names the probe kinds that read it."""

    kind: str = "none"
    url: str = ""  # http
    headers: tuple[tuple[str, str], ...] = ()  # http. The values may carry placeholders.
    host: str = "127.0.0.1"  # pg_isready
    port: int | None = None  # pg_isready
    user: str = ""  # pg_isready: -U, when given
    #: pg_isready: the container to `podman exec` the probe in when the host
    #: has no pg_isready. Filled from the service's own container when unset.
    container: str = ""


@dataclass(frozen=True)
class Step:
    """One command run before a start. It is skipped while ``unless`` names an existing path."""

    run: tuple[str, ...]
    unless: str = ""


@dataclass(frozen=True)
class Service:
    """One service of the file. The fields under a kind's comment belong to services of that kind."""

    name: str
    kind: str
    enabled: bool = True
    depends_on: tuple[str, ...] = ()
    health: Health = field(default_factory=Health)
    port: int | None = None
    exports: tuple[tuple[str, str], ...] = ()
    #: How long ``up`` waits for the health probe, in seconds. The proxy needs more than the default.
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
    """A loaded file: where it came from, its ``python`` and its services in the order of the file."""

    source: str
    python: str | None
    services: tuple[Service, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(service.name for service in self.services)

    def get(self, name: str) -> Service:
        """The service ``name``. Raises KeyError for an unknown name."""
        for service in self.services:
            if service.name == name:
                return service
        raise KeyError(name)

    def order(self, names: Iterable[str] | None = None) -> tuple[Service, ...]:
        """The named services and their dependencies, dependencies first.

        Independent services keep the order of the file, so every run starts them in the same order.
        ``None`` names every enabled service. Raises KeyError for an unknown name.
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
        """The named services plus each service that depends on them, transitively, in the order of the file.

        A service whose dependency stops is broken, so a stop of the named services also stops these.
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
        """The name of each ``{secret:NAME}`` that the file mentions, in the order of first use."""
        names: list[str] = []
        for service in self.services:
            for _, text, _ in _placeholder_fields(service):
                for found in PLACEHOLDER.findall(text):
                    if found.startswith("secret:") and found[7:] not in names:
                        names.append(found[7:])
        return tuple(names)


# --- expansion ----------------------------------------------------------------


@dataclass(frozen=True)
class Context:
    """What the placeholders stand for at run time. The spec holds none of these values."""

    paths: Paths
    python: str
    #: The value of ``{secret:NAME}``, by name. Raises ``ValueError`` when it cannot give one.
    secret: Callable[[str], str]
    #: The value of ``{podman_bin}`` or ``{store}``, by name. Raises ``ValueError`` when it is unknown.
    host: Callable[[str], str]
    environ: Mapping[str, str]


def expand(text: str, ctx: Context) -> str:
    """``text`` with each placeholder resolved against ``ctx``, and with a leading ``~`` expanded.

    Raises :class:`ExpandError` when a placeholder cannot be resolved.
    """

    def value(match: re.Match[str]) -> str:
        key = match.group(1)
        if key == "python":
            return ctx.python
        if key == "repo":
            return str(ctx.paths.repo)
        if key == "config":
            return str(ctx.paths.config)
        if key == "state":
            return str(ctx.paths.state)
        if key in ("podman_bin", "store"):
            try:
                return ctx.host(key)
            except ValueError as error:  # `WorkerDoctor.fact`: the doctor gave no report.
                raise ExpandError(str(error)) from None
        if key.startswith("secret:"):
            try:
                return ctx.secret(key[7:])
            except ValueError as error:  # `secrets.require`: the secret is not made yet.
                raise ExpandError(str(error)) from None
        name = key[4:]
        if name not in ctx.environ:
            raise ExpandError("environment variable %s is not set" % name)
        return ctx.environ[name]

    expanded = PLACEHOLDER.sub(value, text)
    return os.path.expanduser(expanded) if expanded.startswith("~") else expanded


def expand_all(pairs: Iterable[tuple[str, str]], ctx: Context) -> dict[str, str]:
    """Each ``(name, text)`` pair as a mapping, with ``text`` expanded. Raises :class:`ExpandError`."""
    return {key: expand(value, ctx) for key, value in pairs}


# --- loading ------------------------------------------------------------------


def template_path(name: str) -> Path:
    """A template the service manager seeds a host's config from (``resources/services/<name>``)."""
    return resource(f"services/{name}")


def load_spec(path: Path) -> Spec:
    """The spec in the YAML file ``path``. Raises SpecError for invalid YAML or an invalid spec."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise SpecError("%s: not valid YAML: %s" % (path, error)) from None
    return parse_spec(data, str(path))


def parse_spec(data: Any, source: str = "services.yaml") -> Spec:
    """The spec in ``data``, the parsed YAML of ``source``. Raises SpecError at the first invalid part."""
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
                raise SpecError(
                    "%s: service %r, field 'depends_on': unknown service %r"
                    % (source, service.name, dependency)
                )
            if dependency == service.name:
                raise SpecError(
                    "%s: service %r, field 'depends_on': depends on itself" % (source, service.name)
                )
    spec = Spec(source=source, python=python, services=services)
    _check_cycles(spec)
    for service in services:
        for field_name, text, secret_ok in _placeholder_fields(service):
            for braced in _BRACED_WORD.findall(text):
                if not PLACEHOLDER.fullmatch(braced):
                    raise SpecError(
                        "%s: service %r, field %r: unknown placeholder %s (known: {python} {repo} "
                        "{config} {state} {podman_bin} {store} {secret:name} {env:NAME})"
                        % (source, service.name, field_name, braced)
                    )
                if braced.startswith("{secret:") and not secret_ok:
                    raise SpecError(
                        "%s: service %r, field %r: a secret may not appear here: an argv, a mount, a URL "
                        "or a path is logged and printed as typed; pass it through `env` or `exports` "
                        "(or `health.headers`) instead" % (source, service.name, field_name)
                    )
    return spec


def _parse_service(source: str, name: Any, body: Any) -> Service:
    if not isinstance(name, str) or not NAME.match(name):
        raise SpecError(
            "%s: service name %r: lower-case letters, digits and dashes, starting with a letter"
            % (source, name)
        )

    def fail(field_name: str, problem: str) -> SpecError:
        return SpecError("%s: service %r, field %r: %s" % (source, name, field_name, problem))

    if not isinstance(body, dict):
        raise fail("kind", "the service must be a mapping")
    kind = body.get("kind")
    if kind not in KINDS:
        raise fail("kind", "%r is not one of %s" % (kind, ", ".join(KINDS)))
    unknown = set(body) - _KEYS[kind]
    if unknown:
        raise fail(
            sorted(unknown)[0],
            "not a field of a %s service (known: %s)" % (kind, ", ".join(sorted(_KEYS[kind]))),
        )

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
        # Neither is expanded: every podman verb finds the container by this name, and a reference has
        # no brace. A refusal here is better than a podman exit 125 half way through `up`.
        if not isinstance(image, str) or not image:
            raise fail("image", "a podman service needs an image reference")
        if "{" in image:
            raise fail("image", "an image reference is used as written; placeholders are not expanded in it")
        if not isinstance(container, str) or not CONTAINER_NAME.match(container):
            raise fail(
                "container",
                "a podman service needs a container name: letters, digits, `_`, `.`, `-` "
                "(podman's rule); placeholders are not expanded in it",
            )
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
        name=name,
        kind=kind,
        enabled=enabled,
        depends_on=tuple(depends_on),
        health=health,
        port=port,
        exports=exports,
        timeout_s=float(timeout_s),
        command=command,
        env=env,
        unset_env=tuple(_strings_list(body.get("unset_env", []), fail, "unset_env")),
        prepare=prepare,
        image=image,
        container=container,
        publish=body.get("publish") or "",
        volumes=tuple(_strings_list(body.get("volumes", []), fail, "volumes")),
        start_hint=body.get("start_hint") or "",
        check=check,
        path=body.get("path") or "",
    )


def _parse_health(body: Any, fail: Callable[[str, str], SpecError], container: str) -> Health:
    if body is None:
        return Health()
    if not isinstance(body, dict):
        raise fail("health", "must be a mapping with a `kind`")
    unknown = set(body) - _HEALTH_KEYS
    if unknown:
        raise fail(
            "health.%s" % sorted(unknown)[0],
            "not a health field (known: %s)" % ", ".join(sorted(_HEALTH_KEYS)),
        )
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
    return Health(
        kind=kind,
        url=url,
        headers=_string_pairs(body.get("headers", {}), fail, "health.headers"),
        host=body.get("host") or "127.0.0.1",
        port=port,
        user=body.get("user") or "",
        container=body.get("container") or container,
    )


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


def _string_pairs(
    value: Any, fail: Callable[[str, str], SpecError], where: str
) -> tuple[tuple[str, str], ...]:
    """The ``(name, string)`` pairs of a mapping. A number becomes its string, because YAML reads
    the 4000 of `--port 4000` or a bare `user: 70` as numbers."""
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
    """Raise SpecError for a dependency cycle. The error shows the whole loop, not one member."""
    done: set[str] = set()

    def visit(name: str, path: list[str]) -> None:
        if name in path:
            loop = path[path.index(name) :] + [name]
            raise SpecError(
                "%s: service %r, field 'depends_on': dependency cycle %s"
                % (spec.source, name, " -> ".join(loop))
            )
        if name in done:
            return
        for dependency in spec.get(name).depends_on:
            visit(dependency, path + [name])
        done.add(name)

    for service in spec.services:
        visit(service.name, [])


def _placeholder_fields(service: Service) -> Iterable[tuple[str, str, bool]]:
    """Each ``(field, text, secret allowed)`` that a placeholder may appear in.

    The loader validates these fields, so each field that the code expands must be here. ``start_hint``,
    ``image`` and ``container`` are literal. A secret is allowed only where the value becomes a process's
    environment or is printed on request. Each other field is logged and echoed as typed."""
    for text in service.command:
        yield "command", text, False
    for text in service.unset_env:
        yield "unset_env", text, False
    for field_name, pairs in (
        ("env", service.env),
        ("exports", service.exports),
        ("health.headers", service.health.headers),
    ):
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
