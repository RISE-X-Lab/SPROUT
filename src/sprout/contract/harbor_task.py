"""A Harbor task directory as ``harbor==0.22.0`` reads it: ``HarborTask`` and ``load_task``.

A Harbor task is a directory with ``task.toml``, ``instruction.md``,
``environment/``, ``tests/`` and ``solution/``. ``load_task`` applies Harbor's
defaults and inheritance, so each ``HarborTask`` field holds the value that a
Harbor trial uses.

``load_task`` refuses a task that it cannot read, with a ``ValueError`` that
names the file and the field. It refuses a malformed value, a field that it does
not know, a schema version other than 1.0 to 1.4, and the parts of the format
that Sprout does not run: steps, compose services, healthchecks, MCP servers,
GPUs, Windows, network allowlists, task environment variables, task users, and a
separate verifier that does not start from ``tests/Dockerfile``.
ADR-0003 (``docs/decisions/0003-sprout-runs-harbor-verifier.md``) records why
Sprout reads the format itself.
"""

from __future__ import annotations

import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Harbor 0.22.0 reads every schema version up to 1.4 with one model, and so does this module.
_SCHEMA_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4")
_PUBLIC = "public"
_NO_NETWORK = "no-network"
#: ``harbor.constants.ORG_NAME_PATTERN``.
_TASK_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$")
#: The provenance lines that Harbor's ``strip_canary`` removes from the start of an instruction.
_CANARY_LINE = re.compile(r"^(<!--.*canary.*-->|#.*canary.*)$", re.IGNORECASE)
_COMPOSE_FILES = ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")
# Harbor's defaults for the timeouts that task.toml can leave out.
_VERIFIER_TIMEOUT_S = 600.0
_COLLECT_TIMEOUT_S = 60.0

# The fields of each table that this module knows. It refuses any other field.
# The informational fields (source, metadata, the [task] description) do not change a run.
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "source",
        "task",
        "metadata",
        "agent",
        "verifier",
        "environment",
        "solution",
        "artifacts",
    }
)
_TASK_FIELDS = frozenset({"name", "version", "description", "authors", "keywords"})
_AGENT_FIELDS = frozenset({"timeout_sec", "network_mode"})
_VERIFIER_FIELDS = frozenset(
    {"timeout_sec", "network_mode", "environment_mode", "environment", "collect", "env"}
)
_ENVIRONMENT_FIELDS = frozenset(
    {
        "docker_image",
        "workdir",
        "cpus",
        "memory_mb",
        "storage_mb",
        "network_mode",
        "allow_internet",
        "os",
        "gpus",
        "mcp_servers",
        "env",
        # It bounds an image build, which does not change what the agent or the verifier sees.
        "build_timeout_sec",
    }
)
_COLLECT_FIELDS = frozenset({"command", "timeout_sec", "service"})
_SOLUTION_FIELDS = frozenset({"env"})
#: Environment fields that Sprout runs only at these values: no GPUs, MCP servers or host variables.
_ENVIRONMENT_DEFAULTS: tuple[tuple[str, object], ...] = (("gpus", 0), ("mcp_servers", []), ("env", {}))


@dataclass(frozen=True)
class CollectCommand:
    """A ``[[verifier.collect]]`` command, which runs in the agent's sandbox after the agent stops."""

    command: str
    timeout_s: float


@dataclass(frozen=True)
class HarborTask:
    """The effective settings of one Harbor task. Paths are absolute.

    A resource that is ``None`` is not set by the task, so the backend's default
    applies. A network policy is ``"public"`` or ``"no-network"``.
    """

    task_dir: Path
    #: ``[task].name``, or the name of the directory when task.toml has no ``[task]`` table.
    name: str
    #: ``instruction.md`` without its leading canary lines, as Harbor gives it to the agent.
    instruction: str
    #: The prebuilt image of the agent's sandbox. None: the sandbox is built from ``dockerfile``.
    docker_image: str | None
    #: ``environment/Dockerfile``, or None when the task has none.
    dockerfile: Path | None
    #: The working directory of commands in the agent's sandbox. None: the image's ``WORKDIR``.
    workdir: str | None
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    #: The time limit of the agent. None: no limit.
    agent_timeout_s: float | None
    #: The network policy while the agent runs.
    agent_network: str
    verifier_timeout_s: float
    #: The network policy while ``tests/test.sh`` runs.
    verifier_network: str
    #: True when ``tests/test.sh`` runs in the agent's sandbox (``environment_mode = "shared"``).
    verifier_shares_environment: bool
    #: ``tests/Dockerfile``, the image of the separate verifier sandbox. None when the verifier shares.
    verifier_dockerfile: Path | None
    #: The resources of the separate verifier sandbox. None when the verifier shares.
    verifier_cpus: int | None
    verifier_memory_mb: int | None
    verifier_storage_mb: int | None
    collect: tuple[CollectCommand, ...]
    #: The paths that Harbor copies out of the agent's sandbox after the agent stops, besides
    #: ``/logs/artifacts``. A separate verifier sandbox gets them at the same paths.
    artifacts: tuple[str, ...]
    #: ``solution/``, with ``solve.sh``. The directory can be absent.
    solution_dir: Path
    tests_dir: Path


def load_task(task_dir: Path) -> HarborTask:
    """The task in ``task_dir``, with Harbor's defaults applied. Reads files and writes nothing.

    Raises ``ValueError``, which names the file and the field, for a task that
    this module cannot read (see the module docstring).
    """
    task_dir = task_dir.resolve()
    top = _read_config(task_dir / "task.toml")
    _check_schema_version(top)
    _only_known(top, _TOP_FIELDS)
    _check_solution(_section(top, "solution"))
    environment_table = _section(top, "environment")
    environment = _environment(environment_table)
    agent = _section(top, "agent")
    _only_known(agent, _AGENT_FIELDS)
    verifier = _verifier(
        _section(top, "verifier"),
        environment_table=environment_table,
        environment=environment,
        tests_dir=task_dir / "tests",
    )
    return HarborTask(
        task_dir=task_dir,
        name=_task_name(top, task_dir=task_dir),
        instruction=_instruction(task_dir / "instruction.md"),
        docker_image=environment.docker_image,
        dockerfile=_dockerfile(task_dir / "environment", table=environment_table, environment=environment),
        workdir=environment.workdir,
        cpus=environment.cpus,
        memory_mb=environment.memory_mb,
        storage_mb=environment.storage_mb,
        agent_timeout_s=_seconds(agent, "timeout_sec"),
        agent_network=_network_mode(agent) or environment.network,
        verifier_timeout_s=verifier.timeout_s,
        verifier_network=verifier.network,
        verifier_shares_environment=verifier.shares_environment,
        verifier_dockerfile=verifier.dockerfile,
        verifier_cpus=verifier.cpus,
        verifier_memory_mb=verifier.memory_mb,
        verifier_storage_mb=verifier.storage_mb,
        collect=verifier.collect,
        artifacts=_artifacts(top),
        solution_dir=task_dir / "solution",
        tests_dir=task_dir / "tests",
    )


@dataclass(frozen=True)
class _Table:
    """One table of a task.toml, with the dotted name that messages give its fields."""

    file: Path
    name: str
    values: Mapping[str, Any]

    def field(self, key: str) -> str:
        return f"{self.name}.{key}" if self.name else key

    def refuse(self, key: str, problem: str) -> ValueError:
        return ValueError(f"{self.file}: {self.field(key)}: {problem}")


@dataclass(frozen=True)
class _Environment:
    """What an environment table sets: the image, the working directory, the shape and the network."""

    docker_image: str | None
    workdir: str | None
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    #: Harbor's baseline policy of the sandbox, before a phase overrides it.
    network: str


@dataclass(frozen=True)
class _Verifier:
    """The verifier's settings. The sandbox fields are None when the verifier shares the agent's sandbox."""

    timeout_s: float
    network: str
    shares_environment: bool
    dockerfile: Path | None
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    collect: tuple[CollectCommand, ...]


def _read_config(path: Path) -> _Table:
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{path}: cannot read the task config ({error})") from error
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"{path}: not valid TOML ({error})") from error
    return _Table(file=path, name="", values=values)


def _check_schema_version(top: _Table) -> None:
    version = top.values.get("schema_version")
    if version is not None and version not in _SCHEMA_VERSIONS:
        raise top.refuse(
            "schema_version", f"Sprout reads versions {', '.join(_SCHEMA_VERSIONS)}, got {version!r}"
        )


def _check_solution(solution: _Table) -> None:
    _only_known(solution, _SOLUTION_FIELDS)
    _at_default(solution, "env", {})


def _task_name(top: _Table, *, task_dir: Path) -> str:
    if "task" not in top.values:
        return task_dir.name
    task = _section(top, "task")
    _only_known(task, _TASK_FIELDS)
    name = task.values.get("name")
    if not isinstance(name, str) or not _TASK_NAME.match(name) or ".." in name:
        raise task.refuse("name", f"must be '<org>/<name>', got {name!r}")
    return name


def _instruction(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"{path}: cannot read the instruction ({error})") from error
    lines = text.split("\n")
    start = 0
    while start < len(lines) and _CANARY_LINE.match(lines[start].strip()):
        start += 1
    while start < len(lines) and not lines[start].strip():
        start += 1
    return "\n".join(lines[start:])


def _environment(table: _Table) -> _Environment:
    _only_known(table, _ENVIRONMENT_FIELDS)
    for key, default in _ENVIRONMENT_DEFAULTS:
        _at_default(table, key, default)
    os_name = table.values.get("os", "linux")
    if not isinstance(os_name, str) or os_name.lower() != "linux":
        raise table.refuse("os", f"Sprout runs only 'linux', got {os_name!r}")
    return _Environment(
        docker_image=_text(table, "docker_image"),
        workdir=_text(table, "workdir"),
        cpus=_count(table, "cpus"),
        memory_mb=_count(table, "memory_mb"),
        storage_mb=_count(table, "storage_mb"),
        network=_baseline_network(table),
    )


def _dockerfile(environment_dir: Path, *, table: _Table, environment: _Environment) -> Path | None:
    """``environment/Dockerfile``, or None. Refuses compose files, and a task with no image to start from."""
    for name in _COMPOSE_FILES:
        if (environment_dir / name).exists():
            raise ValueError(f"{environment_dir / name}: Sprout does not run multi-service environments")
    dockerfile = environment_dir / "Dockerfile"
    if dockerfile.is_file():
        return dockerfile
    if environment.docker_image is None:
        raise table.refuse("docker_image", f"missing, and {dockerfile} does not exist")
    return None


def _verifier(
    table: _Table, *, environment_table: _Table, environment: _Environment, tests_dir: Path
) -> _Verifier:
    _only_known(table, _VERIFIER_FIELDS)
    _at_default(table, "env", {})
    own_table = _section(table, "environment") if "environment" in table.values else None
    shares = _shares_environment(table, own_table=own_table)
    if shares:
        if not (tests_dir / "test.sh").is_file():
            raise ValueError(f"{tests_dir / 'test.sh'}: missing, and the verifier runs it")
        sandbox = environment
    else:
        sandbox = _separate_sandbox(
            table,
            own_table=own_table,
            environment_table=environment_table,
            environment=environment,
            tests_dir=tests_dir,
        )
    return _Verifier(
        timeout_s=_seconds_or(table, "timeout_sec", default=_VERIFIER_TIMEOUT_S),
        network=_network_mode(table) or sandbox.network,
        shares_environment=shares,
        dockerfile=None if shares else tests_dir / "Dockerfile",
        cpus=None if shares else sandbox.cpus,
        memory_mb=None if shares else sandbox.memory_mb,
        storage_mb=None if shares else sandbox.storage_mb,
        collect=_collect(table),
    )


def _shares_environment(verifier: _Table, *, own_table: _Table | None) -> bool:
    """Harbor's verifier mode: ``environment_mode``, else separate when ``[verifier.environment]`` exists."""
    mode: object = verifier.values.get("environment_mode")
    if mode not in (None, "shared", "separate"):
        raise verifier.refuse("environment_mode", f"must be 'shared' or 'separate', got {mode!r}")
    if mode == "shared" and own_table is not None:
        raise verifier.refuse("environment", "a verifier with environment_mode 'shared' has no environment")
    if mode is None:
        return own_table is None
    return mode == "shared"


def _separate_sandbox(
    verifier: _Table,
    *,
    own_table: _Table | None,
    environment_table: _Table,
    environment: _Environment,
    tests_dir: Path,
) -> _Environment:
    """The environment of a separate verifier sandbox, which starts from ``tests/Dockerfile``."""
    # Without a [verifier.environment] table, Harbor gives the verifier a copy of [environment].
    if own_table is None:
        table, sandbox = environment_table, environment
    else:
        table, sandbox = own_table, _environment(own_table)
        if sandbox.workdir is not None:
            raise own_table.refuse("workdir", "Sprout runs a separate verifier in the WORKDIR of its image")
    # Harbor starts a verifier that names an image from that image, and does not upload tests/ into it.
    if sandbox.docker_image is not None:
        raise table.refuse("docker_image", "Sprout starts a separate verifier sandbox from tests/Dockerfile")
    dockerfile = tests_dir / "Dockerfile"
    if not dockerfile.is_file():
        key = "environment_mode" if "environment_mode" in verifier.values else "environment"
        raise verifier.refuse(key, f"a separate verifier sandbox starts from {dockerfile}, which is missing")
    return sandbox


def _collect(verifier: _Table) -> tuple[CollectCommand, ...]:
    entries = verifier.values.get("collect", [])
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise verifier.refuse("collect", f"must be an array of tables, got {entries!r}")
    name = verifier.field("collect")
    return tuple(
        _collect_command(_Table(file=verifier.file, name=f"{name}[{index}]", values=entry))
        for index, entry in enumerate(entries)
    )


def _collect_command(table: _Table) -> CollectCommand:
    _only_known(table, _COLLECT_FIELDS)
    _at_default(table, "service", "main")
    command = _text(table, "command")
    if command is None:
        raise table.refuse("command", "missing")
    return CollectCommand(
        command=command, timeout_s=_seconds_or(table, "timeout_sec", default=_COLLECT_TIMEOUT_S)
    )


def _artifacts(top: _Table) -> tuple[str, ...]:
    paths = top.values.get("artifacts", [])
    if not isinstance(paths, list) or not all(isinstance(path, str) and path for path in paths):
        raise top.refuse("artifacts", f"Sprout reads only a list of paths, got {paths!r}")
    return tuple(paths)


def _baseline_network(environment: _Table) -> str:
    allow_internet = environment.values.get("allow_internet")
    if allow_internet is not None and type(allow_internet) is not bool:
        raise environment.refuse("allow_internet", f"must be true or false, got {allow_internet!r}")
    mode = _network_mode(environment)
    if mode is not None:
        return mode
    # network_mode replaced allow_internet. Harbor reads allow_internet only when network_mode is absent.
    if allow_internet is None or allow_internet:
        return _PUBLIC
    return _NO_NETWORK


def _network_mode(table: _Table) -> str | None:
    """The ``network_mode`` that ``table`` sets, or None when it sets none."""
    mode = table.values.get("network_mode")
    if mode not in (None, _PUBLIC, _NO_NETWORK):
        raise table.refuse("network_mode", f"Sprout runs only {_PUBLIC!r} and {_NO_NETWORK!r}, got {mode!r}")
    return mode


def _section(table: _Table, key: str) -> _Table:
    """The table at ``key``, empty when ``table`` does not have it."""
    values = table.values.get(key, {})
    if not isinstance(values, dict):
        raise table.refuse(key, f"must be a table, got {values!r}")
    return _Table(file=table.file, name=table.field(key), values=values)


def _only_known(table: _Table, known: frozenset[str]) -> None:
    unknown = sorted(set(table.values) - known)
    if unknown:
        raise table.refuse(unknown[0], "Sprout does not read this field")


def _at_default(table: _Table, key: str, default: object) -> None:
    if key in table.values and table.values[key] != default:
        raise table.refuse(key, f"Sprout runs only {default!r}, got {table.values[key]!r}")


def _text(table: _Table, key: str) -> str | None:
    value = table.values.get(key)
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise table.refuse(key, f"must be a non-empty string, got {value!r}")
    return value


def _count(table: _Table, key: str) -> int | None:
    value = table.values.get(key)
    if value is not None and (type(value) is not int or value <= 0):
        raise table.refuse(key, f"must be a positive integer, got {value!r}")
    return value


def _seconds(table: _Table, key: str) -> float | None:
    value = table.values.get(key)
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise table.refuse(key, f"must be a positive number of seconds, got {value!r}")
    return float(value)


def _seconds_or(table: _Table, key: str, *, default: float) -> float:
    seconds = _seconds(table, key)
    return default if seconds is None else seconds
