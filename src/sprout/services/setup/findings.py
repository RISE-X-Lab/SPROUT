"""The problems in the runstore.json of a host that make its jobs fail, each with its fix.

``init`` writes a runstore.json that has none of them. A file that a person edited or copied can have
these problems:

- the file is not valid JSON
- a value is still an example path of the template, under ``/path/to/``
- a ``worker_env`` reference names a variable that this shell does not set

No function here writes a file. docs/SERVICES.md ("Config checks") lists the checks.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sprout.services.health import Probe

RUNSTORE = "runstore.json"
FAIL = "fail"
WARN = "warn"
#: The fix of a runstore.json that cannot be read.
_UNREADABLE = "fix the file, or move it away and run `python -m sprout.services init`"
#: The prefix of each example path in the templates. A value under it names no path of this host.
_EXAMPLE_PREFIX = "/path/to/"
#: The path of a value in a JSON document: the key of each object and the index of each list.
_Path = tuple[str | int, ...]


@dataclass(frozen=True)
class Finding:
    """One problem in runstore.json, and what to do about it."""

    #: ``fail`` when each job of a profile fails, ``warn`` when the jobs fail unless the worker's
    #: shell differs from this one.
    status: str
    problem: str
    fix: str


class ConfigError(ValueError):
    """runstore.json cannot be read as a Run Store config."""


def host_findings(*, runstore: str, environ: Mapping[str, str]) -> list[Finding]:
    """Each finding in the runstore.json text ``runstore``. ``environ`` is the environment of this shell."""
    try:
        config = load_runstore(runstore)
    except ConfigError as error:
        return [Finding(FAIL, str(error), _UNREADABLE)]
    return [
        *_example_paths(config),
        *_unset_variables(config["profiles"], environ=environ),
    ]


def doctor_checks(found: Sequence[Finding]) -> list[tuple[str, Probe]]:
    """A doctor line for each finding in runstore.json, or one ``ok`` line when there is none."""
    lines = [
        (RUNSTORE, Probe(False, "%s. Fix: %s" % (f.problem, f.fix), warning=f.status == WARN)) for f in found
    ]
    return lines or [(RUNSTORE, Probe(True, "no problem found"))]


def load_runstore(text: str) -> dict[str, Any]:
    """The runstore.json ``text`` as a mapping with ``profiles``. Raises ConfigError for another text."""
    try:
        config = json.loads(text)
    except ValueError as error:
        raise ConfigError("runstore.json is not valid JSON: %s" % error) from None
    if not isinstance(config, dict) or not isinstance(config.get("profiles"), dict):
        raise ConfigError("runstore.json has no `profiles` object")
    return config


def unset_references(profiles: Mapping[str, Any], environ: Mapping[str, str]) -> dict[str, list[str]]:
    """Each variable that a ``worker_env`` reference names and ``environ`` does not set, with its profiles.

    The variables are sorted, and so are the names of the profiles of each one.
    """
    unset: dict[str, list[str]] = {}
    for name, profile in sorted(profiles.items()):
        worker_env = profile.get("worker_env") if isinstance(profile, dict) else None
        for variable in sorted(_references(worker_env)):
            if variable not in environ:
                unset.setdefault(variable, []).append(name)
    return dict(sorted(unset.items()))


def _example_paths(config: dict[str, Any]) -> list[Finding]:
    """A ``fail`` for each key and example path of the template, with the places that hold it."""
    places: dict[tuple[str, str], list[_Path]] = {}
    for path, value in _strings(config):
        if value.startswith(_EXAMPLE_PREFIX):
            places.setdefault((_key(path), value), []).append(path)
    return [
        Finding(
            FAIL,
            "`%s` is the example path %s of the template, %s" % (key, value, _where(paths)),
            "set it to a path on this host",
        )
        for (key, value), paths in sorted(places.items())
    ]


def _unset_variables(profiles: dict[str, Any], *, environ: Mapping[str, str]) -> list[Finding]:
    """A ``warn`` for each variable that a ``worker_env`` names and this shell does not set."""
    found = []
    for variable, names in unset_references(profiles, environ).items():
        problem = (
            "%s is not set in this shell, and the worker_env of %s names it: each job of %s fails with "
            "KeyError unless the shell of the worker sets it"
            % (variable, _profiles(names), "these profiles" if len(names) > 1 else "the profile")
        )
        fix = (
            "export %s in the shell that starts the worker, or remove the reference from runstore.json"
            % variable
        )
        found.append(Finding(WARN, problem, fix))
    return found


def _references(value: Any) -> set[str]:
    """The variables that the ``{"$env": NAME}`` references in ``value`` name.

    ``referenced_env`` of ``sprout.runstore.config`` reads them by the same rule.
    """
    if isinstance(value, dict):
        if set(value) == {"$env"}:
            return {value["$env"]}
        return set().union(*(_references(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_references(item) for item in value))
    return set()


def _strings(value: Any, path: _Path = ()) -> Iterator[tuple[_Path, str]]:
    """Each string in the JSON ``value``, with its path of keys and list indexes."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _strings(item, (*path, index))
    elif isinstance(value, str) and path:
        yield path, value


def _key(path: _Path) -> str:
    """The last object key on ``path``: the key of the value, or of the list that holds it."""
    return next(key for key in reversed(path) if isinstance(key, str))


def _where(paths: Sequence[_Path]) -> str:
    """Where the paths are: at the top level, in the named profiles, or both."""
    profiles = sorted({str(path[1]) for path in paths if path[0] == "profiles" and len(path) > 2})
    parts = ["at the top level"] if any(path[0] != "profiles" for path in paths) else []
    if profiles:
        parts.append("in %s" % _profiles(profiles))
    return " and ".join(parts)


def _profiles(names: Sequence[str]) -> str:
    """``profile <name>``, or ``profiles <name>, <name>`` for more than one."""
    return "profile%s %s" % ("s" if len(names) > 1 else "", ", ".join(names))
