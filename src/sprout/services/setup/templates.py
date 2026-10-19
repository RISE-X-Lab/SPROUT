"""The config files that ``init`` and ``setup`` write from the packaged templates, with this host's values.

The templates name the paths of an example host under ``/path/to/``. :class:`HostValues` holds the
values of this host that replace them. Each function returns new data and writes no file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.contract.backend import default_podman_store
from sprout.resource import resource

_RUNSTORE_TEMPLATE = resource("runstore/config.example.json")
_RL_DRIVER_TEMPLATE = resource("rl_driver/config.example.json")
#: The sections of a profile whose jobs run in a sandbox, on the backend that the section names.
SANDBOX_SECTIONS = ("run_defaults", "grade_defaults")


@dataclass(frozen=True)
class HostValues:
    """The values of this host that replace the example paths of the templates."""

    #: The Run Store's artifact root, under the state root. ``runstore init`` refuses a volatile one.
    artifact_root: str
    #: The sprout-runtime binary of the checkout.
    runtime_bin: str
    #: The interpreter of each profile's jobs: ``{python}``. The worker runs each job on it.
    python: str
    #: The podman store that ``{store}`` names: ``$SPROUT_PODMAN_STORE``, else ``~/sprout-btrfs``.
    store: str


def host_values(*, state: Path, runtime_bin: Path, python: str) -> HostValues:
    """The values of this host, with the state root ``state`` and the interpreter ``python``."""
    return HostValues(
        artifact_root=str(state / "runstore-artifacts"),
        runtime_bin=str(runtime_bin),
        python=python,
        store=default_podman_store(),
    )


def runstore_config(values: HostValues) -> dict[str, Any]:
    """The Run Store template with the artifact root of ``values``, and each profile from ``host_profile``."""
    config: dict[str, Any] = json.loads(_RUNSTORE_TEMPLATE.read_text(encoding="utf-8"))
    config["artifact_root"] = values.artifact_root
    config["profiles"] = {name: host_profile(profile, values) for name, profile in config["profiles"].items()}
    return config


def host_profile(profile: dict[str, Any], values: HostValues) -> dict[str, Any]:
    """A copy of ``profile`` with the interpreter, each runtime binary and each podman store of ``values``."""
    result: dict[str, Any] = _rewrite(
        _rewrite(profile, "python", values.python), "runtime_bin", values.runtime_bin
    )
    for section in SANDBOX_SECTIONS:
        podman = result.get(section, {}).get("backend", {}).get("podman")
        if isinstance(podman, dict):
            podman["store"] = values.store
    return result


def json_text(config: dict[str, Any]) -> str:
    """``config`` as the text of a config file: JSON with an indent of 2 and a final newline."""
    return json.dumps(config, indent=2) + "\n"


def rl_driver_text(state: Path) -> str:
    """The RL driver template with its ledger under ``state``, and its token in ``SPROUT_RL_DRIVER_TOKEN``."""
    config = json.loads(_RL_DRIVER_TEMPLATE.read_text(encoding="utf-8"))
    config["ledger"] = str(state / "rl-driver" / "groups.sqlite3")
    config["driver_token_env"] = "SPROUT_RL_DRIVER_TOKEN"
    return json_text(config)


def _rewrite(value: Any, key: str, replacement: Any) -> Any:
    """A copy of ``value`` with each ``key`` at any depth set to ``replacement``."""
    if isinstance(value, dict):
        return {k: (replacement if k == key else _rewrite(v, key, replacement)) for k, v in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, key, replacement) for item in value]
    return value
