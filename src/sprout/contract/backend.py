"""The backend section of a run's config: which backend runs it, and where its snapshots live."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_BACKEND = "podman"


def default_podman_store() -> str:
    """The podman store when none is named: ``$SPROUT_PODMAN_STORE``, else ``~/sprout-btrfs``.

    ``ops/setup_podman.sh`` makes ``~/sprout-btrfs``. A pool refuses a store that is not a btrfs
    mount that it can manage, so a host without a store fails at the first spawn, with what to run.
    """
    return os.environ.get("SPROUT_PODMAN_STORE") or str(Path.home() / "sprout-btrfs")


#: Recognised keys of the ``podman`` section. A container has no lease: it lives until it is removed.
PODMAN_KEYS = {
    # Required: a directory of this user on a btrfs filesystem mounted with user_subvol_rm_allowed.
    # Each sandbox is a subvolume in it. ops/btrfs_store.sh makes it once, as root.
    "store",
    # Required: the sprout-runtime binary. Each container mounts it and starts it as its entrypoint,
    # so no image needs a rebuild. A musl image needs the static CGO_ENABLED=0 build.
    "runtime_bin",
    # The podman executable. Unset means $SPROUT_PODMAN_BIN, else the `podman` on PATH.
    "podman_bin",
    # False starts each sandbox disconnected. Unset or True starts it connected.
    # A running sandbox can switch either way (set_network).
    "allow_internet",
    # The directory of the per-sandbox runtime socket directories. Unset means <store>/sockets.
    "socket_dir",
    # True runs the agent's commands, solve.sh and test.sh as the image's USER
    # (worker/sandbox/image_shell.py). Unset or False runs them as root, like each other command.
    "image_user",
    # True gives those commands, with image_user, the image's ENV again after the switch of user.
    # The container starts with the image's ENV either way.
    "image_env",
}


def backend_name(config: Mapping[str, Any]) -> str:
    """The name of the backend that ``config`` selects, in lower case. ``podman`` when it names none."""
    return str(config.get("backend") or DEFAULT_BACKEND).strip().lower()


def snapshot_store(config: Mapping[str, Any]) -> str | None:
    """The absolute path of the podman store of ``config``, where its snapshots are files on this host.

    ``~`` is expanded. ``None`` for the microVM backend, whose snapshots are AgentENV records, and
    for a podman section that names no store.
    """
    if backend_name(config) != "podman":
        return None
    store = backend_section(config, "podman", PODMAN_KEYS).get("store")
    return str(Path(expanded_path(store)).resolve()) if store else None


def expanded_path(path: str | os.PathLike[str]) -> str:
    """A path from a config with ``~`` expanded: a config file names the home that a shell expands."""
    return os.path.expanduser(str(path))


def backend_section(config: Mapping[str, Any], name: str, keys: set[str]) -> dict[str, Any]:
    """A copy of the ``name`` section of ``config``, empty when there is none.

    Raises BackendError when the section is not a mapping, or when it has a key outside ``keys``.
    """
    section = config.get(name) or {}
    if not isinstance(section, dict):
        raise BackendError(f"config key {name!r} must be a mapping of settings")
    unknown = set(section) - keys
    if unknown:
        raise BackendError(
            f"unknown {name} setting(s): {', '.join(sorted(unknown))}; known: {', '.join(sorted(keys))}"
        )
    return dict(section)


class BackendError(ValueError):
    """A backend was named or configured in a way that cannot work."""
