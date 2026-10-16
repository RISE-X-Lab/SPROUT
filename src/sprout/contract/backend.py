"""The backend section of a run's config: which backend runs it, and where its snapshots live."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_BACKEND = "podman"


def default_podman_store() -> str:
    """The podman store when none is named: ``$SPROUT_PODMAN_STORE``, else
    ``~/sprout-btrfs``, the directory ``ops/setup_podman.sh`` makes. A pool
    refuses one that is not a btrfs mount it can manage, so a host without a
    store fails at the first spawn with what to run."""
    return os.environ.get("SPROUT_PODMAN_STORE") or str(Path.home() / "sprout-btrfs")


#: Recognised keys of the ``podman`` section. No lease; a container lives
#: until it is removed.
PODMAN_KEYS = {
    # A directory on a btrfs filesystem mounted with user_subvol_rm_allowed and
    # owned by this user, required: every sandbox is a subvolume in it. Made
    # once as root with ops/btrfs_store.sh.
    "store",
    # The sprout-runtime binary, required: it is bind-mounted into every
    # container and started as its entrypoint, so an image needs no rebuild
    # (a musl image needs the static CGO_ENABLED=0 build of it).
    "runtime_bin",
    # The podman executable, when it is not $SPROUT_PODMAN_BIN / the `podman` on PATH.
    "podman_bin",
    # False = every sandbox is disconnected before it starts; unset/True =
    # connected. A benchmark whose tasks declare no-network sets it, and a
    # running sandbox can be toggled either way (set_network).
    "allow_internet",
    # Where the per-sandbox runtime socket directories live (default
    # <store>/sockets).
    "socket_dir",
}


def backend_name(config: dict) -> str:
    """The backend this config selects."""
    return str(config.get("backend") or DEFAULT_BACKEND).strip().lower()


def snapshot_store(config: dict) -> str | None:
    """Where this config's snapshots live when they are files on this host.

    The podman store, expanded and made absolute, or ``None`` for the microVM
    backend, whose snapshots are AgentENV records. The orchestrator puts it on
    the run's record (``sandbox.created``), so ``harness sweep`` can delete a
    finished run's subvolumes given nothing but its journal.
    """
    if backend_name(config) != "podman":
        return None
    store = backend_section(config, "podman", PODMAN_KEYS).get("store")
    return str(Path(expanded_path(store)).resolve()) if store else None


def expanded_path(path) -> str:
    """A path from a config with ``~`` expanded: a YAML file names the home the
    shell would have expanded on a command line."""
    return os.path.expanduser(str(path))


def backend_section(config: dict, name: str, keys: set) -> dict:
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
