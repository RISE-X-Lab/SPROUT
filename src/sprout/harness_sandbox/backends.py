"""Where sandboxes come from — config to a pool, in one place.

The session names no concrete pool: it hands its ``execution`` section here and
gets back a pool built from the named backend's own section::

    execution:
      backend: podman                   # the default when none is named
      podman:
        store: /home/ubuntu/sprout-btrfs      # a directory on btrfs (ops/btrfs_store.sh)
        runtime_bin: runtime/sprout-runtime

    execution:
      backend: microvm
      microvm:
        server_url: http://127.0.0.1:8000
        api_key_file: ~/.config/sprout/aenv-key

Two backends, and both can snapshot -- every run leaves snapshots, so a backend
that cannot take them has no place here. ``podman`` (rootless containers on this
host, each on a btrfs subvolume of the store) is the default: it snapshots with
``btrfs subvolume snapshot``, disk-only, nothing to compact, egress on or off per
sandbox, and needs no service: this host's podman and a btrfs store. ``microvm`` (Firecracker VMs via
AgentENV) adds memory snapshots, server-side chain compaction and the Run Store,
for the price of the snapshot service.
Each section is checked key by key, so an unknown key there is an error rather
than a silently ignored typo, because a mistyped `server_url` would otherwise
look like a connection failure.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

from sprout_sandbox import MicroVMPool, PodmanPool
from sprout_sandbox.podman import default_podman_bin

__all__ = ["build_pool", "backend_name", "BackendError", "BACKENDS",
           "resolve_microvm_endpoint", "with_sandbox_budget",
           "snapshot_store", "spawn_takes_shape", "default_podman_store",
           "default_aenv_server_url"]


class BackendError(ValueError):
    """A backend was named or configured in a way that cannot work."""


DEFAULT_BACKEND = "podman"
VM_LIFETIME_MARGIN_S = 600
#: Where an AgentENV server listens unless it is told otherwise: its own
#: ``API_ADDR`` default, port 8000. A server elsewhere (this project's bench
#: instance listens on 18000) is named in AENV_SERVER_URL.
DEFAULT_AENV_SERVER_URL = "http://127.0.0.1:8000"


def default_podman_store() -> str:
    """The podman store when none is named: ``$SPROUT_PODMAN_STORE``, else
    ``~/sprout-btrfs``, the directory ``ops/setup_podman.sh`` makes. A pool
    refuses one that is not a btrfs mount it can manage, so a host without a
    store fails at the first spawn with what to run."""
    return os.environ.get("SPROUT_PODMAN_STORE") or str(Path.home() / "sprout-btrfs")


def default_aenv_server_url() -> str:
    """The AgentENV server when none is named: ``$AENV_SERVER_URL``, else
    :data:`DEFAULT_AENV_SERVER_URL`. For the callers that may assume one --
    the Harbor environment and the cleanup client (``harness reap`` and
    ``sweep``, a memory snapshot's release), which have to agree on it.
    ``build_pool`` assumes none: a config that names no server is refused."""
    return os.environ.get("AENV_SERVER_URL") or DEFAULT_AENV_SERVER_URL


def with_sandbox_budget(config: dict, budget_s: float) -> dict:
    """Cover a bounded operation's wall budget without changing its timeout.

    A VM lease is not an idle/tool timeout. Preserve a longer explicit lease,
    but lift a missing/short one to budget + provisioning/teardown allowance.
    Only the microVM backend leases: a podman container lives until it is
    removed, so its config passes through untouched.
    """
    result = dict(config)
    if backend_name(result) != "microvm":
        return result
    if (isinstance(budget_s, bool) or not isinstance(budget_s, (int, float))
            or not math.isfinite(budget_s) or budget_s <= 0):
        raise BackendError("VM lifetime requires a finite positive operation budget")
    section = _section(config, "microvm", MICROVM_KEYS)
    explicit = section.get("sandbox_ttl", 0)
    if (isinstance(explicit, bool) or not isinstance(explicit, (int, float))
            or not math.isfinite(explicit) or explicit < 0):
        raise BackendError("microvm.sandbox_ttl must be a finite non-negative number")
    section["sandbox_ttl"] = max(math.ceil(explicit), math.ceil(budget_s) + VM_LIFETIME_MARGIN_S)
    result["microvm"] = section
    return result


#: Recognised keys of the ``microvm`` section, so a typo is reported rather than ignored.
MICROVM_KEYS = {"server_url", "runtime_port", "api_key", "api_key_file",
                "request_timeout", "sandbox_ttl",
                # Whether this harness's image names are OCI references to
                # cold-start (a benchmark's per-instance images) rather than
                # names of snapshots already in the backend's catalog. Only the
                # cold-start path accepts an image reference, so the answer is
                # configuration, not something to guess from the string.
                "from_image",
                # Path to a local sprout-runtime binary. Set it and the harness
                # builds a template per instance image on demand, uploading
                # the binary through the backend's file service
                # (harness_sandbox/templates.py); leave it unset and the harness
                # expects templates to exist already.
                "runtime_bin",
                # False = no egress from any sandbox this pool creates
                # (AgentENV `allowInternetAccess`). A benchmark whose tasks
                # declare no-network sets it; unset keeps the server default.
                "allow_internet",
                # True = templates launch the runtime under the image's own
                # OCI ENV (PATH included), read with regctl at build time. The
                # guest agent otherwise rebuilds PATH and drops e.g. a venv's
                # bin. Opt-in so existing template names stay stable.
                "image_env"}

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


def spawn_takes_shape(config: dict) -> bool:
    """Whether every spawn on this backend takes the sandbox's shape.

    A microVM template or snapshot has its shape baked in: only a cold start
    from an image chooses one, and ``MicroVMPool.spawn`` refuses the argument.
    A podman sandbox is a subvolume, which carries none, so the pool takes
    ``resources`` on every spawn -- from an image reference and from a
    snapshot id alike, through the one path it has -- and a branch that wants
    its parent's shape passes it again. The session reads this to know which
    path to hand the task's resources to.
    """
    return backend_name(config) == "podman"


def snapshot_store(config: dict) -> "str | None":
    """Where this config's snapshots live when they are files on this host.

    The podman store, expanded and made absolute, or ``None`` for the microVM
    backend, whose snapshots are AgentENV records. The orchestrator puts it on
    the run's record (``sandbox.created``), so ``harness sweep`` can delete a
    finished run's subvolumes given nothing but its journal.
    """
    if backend_name(config) != "podman":
        return None
    store = _section(config, "podman", PODMAN_KEYS).get("store")
    return str(Path(_expanded(store)).resolve()) if store else None


def _expanded(path) -> str:
    """A path from a config with ``~`` expanded: a YAML file names the home the
    shell would have expanded on a command line."""
    return os.path.expanduser(str(path))


def _section(config: dict, name: str, keys: set) -> dict:
    section = config.get(name) or {}
    if not isinstance(section, dict):
        raise BackendError(f"config key {name!r} must be a mapping of settings")
    unknown = set(section) - keys
    if unknown:
        raise BackendError(
            f"unknown {name} setting(s): {', '.join(sorted(unknown))}; "
            f"known: {', '.join(sorted(keys))}")
    return dict(section)


def _read_api_key(section: dict, env_var: str) -> str:
    """The API key from config, a file, or the environment — in that order.

    A file is offered because a key in a YAML config tends to reach a commit.
    """
    if section.get("api_key"):
        return str(section["api_key"])
    path = section.get("api_key_file")
    if path:
        resolved = Path(os.path.expanduser(str(path)))
        if not resolved.is_file():
            raise BackendError(f"api_key_file not found: {resolved}")
        return resolved.read_text().strip()
    return os.environ.get(env_var, "")


def resolve_microvm_endpoint(section: dict) -> "tuple[str, str]":
    """Where AgentENV is and how to authenticate, as (server_url, api_key).

    One resolver so the pool and the template builder cannot disagree. They did:
    the pool fell back to the environment while the builder required both settings
    in the config, so a configuration that spawned sandboxes failed to build a
    template -- and the failure surfaced as "could not create a sandbox", which
    points at the image rather than at the missing setting.

    ``server_url`` comes back empty when neither config nor environment names one;
    each caller phrases its own error, because what is impossible without it
    differs.
    """
    server_url = section.get("server_url") or os.environ.get("AENV_SERVER_URL") or ""
    return str(server_url), _read_api_key(section, "AENV_API_KEY")


def _microvm(config: dict) -> MicroVMPool:
    section = _section(config, "microvm", MICROVM_KEYS)
    server_url, api_key = resolve_microvm_endpoint(section)
    if not server_url:
        raise BackendError(
            "backend 'microvm' needs microvm.server_url (or AENV_SERVER_URL), "
            f"e.g. {DEFAULT_AENV_SERVER_URL}")
    return MicroVMPool(
        server_url=str(server_url),
        runtime_port=int(section.get("runtime_port", 3000)),
        api_key=api_key,
        request_timeout=float(section.get("request_timeout", 120)),
        sandbox_ttl=int(section.get("sandbox_ttl", 600)),
        allow_internet=(None if section.get("allow_internet") is None
                        else bool(section["allow_internet"])),
    )


def _podman(config: dict) -> PodmanPool:
    section = _section(config, "podman", PODMAN_KEYS)
    # The CLI hands --runtime-bin over at the top level as well, as it does for
    # the template builder; either spelling names the same binary.
    runtime_bin = section.get("runtime_bin") or config.get("runtime_bin")
    if not runtime_bin:
        raise BackendError(
            "backend 'podman' needs podman.runtime_bin: the sprout-runtime binary is "
            "mounted into every container (--runtime-bin runtime/sprout-runtime)")
    if not section.get("store"):
        raise BackendError(
            "backend 'podman' needs podman.store: a directory on a btrfs filesystem "
            "mounted with user_subvol_rm_allowed (sudo ops/btrfs_store.sh <mount point> "
            "<image file> <size>; then --store <mount point>)")
    try:
        return PodmanPool(
            store=_expanded(section["store"]),
            runtime_bin=_expanded(runtime_bin),
            podman_bin=str(section.get("podman_bin") or default_podman_bin()),
            allow_internet=(None if section.get("allow_internet") is None
                            else bool(section["allow_internet"])),
            socket_dir=_expanded(section["socket_dir"]) if section.get("socket_dir") else None,
        )
    except ValueError as error:
        # The pool's own refusals (a missing binary, a store that is not on
        # btrfs) are configuration errors from the harness's point of view.
        raise BackendError(f"backend 'podman': {error}") from error


#: Every backend a config may name, and how its pool is built.
BACKENDS: "dict[str, Callable[[dict], MicroVMPool | PodmanPool]]" = {
    "microvm": _microvm,
    "podman": _podman,
}


def build_pool(config: dict | None = None) -> "MicroVMPool | PodmanPool":
    """The pool this config asks for."""
    config = dict(config or {})
    backend = backend_name(config)
    builder = BACKENDS.get(backend)
    if builder is None:
        raise BackendError(
            f"unknown backend {backend!r}; the backends are "
            + ", ".join(repr(name) for name in BACKENDS))
    return builder(config)
