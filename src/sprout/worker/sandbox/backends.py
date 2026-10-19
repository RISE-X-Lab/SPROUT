"""The sandbox pool that a config's ``execution`` section asks for, built in one place.

:func:`build_pool` reads ``backend``, then the section of that backend::

    execution:
      backend: podman                   # the default when none is named
      podman:
        store: /path/to/sprout-btrfs      # a directory on btrfs (ops/btrfs_store.sh)
        runtime_bin: runtime/sprout-runtime

Every run leaves snapshots, so each backend takes them:

- ``podman``: rootless containers on this host, each on a btrfs subvolume of the store, with
  disk-only snapshots and no service to run.
- ``microvm``: Firecracker VMs from an AgentENV server, with memory snapshots and chain
  compaction on the server.

An unknown key in a section raises :class:`BackendError`, so a mistyped ``server_url`` does
not look like a connection failure. The two backends in use: ``docs/EXPERIMENTS.md``, section 8.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sprout.contract.backend import PODMAN_KEYS, BackendError, backend_name, backend_section, expanded_path
from sprout_sandbox import MicroVMPool, PodmanPool
from sprout_sandbox.podman_cli import default_podman_bin

__all__ = [
    "build_pool",
    "BACKENDS",
    "resolve_microvm_endpoint",
    "with_egress",
    "with_sandbox_budget",
    "spawn_takes_shape",
    "default_aenv_server_url",
]


#: Seconds that a microVM lease adds to an operation's budget, for provisioning and teardown.
VM_LIFETIME_MARGIN_S = 600
#: Where an AgentENV server listens unless it is told otherwise: its own ``API_ADDR``
#: default, port 8000. A server elsewhere is named in ``AENV_SERVER_URL``.
DEFAULT_AENV_SERVER_URL = "http://127.0.0.1:8000"


def default_aenv_server_url() -> str:
    """The AgentENV server when none is named: ``$AENV_SERVER_URL``, else :data:`DEFAULT_AENV_SERVER_URL`.

    Each caller that can assume a server reads it here, so that they agree. :func:`build_pool`
    assumes none: it refuses a config that names no server."""
    return os.environ.get("AENV_SERVER_URL") or DEFAULT_AENV_SERVER_URL


def with_egress(config: dict[str, Any], *, allow: bool) -> dict[str, Any]:
    """``config`` with its sandboxes starting connected (``allow``) or offline, on either backend."""
    name = backend_name(config)
    return {**config, name: {**(config.get(name) or {}), "allow_internet": allow}}


def with_sandbox_budget(config: dict[str, Any], budget_s: float) -> dict[str, Any]:
    """``config`` with a microVM lease that covers an operation of ``budget_s`` seconds.

    A missing or shorter ``sandbox_ttl`` becomes the budget plus :data:`VM_LIFETIME_MARGIN_S`, and
    a longer one stays. A podman config has no lease and comes back unchanged. Raises BackendError
    for a budget that is not finite and positive, or a ``sandbox_ttl`` that is negative or not finite."""
    result = dict(config)
    if backend_name(result) != "microvm":
        return result
    if (
        isinstance(budget_s, bool)
        or not isinstance(budget_s, (int, float))
        or not math.isfinite(budget_s)
        or budget_s <= 0
    ):
        raise BackendError("VM lifetime requires a finite positive operation budget")
    section = backend_section(config, "microvm", MICROVM_KEYS)
    explicit = section.get("sandbox_ttl", 0)
    if (
        isinstance(explicit, bool)
        or not isinstance(explicit, (int, float))
        or not math.isfinite(explicit)
        or explicit < 0
    ):
        raise BackendError("microvm.sandbox_ttl must be a finite non-negative number")
    section["sandbox_ttl"] = max(math.ceil(explicit), math.ceil(budget_s) + VM_LIFETIME_MARGIN_S)
    result["microvm"] = section
    return result


#: Recognised keys of the ``microvm`` section, so a typo is reported rather than ignored.
MICROVM_KEYS = {
    "server_url",
    "runtime_port",
    "api_key",
    "api_key_file",
    "request_timeout",
    "sandbox_ttl",
    # True = image names are OCI references to cold-start, such as a benchmark's per-instance
    # images, not names of snapshots in the catalog. The string alone cannot tell which.
    "from_image",
    # A local sprout-runtime binary. If set, a template is built for each image on demand
    # (sprout.worker.sandbox.templates). If unset, the templates must exist already.
    "runtime_bin",
    # False = no egress from any sandbox that this pool creates (AgentENV `allowInternetAccess`).
    # Unset keeps the server's default.
    "allow_internet",
    # True = templates start the runtime with the image's OCI ENV, read with regctl, not with
    # the PATH that the guest agent rebuilds. Opt-in, because it changes the template names.
    "image_env",
    # True = the agent's commands, solve.sh and test.sh run as the image's USER, read
    # with regctl (worker/sandbox/image_shell.py). With image_env, they also get its ENV.
    "image_user",
}


def spawn_takes_shape(config: dict[str, Any]) -> bool:
    """Whether every spawn on this backend takes the sandbox's shape, ``resources``.

    True for podman: a subvolume carries no shape, so a spawn from an image or from a snapshot
    takes ``resources``, and a branch passes its parent's again. A microVM template or snapshot
    has its shape built in, and only a cold start from an image chooses one."""
    return backend_name(config) == "podman"


def _read_api_key(section: dict[str, Any], env_var: str) -> str:
    """The API key from config, a file, or the environment, in that order.

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


def resolve_microvm_endpoint(section: dict[str, Any]) -> tuple[str, str]:
    """Where AgentENV is and how to authenticate, as ``(server_url, api_key)``.

    Every user of the endpoint resolves it here, so a config that spawns sandboxes also builds
    templates. ``server_url`` is empty when neither the section nor ``AENV_SERVER_URL`` names a
    server. Each caller then raises its own error."""
    server_url = section.get("server_url") or os.environ.get("AENV_SERVER_URL") or ""
    return str(server_url), _read_api_key(section, "AENV_API_KEY")


def _microvm(config: dict[str, Any]) -> MicroVMPool:
    section = backend_section(config, "microvm", MICROVM_KEYS)
    server_url, api_key = resolve_microvm_endpoint(section)
    if not server_url:
        raise BackendError(
            f"backend 'microvm' needs microvm.server_url (or AENV_SERVER_URL), e.g. {DEFAULT_AENV_SERVER_URL}"
        )
    return MicroVMPool(
        server_url=str(server_url),
        runtime_port=int(section.get("runtime_port", 3000)),
        api_key=api_key,
        request_timeout=float(section.get("request_timeout", 120)),
        sandbox_ttl=int(section.get("sandbox_ttl", 600)),
        allow_internet=(None if section.get("allow_internet") is None else bool(section["allow_internet"])),
    )


def _podman(config: dict[str, Any]) -> PodmanPool:
    section = backend_section(config, "podman", PODMAN_KEYS)
    # The CLI also passes --runtime-bin at the top level, as it does for the template builder.
    # Both spellings name the same binary.
    runtime_bin = section.get("runtime_bin") or config.get("runtime_bin")
    if not runtime_bin:
        raise BackendError(
            "backend 'podman' needs podman.runtime_bin: the sprout-runtime binary is "
            "mounted into every container (--runtime-bin runtime/sprout-runtime)"
        )
    if not section.get("store"):
        raise BackendError(
            "backend 'podman' needs podman.store: a directory on a btrfs filesystem "
            "mounted with user_subvol_rm_allowed (sudo ops/btrfs_store.sh <mount point> "
            "<image file> <size>; then --store <mount point>)"
        )
    try:
        return PodmanPool(
            store=expanded_path(section["store"]),
            runtime_bin=expanded_path(runtime_bin),
            podman_bin=str(section.get("podman_bin") or default_podman_bin()),
            allow_internet=(
                None if section.get("allow_internet") is None else bool(section["allow_internet"])
            ),
            socket_dir=expanded_path(section["socket_dir"]) if section.get("socket_dir") else None,
        )
    except ValueError as error:
        # The pool's own refusals, such as a missing binary or a store that is not on btrfs,
        # are configuration errors for the caller.
        raise BackendError(f"backend 'podman': {error}") from error


#: Every backend a config may name, and how its pool is built.
BACKENDS: dict[str, Callable[[dict[str, Any]], MicroVMPool | PodmanPool]] = {
    "microvm": _microvm,
    "podman": _podman,
}


def build_pool(config: dict[str, Any] | None = None) -> MicroVMPool | PodmanPool:
    """The pool that ``config`` asks for. Raises BackendError for an unknown backend or a bad section."""
    config = dict(config or {})
    backend = backend_name(config)
    builder = BACKENDS.get(backend)
    if builder is None:
        raise BackendError(
            f"unknown backend {backend!r}; the backends are " + ", ".join(repr(name) for name in BACKENDS)
        )
    return builder(config)
