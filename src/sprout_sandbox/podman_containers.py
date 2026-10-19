"""``PodmanContainers``: the container, network and socket directory of each sandbox of a pool.

A container runs on a rootfs subvolume, with the runtime as its entrypoint. The runtime listens on a
Unix socket in a directory that the host bind-mounts, so no port is published, and a sandbox can go
offline while the socket keeps answering. The runtime runs as the container's root, which is this
user under rootless podman, whatever the image's ``User`` says. The harness drops to that user for
each command (``ShellContext``), and Harbor's own steps stay root.

Each sandbox has a bridge network of its own, without podman's DNS server and with no route to the
network of another sandbox. An offline sandbox is created, disconnected and only then started, so its
first command already runs offline. :meth:`PodmanPool.set_network` connects or disconnects a running
sandbox in about a tenth of a second.

Each container and each network carries ``sprout.managed=1`` and ``sprout.store=<store>``.
``python -m sprout.worker reap`` finds a leaked one by these labels and removes it in the order that
:meth:`PodmanContainers.discard` uses. Measurements: ``docs/HARBOR_PLUGIN.md``.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

from .backends import HttpBackend
from .btrfs_store import BtrfsStore
from .podman_cli import PodmanError, remove_container, run_command
from .sandbox import Sandbox

#: Where the runtime binary is mounted inside every container.
RUNTIME_PATH = "/usr/local/bin/sprout-runtime"
#: The per-sandbox socket directory as the container sees it, and the socket in it.
SOCKET_MOUNT = "/run/sprout"
SOCKET_FILE = "runtime.sock"
#: The labels of every container that this module starts and of every sandbox's network.
#: ``STORE_LABEL`` names the store that the sandbox runs from.
MANAGED_LABEL = "sprout.managed=1"
STORE_LABEL = "sprout.store"
#: No aardvark-dns, which serves every network and reloads on each change of membership, so
#: ``resolv.conf`` names the host's resolvers. No route to the network of any other sandbox.
NETWORK_OPTIONS = ("--disable-dns", "--opt", "isolate=true")
#: The empty file that conmon creates in its working directory when the kernel OOM-kills a process
#: in the container. The pool starts each container from its socket directory, so the file lands there.
OOM_MARKER = "oom"


class PodmanContainers:
    """The containers, networks and socket directories of one pool's sandboxes, by sandbox id."""

    #: Seconds that a runtime gets to answer ``tools/list`` after its container starts. A static binary
    #: on a warm host answers in well under a second, and the rest is margin for a loaded host.
    READY_TIMEOUT_S = 30.0
    READY_POLL_S = 0.25

    def __init__(
        self, store: BtrfsStore, runtime_bin: Path, podman_bin: str, socket_dir: Path, timeout: float
    ) -> None:
        self.store = store
        self.runtime_bin = runtime_bin
        self.podman_bin = podman_bin
        self.socket_dir = socket_dir
        self.timeout = timeout

    async def podman(self, *args: str, timeout: float | None = None, cwd: Path | None = None) -> str:
        """The stdout of podman with ``args``, run from ``cwd`` or else the store.

        It never runs from the caller's directory, where conmon could leave its ``oom`` marker (see
        :func:`run_command`). Raises ``PodmanError`` if the command fails."""
        return await run_command(
            [self.podman_bin, *args], timeout=timeout or self.timeout, cwd=cwd or self.store.path
        )

    async def _create_network(self, sandbox_id: str) -> None:
        """The sandbox's own bridge network, named after the sandbox."""
        # One bridge per sandbox: on a shared one, a container that joins or leaves can move the gateway
        # MAC, and the others lose traffic until their neighbour entries refresh.
        await self.podman(
            "network",
            "create",
            *NETWORK_OPTIONS,
            "--label",
            MANAGED_LABEL,
            "--label",
            "%s=%s" % (STORE_LABEL, self.store.path),
            sandbox_id,
        )

    async def _remove_network(self, sandbox_id: str) -> None:
        """``podman network rm`` without ``-f``, which would also remove the containers on the network.

        A network that is already gone is not an error."""
        try:
            await self.podman("network", "rm", sandbox_id)
        except PodmanError as error:
            if "network not found" not in str(error):
                raise

    def socket_path(self, sandbox_id: str) -> Path:
        """The host path of the runtime's socket in the sandbox ``sandbox_id``."""
        return self.socket_dir / sandbox_id / SOCKET_FILE

    async def start(
        self,
        sandbox_id: str,
        rootfs: Path,
        config: dict[str, Any],
        env: dict[str, str] | None,
        resources: dict[str, Any] | None,
        *,
        online: bool,
    ) -> None:
        """Start the sandbox's container on ``rootfs`` with its own network, connected when ``online``."""
        socket_dir = self.socket_dir / sandbox_id
        # The runtime binds its socket here as the container's root. That root is this user, who
        # owns the directory, so no change of owner is needed.
        socket_dir.mkdir()
        await self._create_network(sandbox_id)
        argv = self._run_argv(sandbox_id, rootfs, socket_dir, config, env, resources)
        # Run podman from the sandbox's directory. conmon inherits it, so its `oom` marker lands
        # beside the socket, where `diagnosis` reads it and `discard` removes it.
        if online:
            await self.podman("run", "-d", *argv, cwd=socket_dir)
        else:
            # Disconnect before the start, so that the first command already runs offline. `run`
            # would give the container the network for an instant.
            await self.podman("create", *argv, cwd=socket_dir)
            await self.podman("network", "disconnect", sandbox_id, sandbox_id)
            await self.podman("start", sandbox_id, cwd=socket_dir)

    def _run_argv(
        self,
        sandbox_id: str,
        rootfs: Path,
        socket_dir: Path,
        config: dict[str, Any],
        env: dict[str, str] | None,
        resources: dict[str, Any] | None,
    ) -> list[str]:
        """The arguments after ``podman run -d`` or ``podman create``.

        The options come first: ``--rootfs`` is a flag, its path takes the image's positional slot, and
        the rest is the command. ``User`` is left out, so the runtime stays root for a trial's root steps."""
        argv = [
            "--name",
            sandbox_id,
            "--network",
            sandbox_id,
            "--label",
            MANAGED_LABEL,
            "--label",
            "%s=%s" % (STORE_LABEL, self.store.path),
            "-v",
            "%s:%s" % (socket_dir, SOCKET_MOUNT),
            # Every image runs the host's binary. A musl image such as alpine needs the static
            # `CGO_ENABLED=0 go build`, because a plain build links the host's glibc.
            "-v",
            "%s:%s:ro" % (self.runtime_bin, RUNTIME_PATH),
            "--entrypoint",
            RUNTIME_PATH,
        ]
        # podman reads none of the image's config for --rootfs, so the saved Env and WorkingDir go here.
        for entry in config.get("Env") or []:
            argv += ["-e", entry]
        for key, value in (env or {}).items():
            argv += ["-e", "%s=%s" % (key, value)]
        if config.get("WorkingDir"):
            argv += ["--workdir", config["WorkingDir"]]
        argv += _resource_flags(resources)
        argv += ["--rootfs", str(rootfs), "--listen", "unix:%s/%s" % (SOCKET_MOUNT, SOCKET_FILE)]
        return argv

    async def connect(self, sandbox_id: str, allow: bool) -> None:
        """Connect the container to its own network, or disconnect it.

        Podman's refusals "already connected" and "is not connected" count as done."""
        try:
            await self.podman("network", "connect" if allow else "disconnect", sandbox_id, sandbox_id)
        except PodmanError as error:
            settled = "already connected" if allow else "is not connected"
            if settled not in str(error):
                raise

    async def wait_for_runtime(self, sandbox: Sandbox) -> None:
        """Return once the container's runtime answers ``tools/list``.

        Raises ``PodmanError`` at once if the container exits first, for example because a glibc runtime
        cannot run in a musl image. Raises it after ``READY_TIMEOUT_S`` if the runtime never answers.
        Both errors carry the container's state and logs."""
        sandbox_id = sandbox.require_id()
        backend = sandbox.backend
        if not isinstance(backend, HttpBackend):
            raise TypeError("a podman sandbox's runtime is reached through an HttpBackend")
        # `podman run -d` returns before the runtime listens. A handle adopted earlier would fail its
        # first tool call with a connection error.
        deadline = asyncio.get_running_loop().time() + self.READY_TIMEOUT_S
        last_error: BaseException | str = "socket %s not created yet" % self.socket_path(sandbox_id)
        while True:
            # The container's root creates the socket with mode 0660. Under rootless podman that root
            # is this user, so the probe can connect as soon as the socket exists.
            try:
                await backend.list_tools()
                return
            except Exception as error:  # noqa: BLE001 - a probe, retried
                last_error = error
            state = await self.state(sandbox_id)
            if state.startswith(("exited", "stopped")):
                raise PodmanError(
                    "runtime in %s exited before answering; %s"
                    % (sandbox_id, await self.diagnosis(sandbox_id))
                )
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(self.READY_POLL_S)
        raise PodmanError(
            "runtime in %s did not answer within %.0f s (%s); %s"
            % (sandbox_id, self.READY_TIMEOUT_S, last_error, await self.diagnosis(sandbox_id))
        )

    async def state(self, container: str) -> str:
        """The container's state as ``"<status> (exit <code>)"``, or ``"state unknown (<error>)"``."""
        try:
            raw = await self.podman(
                "inspect",
                "--type",
                "container",
                "--format",
                "{{.State.Status}} {{.State.ExitCode}}",
                container,
            )
        except PodmanError as error:
            return "state unknown (%s)" % error
        status, _, code = raw.strip().partition(" ")
        return "%s (exit %s)" % (status or "unknown", code or "?")

    async def diagnosis(self, container: str) -> str:
        """The container's state, whether it was OOM-killed, and its last log lines, for an error message."""
        logs = ""
        try:
            logs = await self.podman("logs", "--tail", "20", container)
        except PodmanError:
            pass
        oom = (
            "; a process in it was OOM-killed (conmon's %s marker)" % OOM_MARKER
            if self.oom_killed(container)
            else ""
        )
        return "container %s%s; logs: %s" % (
            await self.state(container),
            oom,
            logs.strip() or "(none)",
        )

    def oom_killed(self, sandbox_id: str) -> bool:
        """Whether the memory cgroup killed a process in the container, by conmon's ``oom`` marker."""
        return (self.socket_dir / sandbox_id / OOM_MARKER).exists()

    async def discard(self, sandbox_id: str) -> None:
        """Remove a sandbox's container, network, rootfs subvolume and socket directory, in that order.

        A part that is already gone is not an error. If the network or the rootfs does not go, the other
        still goes, and the ``PodmanError`` names what is left."""
        # The container goes first. Its rootfs is mounted while it exists, a mounted subvolume cannot be
        # deleted, and podman does not remove a network that has a container on it.
        await remove_container(self.podman_bin, sandbox_id, timeout=self.timeout, cwd=self.store.path)
        rootfs = self.store.sandboxes / sandbox_id
        left = []
        try:
            await self._remove_network(sandbox_id)
        except PodmanError as error:
            left.append("its network %s is not (%s)" % (sandbox_id, error))
        try:
            await self.store.discard_rootfs(rootfs, podman_bin=self.podman_bin)
        except PodmanError as error:
            left.append("its rootfs %s is not (%s)" % (rootfs, error))
        shutil.rmtree(self.socket_dir / sandbox_id, ignore_errors=True)
        if left:
            raise PodmanError(
                "container %s is removed but %s; `harness reap --store %s` reclaims %s"
                % (sandbox_id, " and ".join(left), self.store.path, "it" if len(left) == 1 else "them")
            )


def _resource_flags(resources: dict[str, Any] | None) -> list[str]:
    """``podman run`` flags for a sandbox shape. ``disk_size_mb`` has no counterpart."""
    flags: list[str] = []
    if not resources:
        return flags
    if "cpu" in resources:
        flags += ["--cpus", str(resources["cpu"])]
    if "memory_mb" in resources:
        flags += ["--memory", "%dm" % int(resources["memory_mb"])]
    return flags
