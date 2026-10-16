"""The podman side of each sandbox: a container over a rootfs with the runtime on a bind-mounted Unix
socket, a bridge network of its own, the readiness probe, failure diagnosis and ordered teardown.

**The runtime is reached over a Unix socket, so egress can be denied.** podman
reads none of the image's config for ``--rootfs``, so the pool passes the saved
``Env`` and ``WorkingDir`` itself and starts the runtime binary, bind-mounted
read-only, as the entrypoint with ``--listen unix:/run/sprout/runtime.sock`` on
a per-sandbox directory mounted from the host. The runtime runs as the
container's root -- under rootless podman, this user -- whatever the image's
``User`` says, exactly as it does in a microVM: the harness drops to that user
per command (``shell_context``), and Harbor's own steps (the ``/logs`` layout,
an overlay's ``RUN``) stay root. The ``User`` is saved beside the base for that
purpose, not passed to ``podman run``; passed, it made the runtime itself
unprivileged and every root step of a trial fail on a non-root image. No port
is published, so a container's network can be taken away: every sandbox has
a bridge network of its own, an offline sandbox is created, disconnected and
only then started (its first command already runs offline), and
:meth:`PodmanPool.set_network` connects or disconnects a running one in about
a tenth of a second while the socket keeps answering.

**One network per sandbox, without podman's DNS server.** A Linux bridge whose
address nobody set takes the lowest MAC among its ports, so on one bridge
shared by every sandbox each container that joined or left could move the
gateway's MAC under all the others, and their packets went nowhere until
their neighbour entry was refreshed: measured on netavark 1.4 (Ubuntu
24.04's), 47 MAC changes in 20 s of churn and every resolver timing out
while it lasted, and 42 of 176 TerminalBench gate trials at 32 workers
logged failed lookups. A sandbox's own bridge has one port. The network is
created with ``--disable-dns`` -- aardvark-dns is one process for every
network and reloads on each membership change -- so ``resolv.conf`` names
the host's upstream resolvers, and with ``isolate=true`` no sandbox reaches
another's. It is named after the sandbox and labelled like its container,
which is how ``harness reap`` finds one that leaked.

**Cleaning up.** Containers carry ``sprout.managed=1`` and ``sprout.store=<store>``,
so ``podman ps -a --filter label=sprout.managed=1`` lists them and the label
says which store holds each one's rootfs -- what ``harness reap`` reads to
remove a leaked container the way :meth:`PodmanPool.destroy` would (``podman rm
-f``, then its network, the subvolume and the socket directory).
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

from .btrfs_store import BtrfsStore
from .podman_cli import PodmanError, remove_container, run_command
from .sandbox import Sandbox

#: Where the runtime binary is mounted inside every container.
RUNTIME_PATH = "/usr/local/bin/sprout-runtime"
#: The per-sandbox socket directory as the container sees it, and the socket in it.
SOCKET_MOUNT = "/run/sprout"
SOCKET_FILE = "runtime.sock"
#: Every container this module starts, and every sandbox's network; the second
#: names the store it runs from.
MANAGED_LABEL = "sprout.managed=1"
STORE_LABEL = "sprout.store"
#: How a sandbox's own network is made (module docstring): no aardvark-dns,
#: and no route to any other sandbox's network.
NETWORK_OPTIONS = ("--disable-dns", "--opt", "isolate=true")
#: The empty file conmon creates in its working directory when a process in
#: the container is OOM-killed; the pool starts containers from the sandbox's
#: socket directory so it lands there.
OOM_MARKER = "oom"


class PodmanContainers:
    """The containers, networks and socket directories of one pool's sandboxes, by sandbox id."""

    #: How long a runtime gets to answer ``tools/list`` after its container
    #: starts. A static binary on a warm host answers in well under a second;
    #: the margin is for a loaded machine.
    READY_TIMEOUT_S = 30.0
    READY_POLL_S = 0.25

    def __init__(
        self, store: BtrfsStore, runtime_bin: Path, podman_bin: str, socket_dir: Path, timeout: float
    ):
        self.store = store
        self.runtime_bin = runtime_bin
        self.podman_bin = podman_bin
        self.socket_dir = socket_dir
        self.timeout = timeout

    async def podman(self, *args: str, timeout: float | None = None, cwd: Path | None = None) -> str:
        """Run podman from the store (or ``cwd``), never from the caller's directory (see :func:`run_command`)."""
        return await run_command(
            [self.podman_bin, *args], timeout=timeout or self.timeout, cwd=cwd or self.store.path
        )

    async def _create_network(self, sandbox_id: str) -> None:
        """The sandbox's own bridge network, named after it (module docstring)."""
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
        """``podman network rm``, never ``-f`` (which removes the containers on it);
        a network that is already gone is not an error."""
        try:
            await self.podman("network", "rm", sandbox_id)
        except PodmanError as error:
            if "network not found" not in str(error):
                raise

    def socket_path(self, sandbox_id: str) -> Path:
        return self.socket_dir / sandbox_id / SOCKET_FILE

    async def start(
        self,
        sandbox_id: str,
        rootfs: Path,
        config: dict,
        env: dict[str, str] | None,
        resources: dict | None,
        *,
        online: bool,
    ) -> None:
        """Start the sandbox's container on ``rootfs`` with its own network, connected when ``online``."""
        socket_dir = self.socket_dir / sandbox_id
        # The runtime binds its socket here as the container's root, which
        # is this user: the directory's owner, with no hand-over needed.
        socket_dir.mkdir()
        await self._create_network(sandbox_id)
        argv = self._run_argv(sandbox_id, rootfs, socket_dir, config, env, resources)
        # From the sandbox's own directory: conmon inherits it, so its
        # `oom` marker (a process in the container was OOM-killed) lands
        # beside the socket, where diagnosis reads it and destroy removes it.
        if online:
            await self.podman("run", "-d", *argv, cwd=socket_dir)
        else:
            # Disconnected before it starts, so the first command already
            # runs offline; `run` would give it the network for an instant.
            await self.podman("create", *argv, cwd=socket_dir)
            await self.podman("network", "disconnect", sandbox_id, sandbox_id)
            await self.podman("start", sandbox_id, cwd=socket_dir)

    def _run_argv(
        self,
        sandbox_id: str,
        rootfs: Path,
        socket_dir: Path,
        config: dict,
        env: dict[str, str] | None,
        resources: dict | None,
    ) -> list[str]:
        """Everything after ``podman run -d`` / ``podman create``.

        Options first: ``--rootfs`` is a flag and the path takes the image's
        positional slot, so anything after it is the command. The image's
        ``User`` is deliberately absent: the runtime runs as the container's
        root and the harness drops to that user per command (module docstring).
        """
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
            "-v",
            "%s:%s:ro" % (self.runtime_bin, RUNTIME_PATH),
            "--entrypoint",
            RUNTIME_PATH,
        ]
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
        """Connect the container to its own network, or disconnect it; podman's
        "already connected" / "not connected" refusals are read as done."""
        try:
            await self.podman("network", "connect" if allow else "disconnect", sandbox_id, sandbox_id)
        except PodmanError as error:
            settled = "already connected" if allow else "is not connected"
            if settled not in str(error):
                raise

    async def wait_for_runtime(self, sandbox: Sandbox) -> None:
        """Block until the container's runtime answers ``tools/list``.

        ``podman run -d`` returns once the container exists, not once the
        process in it listens; adopting the handle earlier would turn the
        first tool call into a connection error. A container that exited
        meanwhile (the runtime failed to exec: a glibc binary in a musl
        image) is reported at once, with its state and logs. The socket is
        created by the container's root, mode 0660: under rootless podman that
        is this user, so it is ours to connect to as soon as it exists.
        """
        sandbox_id = sandbox.require_id()
        deadline = asyncio.get_running_loop().time() + self.READY_TIMEOUT_S
        last_error: BaseException | str = "socket %s not created yet" % self.socket_path(sandbox_id)
        while True:
            try:
                await sandbox.backend.list_tools()
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
        """``"<status> (exit <code>)"`` from one inspect, or what went wrong."""
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
        """Whether the memory cgroup killed a process in the container (conmon's marker, see :func:`run_command`)."""
        return (self.socket_dir / sandbox_id / OOM_MARKER).exists()

    async def discard(self, sandbox_id: str) -> None:
        """Remove a sandbox's container, network, rootfs subvolume and socket directory, in that order.

        The container first: its rootfs is mounted while it exists, a mounted
        subvolume cannot be deleted, and a network with a container on it is
        not removed. A network or rootfs that will not go does not keep the
        other from going; the error names whatever is left.
        """
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


def _resource_flags(resources: dict | None) -> list[str]:
    """``podman run`` flags for a sandbox shape; ``disk_size_mb`` has no counterpart."""
    flags: list[str] = []
    if not resources:
        return flags
    if "cpu" in resources:
        flags += ["--cpus", str(resources["cpu"])]
    if "memory_mb" in resources:
        flags += ["--memory", "%dm" % int(resources["memory_mb"])]
    return flags
