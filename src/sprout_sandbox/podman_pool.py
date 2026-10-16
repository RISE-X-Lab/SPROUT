"""The podman pool: MicroVMPool's contract on rootless podman and btrfs -- the registry of live
handles, online state, sandbox and snapshot ids, spawn, disk-only snapshots, destroy, forget and close.

Sandboxes as rootless podman containers whose rootfs is a btrfs subvolume.

The microVM backend needs an AgentENV server, a patched Firecracker and
``/dev/kvm``. This one needs ``podman``, ``btrfs-progs`` and one directory on a
btrfs filesystem -- the *store*, made once as root by ``ops/btrfs_store.sh`` --
and runs everything else as the unprivileged user, on the same runtime binary,
driven by the same ``SandboxSession``; a branch is a second run started from a
snapshot id with the retained history. It is a client of two CLIs, ``podman``
and ``btrfs``: every operation below is one or two commands, and those are what
a person debugging a leftover container or subvolume reaches for.

**Snapshots are disk-only.** A sandbox started from one has the files and none
of the processes, which is exactly the microVM backend's ``disk_only`` snapshot;
``disk_only=False`` raises ``NotImplementedError`` rather than journal a
"memory snapshot" that restores nothing.

**Cleaning up.** Nothing here deletes snapshots (:meth:`BtrfsStore.delete_snapshot`
does one, and ``harness sweep`` calls it for the ids a run's journal records),
and a killed process leaves its container running on a subvolume nothing else
references.
"""

from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path

from .backends import HttpBackend
from .btrfs_store import BtrfsStore, config_path
from .podman_cli import PodmanError, default_podman_bin
from .podman_containers import PodmanContainers
from .podman_images import ensure_base, read_config
from .pool import Snapshot
from .sandbox import Sandbox

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
#: The ids this pool mints for snapshots: ``<sandbox id>-<counter>[-<name>]``.
_SNAPSHOT_ID = re.compile(r"^sprout-[0-9a-f]{12}-\d{4}(?:-[A-Za-z0-9._-]+)?$")


class PodmanPool:
    """Sandboxes as rootless podman containers on btrfs subvolumes.

    The same surface as :class:`~sprout_sandbox.pool.MicroVMPool`, so the
    session drives either without knowing which. ``spawn`` and
    ``spawn_from_image`` are one operation here: an OCI reference and a
    snapshot id both name a read-only subvolume to snapshot a rootfs from
    (the image's is extracted first when the store lacks it).
    """

    #: Ceiling for the commands whose duration scales with image size: a pull,
    #: the extraction of a base rootfs, ``podman cp`` of a large file.
    #: ``timeout`` bounds the rest.
    BULK_TIMEOUT_S = 1800.0

    def __init__(
        self,
        store: str | Path,
        runtime_bin: str | Path,
        podman_bin: str | None = None,
        allow_internet: bool | None = None,
        socket_dir: str | Path | None = None,
        timeout: float = 120,
    ):
        """
        Args:
            store: A directory on a btrfs filesystem mounted with
                ``user_subvol_rm_allowed`` and owned by this user (see
                ``ops/btrfs_store.sh``). Refused when it is not, either way.
            runtime_bin: The sprout-runtime binary on this host. Bind-mounted
                read-only into every container and started as its entrypoint,
                so an image needs no rebuild -- but the binary has to run
                inside it: a plain ``go build`` links the host's glibc, so a
                musl image (alpine) needs the static ``CGO_ENABLED=0 go build``
                (after ``go generate ./...``, which fetches the ripgrep it embeds).
            podman_bin: The ``podman`` executable (a path, or a name on PATH);
                default ``$SPROUT_PODMAN_BIN``, else ``podman`` (:func:`default_podman_bin`).
            allow_internet: ``None`` and ``True`` start sandboxes connected;
                ``False`` disconnects each one before it starts, so even its
                first command runs offline. :meth:`set_network` changes a
                running sandbox either way.
            socket_dir: Where the per-sandbox socket directories live
                (default ``<store>/sockets``).
            timeout: Seconds one control command (run, inspect, rm, a btrfs
                operation) may take.
        """
        # Absolute, or `-v runtime/sprout-runtime:...` names a podman volume
        # rather than a host file.
        self.runtime_bin = Path(runtime_bin).resolve()
        if not self.runtime_bin.is_file():
            raise ValueError(
                "runtime_bin does not exist: %s (cd runtime && go generate ./... "
                "&& go build -o sprout-runtime .)" % self.runtime_bin
            )
        self.timeout = float(timeout)
        self.store = BtrfsStore(store, timeout=self.timeout)
        self.podman_bin = podman_bin or default_podman_bin()
        self.allow_internet = allow_internet
        self.socket_dir = (Path(socket_dir) if socket_dir else self.store.path / "sockets").resolve()
        self.socket_dir.mkdir(parents=True, exist_ok=True)
        self._containers = PodmanContainers(
            self.store, self.runtime_bin, self.podman_bin, self.socket_dir, self.timeout
        )
        #: Live sandboxes by id, whether each is on its network, and how many
        #: snapshots each has taken (the counter in a snapshot id).
        self._sandboxes: dict[str, Sandbox] = {}
        self._online: dict[str, bool] = {}
        self._snapshots_taken: dict[str, int] = {}

    # --- Capabilities ---

    def supports_snapshot(self) -> bool:
        return True

    def supports_cold_start(self) -> bool:
        return True

    def supports_upload(self) -> bool:
        return True

    def supports_download(self) -> bool:
        return True

    # --- Host files ---

    async def upload_file(
        self, sandbox: Sandbox, source, destination: str, timeout: float | None = None
    ) -> None:
        """``podman cp`` a host file into the container, bypassing the runtime."""
        await self._containers.podman(
            "cp",
            str(source),
            "%s:%s" % (sandbox.require_id(), destination),
            timeout=timeout or self.BULK_TIMEOUT_S,
        )

    async def download_file(
        self, sandbox: Sandbox, source: str, destination, timeout: float | None = None
    ) -> None:
        """``podman cp`` a container file to the host, outside the runtime."""
        await self._containers.podman(
            "cp",
            "%s:%s" % (sandbox.require_id(), source),
            str(destination),
            timeout=timeout or self.BULK_TIMEOUT_S,
        )

    # --- Lifecycle ---

    def list(self) -> list[Sandbox]:
        """Sandboxes this pool currently holds."""
        return list(self._sandboxes.values())

    async def spawn(
        self,
        image: str,
        env: dict[str, str] | None = None,
        agent_id: str = "",
        resources: dict | None = None,
    ) -> Sandbox:
        """Start a container from an image reference or a snapshot id.

        Both name a read-only subvolume to give the container a writable
        snapshot of; a snapshot of this store wins over an image of the same
        name, and a snapshot id this store does not hold (swept, or taken on
        another store) is an error, not an image to pull. ``agent_id`` binds
        the returned handle's identity, so every call
        made through it is attributed to that agent. ``resources`` is the
        shape (``cpu``, ``memory_mb``); a subvolume carries none, so a branch
        that wants its parent's shape passes it again.
        """
        return await self._start(image, env=env, agent_id=agent_id, resources=resources)

    async def spawn_from_image(
        self, image: str, agent_id: str = "", resources: dict | None = None
    ) -> Sandbox:
        """The same as :meth:`spawn`: a reference or a snapshot id, with a shape.

        ``resources`` maps ``cpu`` to ``--cpus`` and ``memory_mb`` to
        ``--memory``; ``disk_size_mb`` is ignored, because the rootfs is a
        subvolume of the store and has no disk of its own to size. The first
        use of an image pulls it when it is not local and extracts its rootfs
        into the store, which takes seconds for a slim image and minutes for
        one of several GB; every later container from that image is a
        snapshot of the extracted base.
        """
        return await self._start(image, agent_id=agent_id, resources=resources)

    async def prepare(self, image: str) -> Path:
        """Pull ``image`` when it is not local and extract its base; start nothing.

        The first spawn of an image does this anyway, and pays seconds for a
        slim image or minutes for one of several GB inside the first trial of
        a task. A batch does better paying for every image up front, when no
        agent is waiting on it (``ops/prepull_images.py --backend podman``).
        Returns the base subvolume; free once it exists.
        """
        return await ensure_base(
            self.store,
            image,
            podman_bin=self.podman_bin,
            timeout=self.timeout,
            bulk_timeout=self.BULK_TIMEOUT_S,
        )

    async def _start(
        self,
        image: str,
        *,
        env: dict[str, str] | None = None,
        agent_id: str = "",
        resources: dict | None = None,
    ) -> Sandbox:
        if not image:
            raise ValueError("spawn needs an image reference or a snapshot id")
        source = await self._source_for(image)
        sandbox_id = "sprout-" + secrets.token_hex(6)
        rootfs = self.store.sandboxes / sandbox_id
        await self.store.snapshot(source, rootfs)
        online = self.allow_internet is not False
        try:
            shutil.copyfile(config_path(source), config_path(rootfs))
            config = read_config(rootfs)
            await self._containers.start(sandbox_id, rootfs, config, env, resources, online=online)
            sandbox = self._handle(sandbox_id, agent_id)
            await self._containers.wait_for_runtime(sandbox)
        except BaseException:
            # Nothing can reach this container; do not leak it, its network,
            # its rootfs or its socket directory while the caller sees only an
            # exception.
            await self._containers.discard(sandbox_id)
            raise
        self._sandboxes[sandbox_id] = sandbox
        self._online[sandbox_id] = online
        return sandbox

    async def _source_for(self, image: str) -> Path:
        """The read-only subvolume ``image`` names: a snapshot of this store, or the image's base.

        A name shaped like one of this pool's snapshot ids that the store does
        not hold is refused here: handed to podman it would be pulled as an
        image and fail with a registry error that never mentions snapshots.
        """
        if "/" not in image:
            snapshot = self.store.snapshots / image
            if snapshot.is_dir():
                return snapshot
            if _SNAPSHOT_ID.match(image):
                raise PodmanError(
                    "no snapshot %r in %s (deleted, swept, or taken on another store)"
                    % (image, self.store.snapshots)
                )
        return await self.prepare(image)

    async def destroy(self, *sandboxes: Sandbox) -> None:
        """Tear down specific sandboxes: the container, then its network, rootfs and socket directory.

        A sandbox whose teardown fails half-way is dropped all the same and
        the error names what is left: retrying cannot finish the job (``close``
        would raise again and again), while ``harness reap --store`` finds a
        rootfs without a container in the store, and a network by its label.
        """
        for sb in sandboxes:
            sandbox_id = sb._container_id
            if not sandbox_id or sandbox_id not in self._sandboxes:
                continue
            try:
                await self._containers.discard(sandbox_id)
            finally:
                self._release(sandbox_id, sb)
                await sb.backend.close()

    async def forget(self, *sandboxes: Sandbox) -> None:
        """Drop the handles and leave the sandboxes running.

        For a sandbox that must outlive its pool: one whose final snapshot
        failed holds the only copy of the rollout, and Harbor's ``stop`` keeps
        it for ``harness reap`` or a person. Closing the pool destroys every
        sandbox it still holds, which a forgotten one no longer is.
        """
        for sb in sandboxes:
            sandbox_id = sb._container_id
            if sandbox_id and sandbox_id in self._sandboxes:
                self._release(sandbox_id, sb)
                await sb.backend.close()

    def _release(self, sandbox_id: str, sandbox: Sandbox) -> None:
        """Drop everything this pool tracks about a sandbox; the handle no longer names it."""
        del self._sandboxes[sandbox_id]
        self._online.pop(sandbox_id, None)
        self._snapshots_taken.pop(sandbox_id, None)
        sandbox._container_id = None

    async def destroy_all(self) -> None:
        """Tear down everything this pool created; every sandbox is tried before an error is raised."""
        errors = []
        for sb in self.list():
            try:
                await self.destroy(sb)
            except PodmanError as error:
                errors.append(str(error))
        if errors:
            raise PodmanError("; ".join(errors))

    async def close(self) -> None:
        await self.destroy_all()

    # --- Network ---

    async def set_network(self, sandbox: Sandbox, allow: bool) -> None:
        """Connect the running sandbox to its own network, or disconnect it.

        Idempotent: the state is tracked, and podman's own "already
        connected" / "not connected" refusals are read as done. The runtime's
        socket is a bind mount, so it answers either way.
        """
        sandbox_id = sandbox.require_id()
        allow = bool(allow)
        if self._online.get(sandbox_id) is allow:
            return
        await self._containers.connect(sandbox_id, allow)
        self._online[sandbox_id] = allow

    # --- Snapshots ---

    async def snapshot(
        self, sandbox: Sandbox, *, name: str | None = None, disk_only: bool = True
    ) -> Snapshot:
        """A read-only snapshot of the sandbox's rootfs; the container keeps running.

        Milliseconds, whatever the rootfs holds: a btrfs snapshot shares every
        unchanged extent with its source. The id is the subvolume's name --
        the sandbox id, a counter and the sanitised ``name`` when one is
        given -- and ``spawn(image=<id>)`` later starts a fresh container from
        it. Only ``disk_only`` is possible here; nothing stacks, so the
        returned snapshot reports no layer count. A rootfs holding subvolumes
        of its own is refused: a snapshot of it would show each as an empty
        directory (``btrfs_store`` module docstring), and a refused capture is recorded as
        such rather than becoming a branch point with files missing.
        """
        if not disk_only:
            raise NotImplementedError(
                "podman snapshots are disk-only: a btrfs snapshot holds files, not processes"
            )
        sandbox_id = sandbox.require_id()
        rootfs = self.store.sandboxes / sandbox_id
        nested = await self.store.nested_subvolumes(rootfs, podman_bin=self.podman_bin)
        if nested:
            raise PodmanError(
                "rootfs of %s holds nested subvolumes (%s); a snapshot would omit their contents"
                % (sandbox_id, ", ".join("/" + str(path.relative_to(rootfs)) for path in nested))
            )
        count = self._snapshots_taken.get(sandbox_id, 0) + 1
        snapshot_id = "%s-%04d" % (sandbox_id, count)
        if name:
            snapshot_id += "-" + _safe(name)
        target = self.store.snapshots / snapshot_id
        await self.store.snapshot(rootfs, target, read_only=True)
        shutil.copyfile(config_path(rootfs), config_path(target))
        self._snapshots_taken[sandbox_id] = count
        return Snapshot(id=snapshot_id, disk_only=True)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """:meth:`BtrfsStore.delete_snapshot` on this pool's store."""
        await self.store.delete_snapshot(snapshot_id)

    # --- Handles ---

    def _handle(self, sandbox_id: str, agent_id: str) -> Sandbox:
        sb = Sandbox(
            backend=HttpBackend("http://sprout-runtime", uds=str(self._containers.socket_path(sandbox_id))),
            agent_id=agent_id,
        )
        sb._container_id = sandbox_id
        return sb


def _safe(name: str) -> str:
    """A caller's snapshot name as a path component: letters, digits, ``._-``."""
    return _UNSAFE.sub("-", name).strip("-.")[:48] or "snapshot"
