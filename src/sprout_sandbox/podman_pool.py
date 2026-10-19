"""``PodmanPool``: sandboxes as rootless podman containers whose rootfs is a btrfs subvolume.

The pool keeps the contract of ``MicroVMPool``: the registry of live handles, spawn, snapshots,
destroy, forget and close. It needs ``podman``, ``btrfs-progs`` and a btrfs *store* that
``ops/btrfs_store.sh`` makes once, as root. Everything else runs as the unprivileged user, with the
same runtime binary and the same ``SandboxSession`` as the microVM backend. Each operation is one or
two ``podman`` or ``btrfs`` commands, the ones that a person uses on a leftover container or subvolume.

Snapshots are disk-only. A sandbox started from one has the files and none of the processes, like a
``disk_only`` snapshot of the microVM backend. ``disk_only=False`` raises ``NotImplementedError``.

The pool deletes no snapshot. :meth:`BtrfsStore.delete_snapshot` deletes one, and
``python -m sprout.worker sweep`` calls it for the ids that a run's journal records. A killed process
leaves its container running on a subvolume that nothing else references, and
``python -m sprout.worker reap`` removes it.
"""

from __future__ import annotations

import re
import secrets
import shutil
from pathlib import Path
from typing import Any

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
    """Sandboxes as rootless podman containers on btrfs subvolumes, with the surface of ``MicroVMPool``.

    The session drives either pool without knowing which. ``spawn`` and ``spawn_from_image`` are one
    operation: an image reference or a snapshot id names a read-only subvolume for the rootfs to snapshot."""

    #: Seconds for a command whose duration grows with the image: a pull, the extraction of a base, or
    #: ``podman cp`` of a large file. ``timeout`` bounds every other command.
    BULK_TIMEOUT_S = 1800.0

    def __init__(
        self,
        store: str | Path,
        runtime_bin: str | Path,
        podman_bin: str | None = None,
        allow_internet: bool | None = None,
        socket_dir: str | Path | None = None,
        timeout: float = 120,
    ) -> None:
        """A pool on the btrfs store ``store``. Every container runs the host binary ``runtime_bin``.

        Raises ``ValueError`` if ``runtime_bin`` is not a file, or if ``store_problem`` refuses ``store``.
        ``allow_internet=False`` starts every sandbox offline. ``timeout`` is in seconds, per control command.
        ``podman_bin`` defaults to :func:`default_podman_bin`, and ``socket_dir`` to ``<store>/sockets``."""
        # Absolute, or `-v runtime/sprout-runtime:...` names a podman volume rather than a host file.
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
        self, sandbox: Sandbox, source: str | Path, destination: str, timeout: float | None = None
    ) -> None:
        """``podman cp`` a host file into the container, bypassing the runtime."""
        await self._containers.podman(
            "cp",
            str(source),
            "%s:%s" % (sandbox.require_id(), destination),
            timeout=timeout or self.BULK_TIMEOUT_S,
        )

    async def download_file(
        self, sandbox: Sandbox, source: str, destination: str | Path, timeout: float | None = None
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
        resources: dict[str, Any] | None = None,
    ) -> Sandbox:
        """Start a container from an image reference or a snapshot id, and return a registered handle.

        A snapshot of this store wins over an image of the same name. Raises ``PodmanError`` for a
        snapshot id that the store does not hold. ``resources`` sets ``cpu`` and ``memory_mb``. A subvolume
        has no shape, so a branch passes its parent's again. Calls through the handle act as ``agent_id``."""
        return await self._start(image, env=env, agent_id=agent_id, resources=resources)

    async def spawn_from_image(
        self, image: str, agent_id: str = "", resources: dict[str, Any] | None = None
    ) -> Sandbox:
        """The same as :meth:`spawn`, without ``env``. ``disk_size_mb`` in ``resources`` is ignored.

        The rootfs is a subvolume of the store, with no disk of its own to size. The first use of an image
        pulls it if it is not local and extracts its rootfs into the store. That takes seconds for a slim
        image and minutes for one of several GB."""
        return await self._start(image, agent_id=agent_id, resources=resources)

    async def prepare(self, image: str) -> Path:
        """Pull ``image`` if it is not local and extract its base, and return the base subvolume.

        The call starts no container, and costs little once the base exists. A batch can pay for its
        images here, before an agent waits (``ops/prepull_images.py --backend podman``)."""
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
        resources: dict[str, Any] | None = None,
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
            # The caller sees only the exception and cannot reach this container. Remove it, its
            # network, its rootfs and its socket directory.
            await self._containers.discard(sandbox_id)
            raise
        self._sandboxes[sandbox_id] = sandbox
        self._online[sandbox_id] = online
        return sandbox

    async def _source_for(self, image: str) -> Path:
        """The read-only subvolume that ``image`` names: a snapshot of this store, or the image's base.

        Raises ``PodmanError`` for a name shaped like a snapshot id of this pool that the store does not
        hold. Podman would try to pull such a name as an image, and fail with a registry error."""
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
        """Remove the container, network, rootfs and socket directory of each sandbox that the pool holds.

        The pool drops a sandbox even if its teardown fails half-way, and the ``PodmanError`` names what
        is left. A retry cannot finish the teardown, but ``python -m sprout.worker reap --store`` can."""
        for sb in sandboxes:
            sandbox_id = sb._container_id
            if not sandbox_id or sandbox_id not in self._sandboxes:
                continue
            try:
                await self._containers.discard(sandbox_id)
            finally:
                self._release(sandbox_id, sb)
                await sb.backend.close()

    def attach(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """A registered handle on the running sandbox ``sandbox_id`` of this store, started by another pool.

        The pool then destroys it like its own. The call checks nothing, so a call through the handle
        fails if the sandbox is gone."""
        sandbox = self._handle(sandbox_id, agent_id)
        self._sandboxes[sandbox_id] = sandbox
        return sandbox

    async def forget(self, *sandboxes: Sandbox) -> None:
        """Drop the handles on ``sandboxes`` and leave them running. Closing the pool then leaves them alone.

        This is for a sandbox that must outlive its pool, such as one whose final snapshot failed and that
        so holds the only copy of the rollout. ``python -m sprout.worker reap`` or a person removes it."""
        for sb in sandboxes:
            sandbox_id = sb._container_id
            if sandbox_id and sandbox_id in self._sandboxes:
                self._release(sandbox_id, sb)
                await sb.backend.close()

    def _release(self, sandbox_id: str, sandbox: Sandbox) -> None:
        """Drop everything that this pool tracks about a sandbox. The handle then names no sandbox."""
        del self._sandboxes[sandbox_id]
        self._online.pop(sandbox_id, None)
        self._snapshots_taken.pop(sandbox_id, None)
        sandbox._container_id = None

    async def destroy_all(self) -> None:
        """Destroy every sandbox that the pool holds. Raises one ``PodmanError`` after it tries them all."""
        errors = []
        for sb in self.list():
            try:
                await self.destroy(sb)
            except PodmanError as error:
                errors.append(str(error))
        if errors:
            raise PodmanError("; ".join(errors))

    async def close(self) -> None:
        """Destroy every sandbox that the pool holds, as :meth:`destroy_all` does."""
        await self.destroy_all()

    # --- Images ---

    def image_config(self, sandbox: Sandbox) -> dict[str, Any]:
        """The OCI config of the image that ``sandbox`` comes from: its ``CONFIG_KEYS``, such as ``User``.

        A sandbox started from a snapshot, or attached by id, has the config of the image that the first
        sandbox of its line started from. The call reads only the store."""
        return read_config(self.store.sandboxes / sandbox.require_id())

    # --- Network ---

    async def set_network(self, sandbox: Sandbox, allow: bool) -> None:
        """Connect the running sandbox to its own network, or disconnect it.

        A second call with the same ``allow`` changes nothing. The runtime's socket is a bind mount, so
        the runtime answers in both states. Raises ``PodmanError`` if podman fails."""
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
        """A read-only snapshot of the sandbox's rootfs. The container keeps running.

        It takes milliseconds and reports no layer count. ``spawn(image=<id>)`` starts a new container
        from it. Raises ``NotImplementedError`` unless ``disk_only``, and ``PodmanError`` if the rootfs
        holds subvolumes of its own, because a snapshot would leave out their contents."""
        if not disk_only:
            raise NotImplementedError(
                "podman snapshots are disk-only: a btrfs snapshot holds files, not processes"
            )
        sandbox_id = sandbox.require_id()
        rootfs = self.store.sandboxes / sandbox_id
        # A snapshot shows a nested subvolume as an empty directory. The refusal records the step as not
        # captured, so it never becomes a branch point with files missing.
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
