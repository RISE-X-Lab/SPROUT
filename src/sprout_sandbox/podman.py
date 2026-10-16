"""Sandboxes as rootless podman containers whose rootfs is a btrfs subvolume.

The microVM backend needs an AgentENV server, a patched Firecracker and
``/dev/kvm``. This one needs ``podman``, ``btrfs-progs`` and one directory on a
btrfs filesystem -- the *store*, made once as root by ``ops/btrfs_store.sh`` --
and runs everything else as the unprivileged user, on the same runtime binary,
driven by the same ``SandboxSession``; a branch is a second run started from a
snapshot id with the retained history. It is a client of two CLIs, ``podman``
and ``btrfs``: every operation below is one or two commands, and those are what
a person debugging a leftover container or subvolume reaches for.

**The image is extracted once; every sandbox is a snapshot of it.** podman keeps
an image as a stack of layers and snapshots a container with ``commit``, which
copies the whole writable layer, stacks a layer per generation of branches and
never compacts (the first version of this backend did that: 30 s per commit of
a 1.5 GB layer, a 500-layer ceiling and a squash policy to stay under it). A
btrfs snapshot is a metadata operation -- milliseconds, whatever the rootfs
holds -- and each one is an independent subvolume, so nothing stacks and
nothing needs squashing: ``Snapshot.rootfs_layers`` is ``None`` here. The pool
extracts an image's rootfs into a read-only *base* subvolume once (``podman
export`` piped through ``tar`` under ``podman unshare``, so file ownership maps
into the user namespace exactly as podman's own storage does), gives each
container a writable snapshot of a base as its ``--rootfs``, and a checkpoint
is a read-only snapshot of that writable subvolume, taken while the container
runs and with its processes untouched. Spawning from a snapshot id is the same
operation as spawning from an image: a writable snapshot of a read-only
subvolume plus a new container.

**Snapshots are disk-only.** A sandbox started from one has the files and none
of the processes, which is exactly the microVM backend's ``disk_only`` snapshot;
``disk_only=False`` raises ``NotImplementedError`` rather than journal a
"memory snapshot" that restores nothing.

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

**Layout of the store**, all of it owned by the user and every subvolume with
the image config it was extracted from saved beside it as ``<name>.json``::

    <store>/bases/<image id>            read-only, one per image (Id from podman image inspect)
    <store>/sandboxes/<sandbox id>      writable, one per container: its rootfs
    <store>/snapshots/<snapshot id>     read-only; the id is "<sandbox id>-<counter>[-<name>]"
    <store>/sockets/<sandbox id>/runtime.sock   (socket_dir; the container sees /run/sprout)

An extraction that dies half-way leaves ``bases/<image id>.partial-<token>``;
a complete base is published by renaming, so a base that exists is whole. Its
config also records the extraction format (``extract``); a base extracted by an
older format is extracted again (format 1 ran tar without ``--xattrs`` and lost
every extended attribute, file capabilities included).

**A subvolume made inside a sandbox is not snapshotted.** ``btrfs subvolume
snapshot`` is not recursive: a subvolume the container created (an
unprivileged ioctl -- ``btrfs subvolume create``, podman-in-podman on btrfs,
snapper) would appear in the snapshot as an empty directory. The pool looks for
them first (``find -inum 256`` inside the user namespace, since a 0700
directory of the container's user is readable only there; about a tenth of a
second on a 100k-inode rootfs) and refuses the snapshot, so the checkpointer
records the step as not captured rather than a branch point with files
missing; ``destroy`` deletes them, deepest first, before the rootfs.

**Cleaning up.** Nothing here deletes snapshots (:meth:`BtrfsStore.delete_snapshot`
does one, and ``harness sweep`` calls it for the ids a run's journal records),
and a killed process leaves its container running on a subvolume nothing else
references. Containers carry ``sprout.managed=1`` and ``sprout.store=<store>``,
so ``podman ps -a --filter label=sprout.managed=1`` lists them and the label
says which store holds each one's rootfs -- what ``harness reap`` reads to
remove a leaked container the way :meth:`PodmanPool.destroy` would (``podman rm
-f``, then its network, the subvolume and the socket directory). The store's directories
are the catalog of subvolumes (``btrfs subvolume list`` needs root in
btrfs-progs 6.6; a subvolume's root directory has inode 256), and a read-only
one is deleted only after ``btrfs property set <path> ro false``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
from pathlib import Path

from .backends import HttpBackend
from .pool import Snapshot
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
#: What a container started from a bare rootfs needs from the image's OCI config
#: (podman reads none of it for ``--rootfs``); saved beside every subvolume.
CONFIG_KEYS = ("Env", "WorkingDir", "User", "Entrypoint", "Cmd")
#: The format of an extracted base, recorded as ``extract`` in its config; a
#: base whose record says less is extracted again. Format 1 ran tar without
#: ``--xattrs`` and so lost every extended attribute -- file capabilities
#: (``ping``, ``node``, gosu-style helpers) among them.
EXTRACTION_FORMAT = 2
#: The empty file conmon creates in its working directory when a process in
#: the container is OOM-killed; the pool starts containers from the sandbox's
#: socket directory so it lands there.
OOM_MARKER = "oom"
#: The mount option that lets the store's unprivileged owner delete the
#: subvolumes it created; without it every spawn works and the first destroy
#: fails after the container is gone.
REQUIRED_MOUNT_OPTION = "user_subvol_rm_allowed"
#: Names the podman executable every entry point runs when no ``podman_bin`` is
#: given; ``python -m sprout.services env`` exports it for an experiment shell. The
#: bare name ``podman`` means the one on PATH (what the fake-tool tests set).
PODMAN_BIN_ENV = "SPROUT_PODMAN_BIN"
#: Where ``ops/setup_podman.sh`` installs podman 6, relative to the home directory: a
#: static build, off PATH so a distribution podman is left alone.
#: ``sprout.services.spec.installed_podman`` names the same path (a test keeps them in step).
INSTALLED_PODMAN = (".local", "opt", "podman", "bin", "podman")


def installed_podman() -> Path:
    return Path.home().joinpath(*INSTALLED_PODMAN)


def default_podman_bin() -> str:
    """``$SPROUT_PODMAN_BIN`` when set (``~`` expanded), else the podman
    ``ops/setup_podman.sh`` installed when it is executable, else ``podman`` on PATH.

    One rule for every entry point (the pool, ``image_info``, the reaper, the Run
    Store worker and its children, the live tests), so two podmans never share a
    store by accident.
    """
    chosen = os.environ.get(PODMAN_BIN_ENV)
    if chosen:
        return os.path.expanduser(chosen)
    installed = installed_podman()
    if installed.is_file() and os.access(installed, os.X_OK):
        return str(installed)
    return "podman"


_HEX_ID = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
#: The ids this pool mints for snapshots: ``<sandbox id>-<counter>[-<name>]``.
_SNAPSHOT_ID = re.compile(r"^sprout-[0-9a-f]{12}-\d{4}(?:-[A-Za-z0-9._-]+)?$")


class PodmanError(RuntimeError):
    """A podman or btrfs command failed; the message carries its stderr."""


async def _command(argv: list[str], *, timeout: float, cwd: str | Path | None = None) -> str:
    """Run one command; its stdout, or ``PodmanError`` with its stderr.

    ``cwd`` is where the command runs. It matters for podman: conmon, the
    monitor podman leaves behind each container, inherits it and writes an
    empty file named ``oom`` THERE when any process in the container is killed
    by the memory cgroup -- a marker that once landed in a run's source
    checkout and failed every later trial's clean-tree check. The pool
    therefore never runs podman from the caller's directory.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=None if cwd is None else str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise PodmanError("%s did not finish within %.0f s" % (_words(argv), timeout))
    if process.returncode != 0:
        raise PodmanError(
            "%s failed (exit %d): %s"
            % (_words(argv), process.returncode, stderr.decode(errors="replace").strip() or "(no stderr)")
        )
    return stdout.decode(errors="replace")


class BtrfsStore:
    """The directory on btrfs a pool keeps its subvolumes in.

    Three directories, one per kind: ``bases`` (read-only, one per image id),
    ``sandboxes`` (writable, one per container: its rootfs) and ``snapshots``
    (read-only). Every operation is one ``btrfs`` command run as the owner --
    no root: the store is mounted with ``user_subvol_rm_allowed`` so the owner
    can delete what it created, which :func:`mount_problem` checks at
    construction together with the filesystem type.
    """

    def __init__(self, path: str | Path, timeout: float = 120):
        self.path = Path(path).resolve()
        self.timeout = float(timeout)
        if not self.path.is_dir():
            raise ValueError(
                "store does not exist: %s (create one with "
                "sudo ops/btrfs_store.sh <mount point> <image file> <size>)" % self.path
            )
        self._probe()
        self.bases = self.path / "bases"
        self.sandboxes = self.path / "sandboxes"
        self.snapshots = self.path / "snapshots"
        for directory in (self.bases, self.sandboxes, self.snapshots):
            directory.mkdir(exist_ok=True)

    def _probe(self) -> None:
        """Refuse a store that cannot work before anything is created in it.

        Synchronous on purpose: this is configuration validation at
        construction, like the runtime binary check, so ``build_pool`` reports
        it as a configuration error rather than the first spawn -- or, for a
        missing mount option, the first destroy -- failing.
        """
        if shutil.which("btrfs") is None:
            raise ValueError(
                "btrfs-progs is not installed (no `btrfs` on PATH); "
                "the podman backend keeps its sandboxes in btrfs subvolumes"
            )
        problem = mount_problem(self.path)
        if problem:
            raise ValueError(problem)

    async def _btrfs(self, *args: str) -> str:
        return await _command(["btrfs", *args], timeout=self.timeout)

    async def create(self, path: Path) -> None:
        await self._btrfs("subvolume", "create", str(path))

    async def snapshot(self, source: Path, destination: Path, *, read_only: bool = False) -> None:
        flags = ["-r"] if read_only else []
        await self._btrfs("subvolume", "snapshot", *flags, str(source), str(destination))

    async def set_read_only(self, path: Path, read_only: bool) -> None:
        await self._btrfs("property", "set", str(path), "ro", "true" if read_only else "false")

    async def delete(self, path: Path) -> None:
        """Delete a subvolume, read-only or not; one already gone is not an error."""
        if not path.is_dir():
            return
        # The unprivileged owner may delete a subvolume only while it is
        # writable; on one that already is, this is a no-op.
        await self.set_read_only(path, False)
        await self._btrfs("subvolume", "delete", str(path))

    async def discard(self, path: Path) -> None:
        """Delete a subvolume of this store and the image config saved beside it."""
        await self.delete(path)
        _config_path(path).unlink(missing_ok=True)

    async def nested_subvolumes(self, rootfs: Path, *, podman_bin: str | None = None) -> list[Path]:
        """Subvolumes made inside ``rootfs``, deepest first.

        A subvolume's root directory is inode 256 (files start at 257), so one
        ``find`` names every nested one, at any depth. It runs inside the user
        namespace: a directory the container's user made 0700 is readable only
        by the uid it mapped to, i.e. only from in there. About a tenth of a
        second on a 100k-inode rootfs.
        """
        podman_bin = podman_bin or default_podman_bin()
        raw = await _command(
            [podman_bin, "unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"],
            timeout=self.timeout,
        )
        found = [Path(line) for line in raw.splitlines() if line.strip()]
        return sorted(found, key=lambda path: len(path.parts), reverse=True)

    async def discard_rootfs(self, rootfs: Path, *, podman_bin: str | None = None) -> None:
        """Delete a container's rootfs subvolume and its config; anything made inside it first.

        A subvolume holding another cannot be deleted ("Directory not empty"),
        so the nested ones go first, deepest first and from inside the user
        namespace, where their owners map to; each is made writable before, as
        :meth:`delete` does.
        """
        podman_bin = podman_bin or default_podman_bin()
        if rootfs.is_dir():
            for nested in await self.nested_subvolumes(rootfs, podman_bin=podman_bin):
                await _command(
                    [podman_bin, "unshare", "btrfs", "property", "set", str(nested), "ro", "false"],
                    timeout=self.timeout,
                )
                await _command(
                    [podman_bin, "unshare", "btrfs", "subvolume", "delete", str(nested)], timeout=self.timeout
                )
        await self.discard(rootfs)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete one snapshot of this store, by id; a snapshot that is not there is an error.

        Sandboxes started from it are unaffected: each is its own subvolume.
        A pool is not needed -- ``harness sweep`` deletes a finished run's
        snapshots from here, with no runtime binary or container in sight.
        """
        if not snapshot_id or "/" in snapshot_id or snapshot_id in (".", ".."):
            raise ValueError("not a snapshot id: %r" % snapshot_id)
        path = self.snapshots / snapshot_id
        if not path.is_dir():
            raise PodmanError("no snapshot %r in %s" % (snapshot_id, self.snapshots))
        await self.discard(path)


class PodmanPool:
    """Sandboxes as rootless podman containers on btrfs subvolumes.

    The same surface as :class:`~sprout_sandbox.pool.MicroVMPool`, so the
    session drives either without knowing which. ``spawn`` and
    ``spawn_from_image`` are one operation here: an OCI reference and a
    snapshot id both name a read-only subvolume to snapshot a rootfs from
    (the image's is extracted first when the store lacks it).
    """

    #: How long a runtime gets to answer ``tools/list`` after its container
    #: starts. A static binary on a warm host answers in well under a second;
    #: the margin is for a loaded machine.
    READY_TIMEOUT_S = 30.0
    READY_POLL_S = 0.25
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

    # --- podman ---

    async def _podman(self, *args: str, timeout: float | None = None, cwd: Path | None = None) -> str:
        """Run podman from the store (or ``cwd``), never from the caller's directory (see :func:`_command`)."""
        return await _command(
            [self.podman_bin, *args], timeout=timeout or self.timeout, cwd=cwd or self.store.path
        )

    async def _remove(self, container: str) -> None:
        """``podman rm -f``; a container that is already gone is not an error."""
        try:
            await self._podman("rm", "-f", container)
        except PodmanError as error:
            text = str(error).lower()
            if "no such container" in text or "no container with" in text:
                return
            raise

    async def _create_network(self, sandbox_id: str) -> None:
        """The sandbox's own bridge network, named after it (module docstring)."""
        await self._podman(
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
            await self._podman("network", "rm", sandbox_id)
        except PodmanError as error:
            if "network not found" not in str(error):
                raise

    # --- Host files ---

    async def upload_file(
        self, sandbox: Sandbox, source, destination: str, timeout: float | None = None
    ) -> None:
        """``podman cp`` a host file into the container, bypassing the runtime."""
        await self._podman(
            "cp",
            str(source),
            "%s:%s" % (sandbox.require_id(), destination),
            timeout=timeout or self.BULK_TIMEOUT_S,
        )

    async def download_file(
        self, sandbox: Sandbox, source: str, destination, timeout: float | None = None
    ) -> None:
        """``podman cp`` a container file to the host, outside the runtime."""
        await self._podman(
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
        return await self._base_for(image)

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
        socket_dir = self.socket_dir / sandbox_id
        await self.store.snapshot(source, rootfs)
        try:
            shutil.copyfile(_config_path(source), _config_path(rootfs))
            config = _read_config(rootfs)
            # The runtime binds its socket here as the container's root, which
            # is this user: the directory's owner, with no hand-over needed.
            socket_dir.mkdir()
            await self._create_network(sandbox_id)
            argv = self._run_argv(sandbox_id, rootfs, socket_dir, config, env, resources)
            online = self.allow_internet is not False
            # From the sandbox's own directory: conmon inherits it, so its
            # `oom` marker (a process in the container was OOM-killed) lands
            # beside the socket, where _diagnosis reads it and destroy removes it.
            if online:
                await self._podman("run", "-d", *argv, cwd=socket_dir)
            else:
                # Disconnected before it starts, so the first command already
                # runs offline; `run` would give it the network for an instant.
                await self._podman("create", *argv, cwd=socket_dir)
                await self._podman("network", "disconnect", sandbox_id, sandbox_id)
                await self._podman("start", sandbox_id, cwd=socket_dir)
            sandbox = self._handle(sandbox_id, agent_id)
            await self._wait_for_runtime(sandbox)
        except BaseException:
            # Nothing can reach this container; do not leak it, its network,
            # its rootfs or its socket directory while the caller sees only an
            # exception.
            await self._discard(sandbox_id)
            raise
        self._sandboxes[sandbox_id] = sandbox
        self._online[sandbox_id] = online
        return sandbox

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
        return await self._base_for(image)

    async def _base_for(self, image: str) -> Path:
        """The image's extracted rootfs, extracted now when the store lacks it.

        Keyed by the image id (the digest of its config), so two references
        to one image share a base and a re-pushed tag gets its own. The
        extraction lands in a ``.partial-`` subvolume and is published by
        renaming, so a base that exists is complete and two processes
        extracting the same image at once both end with the one base. A base
        extracted by an older format (``EXTRACTION_FORMAT``) is deleted and
        extracted again; the sandboxes and snapshots taken from it are their
        own subvolumes and keep what they have.
        """
        image = docker_reference(image)
        info = await image_info(
            image, podman_bin=self.podman_bin, timeout=self.BULK_TIMEOUT_S, cwd=self.store.path
        )
        digest = _image_digest(info.get("Id") or "")
        base = self.store.bases / digest
        if base.is_dir() and _extraction_format(base) == EXTRACTION_FORMAT:
            return base
        if base.is_dir() or _config_path(base).exists():
            await self.store.discard(base)
        partial = self.store.bases / ("%s.partial-%s" % (digest, secrets.token_hex(4)))
        await self.store.create(partial)
        try:
            container = _last_line(await self._podman("create", "--entrypoint", "/bin/true", image))
            try:
                await self._extract(container, partial)
            finally:
                await self._remove(container)
            # The export has no entry for "/" itself, so the new subvolume's mode
            # would be whatever this process's umask left (0700 under 077, and
            # then an image's non-root user cannot reach a single file).
            os.chmod(partial, 0o755)
            oci = info.get("Config") or {}
            _write_config(
                partial,
                {
                    "image": image,
                    "digest": digest,
                    "extract": EXTRACTION_FORMAT,
                    "repo_digests": list(info.get("RepoDigests") or []),
                    "config": {key: oci.get(key) for key in CONFIG_KEYS},
                },
            )
            os.replace(_config_path(partial), _config_path(base))
            try:
                os.rename(partial, base)
            except OSError:
                if not base.is_dir():
                    raise
                # Another process published the same base first; ours is a
                # duplicate.
                await self.store.delete(partial)
            await self.store.set_read_only(base, True)
        except BaseException:
            await self.store.discard(partial)
            raise
        return base

    async def _extract(self, container: str, destination: Path) -> None:
        """``podman export <ctr> | podman unshare tar -x --xattrs ... -C <destination>``.

        The tarball is a whole rootfs and never lands in this process: export
        writes one end of an OS pipe, tar reads the other. tar runs inside the
        user namespace so that the image's root becomes this user and its
        other uids become subordinate ids -- the mapping the container will
        run under. ``--numeric-owner`` keeps tar from consulting the host's
        passwd for the archive's user names. GNU tar reads the archive's
        extended attributes and ACLs but applies them only when asked:
        ``--xattrs --xattrs-include=* --acls`` restores ``security.capability``
        (a setcap'd binary keeps its capabilities), ``user.*`` and POSIX ACLs,
        which the namespace's root may set.
        """
        read_end, write_end = os.pipe()
        export = extract = None
        try:
            export = await asyncio.create_subprocess_exec(
                self.podman_bin,
                "export",
                container,
                cwd=str(self.store.path),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=write_end,
                stderr=asyncio.subprocess.PIPE,
            )
            os.close(write_end)
            write_end = -1
            extract = await asyncio.create_subprocess_exec(
                self.podman_bin,
                "unshare",
                "tar",
                "-x",
                "--numeric-owner",
                "--xattrs",
                "--xattrs-include=*",
                "--acls",
                "-C",
                str(destination),
                cwd=str(self.store.path),
                stdin=read_end,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            os.close(read_end)
            read_end = -1
            (_, export_err), (_, extract_err) = await asyncio.wait_for(
                asyncio.gather(export.communicate(), extract.communicate()), self.BULK_TIMEOUT_S
            )
        except BaseException:
            for process in (export, extract):
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
            raise
        finally:
            for descriptor in (read_end, write_end):
                if descriptor >= 0:
                    os.close(descriptor)
        # Both, when both failed: a tar that dies takes the export with it
        # (broken pipe), and the cause is in whichever stderr is not that.
        failures = [
            "podman %s failed (exit %d): %s"
            % (command, process.returncode, err.decode(errors="replace").strip() or "(no stderr)")
            for command, process, err in (
                ("export", export, export_err),
                ("unshare tar", extract, extract_err),
            )
            if process.returncode != 0
        ]
        if failures:
            raise PodmanError("; ".join(failures))

    async def _wait_for_runtime(self, sandbox: Sandbox) -> None:
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
        last_error: BaseException | str = "socket %s not created yet" % self._socket_path(sandbox_id)
        while True:
            try:
                await sandbox.backend.list_tools()
                return
            except Exception as error:  # noqa: BLE001 - a probe, retried
                last_error = error
            state = await self._container_state(sandbox_id)
            if state.startswith(("exited", "stopped")):
                raise PodmanError(
                    "runtime in %s exited before answering; %s"
                    % (sandbox_id, await self._diagnosis(sandbox_id))
                )
            if asyncio.get_running_loop().time() >= deadline:
                break
            await asyncio.sleep(self.READY_POLL_S)
        raise PodmanError(
            "runtime in %s did not answer within %.0f s (%s); %s"
            % (sandbox_id, self.READY_TIMEOUT_S, last_error, await self._diagnosis(sandbox_id))
        )

    async def _container_state(self, container: str) -> str:
        """``"<status> (exit <code>)"`` from one inspect, or what went wrong."""
        try:
            raw = await self._podman(
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

    async def _diagnosis(self, container: str) -> str:
        """The container's state, whether it was OOM-killed, and its last log lines, for an error message."""
        logs = ""
        try:
            logs = await self._podman("logs", "--tail", "20", container)
        except PodmanError:
            pass
        oom = (
            "; a process in it was OOM-killed (conmon's %s marker)" % OOM_MARKER
            if self.oom_killed(container)
            else ""
        )
        return "container %s%s; logs: %s" % (
            await self._container_state(container),
            oom,
            logs.strip() or "(none)",
        )

    def oom_killed(self, sandbox_id: str) -> bool:
        """Whether the memory cgroup killed a process in the container (conmon's marker, see :func:`_command`)."""
        return (self.socket_dir / sandbox_id / OOM_MARKER).exists()

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
                await self._discard(sandbox_id)
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

    async def _discard(self, sandbox_id: str) -> None:
        """Remove a sandbox's container, network, rootfs subvolume and socket directory, in that order.

        The container first: its rootfs is mounted while it exists, a mounted
        subvolume cannot be deleted, and a network with a container on it is
        not removed. A network or rootfs that will not go does not keep the
        other from going; the error names whatever is left.
        """
        await self._remove(sandbox_id)
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
        try:
            await self._podman("network", "connect" if allow else "disconnect", sandbox_id, sandbox_id)
        except PodmanError as error:
            settled = "already connected" if allow else "is not connected"
            if settled not in str(error):
                raise
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
        directory (module docstring), and a refused capture is recorded as
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
        shutil.copyfile(_config_path(rootfs), _config_path(target))
        self._snapshots_taken[sandbox_id] = count
        return Snapshot(id=snapshot_id, disk_only=True)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """:meth:`BtrfsStore.delete_snapshot` on this pool's store."""
        await self.store.delete_snapshot(snapshot_id)

    # --- Handles ---

    def _socket_path(self, sandbox_id: str) -> Path:
        return self.socket_dir / sandbox_id / SOCKET_FILE

    def _handle(self, sandbox_id: str, agent_id: str) -> Sandbox:
        sb = Sandbox(
            backend=HttpBackend("http://sprout-runtime", uds=str(self._socket_path(sandbox_id))),
            agent_id=agent_id,
        )
        sb._container_id = sandbox_id
        return sb


def docker_reference(image: str) -> str:
    """``image`` as Docker reads it. A name whose first part is no registry host
    (no ``.`` or ``:``, not ``localhost``) is on docker.io, and a one-part name in
    its ``library/`` namespace. Harbor tasks are written for Docker
    (TerminalBench's ``alexgshaw/<task>:<date>``), and podman refuses such a
    short name outright when no registries.conf names a search registry."""
    first, slash, _ = image.split("@", 1)[0].partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return image
    return ("docker.io/" if slash else "docker.io/library/") + image


async def image_info(
    image: str, *, podman_bin: str | None = None, timeout: float = 1800.0, cwd: str | Path | None = None
) -> dict:
    """``podman image inspect`` of ``image`` as one record, pulled first when it is not local.

    One command answers both questions asked of an image: what the pool keys a
    base on (``Id``, ``Config``) and what a run is pinned to (``RepoDigests``,
    see :func:`digest_reference`), so the two cannot disagree. ``timeout``
    bounds the pull; the inspect itself is quick. A Docker short name is
    qualified first (:func:`docker_reference`).
    """
    image = docker_reference(image)
    podman_bin = podman_bin or default_podman_bin()
    try:
        return await _inspect_image(image, podman_bin, cwd)
    except PodmanError as error:
        if "image not known" not in str(error):
            raise
    await _command([podman_bin, "pull", image], timeout=timeout, cwd=cwd)
    return await _inspect_image(image, podman_bin, cwd)


async def _inspect_image(image: str, podman_bin: str, cwd: str | Path | None = None) -> dict:
    raw = await _command([podman_bin, "image", "inspect", "--format", "json", image], timeout=120, cwd=cwd)
    parsed = json.loads(raw)
    return parsed[0] if isinstance(parsed, list) else parsed


def digest_reference(image: str, info: dict) -> str:
    """``image`` pinned to a digest: ``<repository>@sha256:...`` from its inspect record.

    A tag is mutable, so a run records the digest it actually ran; podman lists
    the digests it pulled a local image by in ``RepoDigests``, one per
    repository. The one for the repository ``image`` names wins (a short name
    such as ``python`` matches its qualified ``docker.io/library/python``); an
    image with none, a locally built one, cannot be pinned and is refused.
    """
    repository = image.split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    for candidate in info.get("RepoDigests") or []:
        named = candidate.split("@", 1)[0]
        if named == repository or named.endswith("/" + repository):
            return candidate
    raise PodmanError(
        "image %s has no repository digest to pin to (RepoDigests: %s); "
        "runs record the digest they ran against" % (image, info.get("RepoDigests"))
    )


def mount_problem(path: str | Path) -> str | None:
    """Why ``path`` cannot be a store, or ``None``: btrfs, mounted with ``user_subvol_rm_allowed``.

    ``findmnt`` names the filesystem type and the options of the mount holding
    the path. The store asks this at construction and the live tests ask it
    to decide whether to run, so what the tests skip on is exactly what the
    pool refuses.
    """
    try:
        completed = subprocess.run(
            ["findmnt", "-no", "FSTYPE,OPTIONS", "--target", str(path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return "findmnt (util-linux) is not installed; it is how the store's mount is checked"
    fields = completed.stdout.split()
    if completed.returncode != 0 or len(fields) < 2:
        return "findmnt cannot tell what %s is mounted on: %s" % (
            path,
            completed.stderr.strip() or "(no output)",
        )
    fstype, options = fields[0], fields[1]
    if fstype != "btrfs" or REQUIRED_MOUNT_OPTION not in options.split(","):
        return (
            "store %s is on %s mounted with %s; the podman backend needs btrfs mounted with %s "
            "(sudo ops/btrfs_store.sh <mount point> <image file> <size>)"
            % (path, fstype, options, REQUIRED_MOUNT_OPTION)
        )
    return None


def _config_path(subvolume: Path) -> Path:
    """The image config saved beside a subvolume."""
    return subvolume.with_name(subvolume.name + ".json")


def _write_config(subvolume: Path, payload: dict) -> None:
    _config_path(subvolume).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_config(subvolume: Path) -> dict:
    """The OCI config subset saved beside a subvolume (``CONFIG_KEYS``)."""
    return json.loads(_config_path(subvolume).read_text()).get("config") or {}


def _extraction_format(base: Path) -> int:
    """The format a base was extracted with; 0 for a record that says nothing (or is missing)."""
    try:
        return int(json.loads(_config_path(base).read_text()).get("extract") or 0)
    except (OSError, ValueError):
        return 0


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


def _image_digest(text: str) -> str:
    """The bare hex id ``podman image inspect`` reports as ``Id``."""
    match = _HEX_ID.match(text.strip())
    if not match:
        raise PodmanError("expected an image id, got %r" % text)
    return match.group(1)


def _safe(name: str) -> str:
    """A caller's snapshot name as a path component: letters, digits, ``._-``."""
    return _UNSAFE.sub("-", name).strip("-.")[:48] or "snapshot"


def _words(argv: list[str]) -> str:
    """``podman image inspect`` / ``btrfs subvolume delete``: the tool and its command words."""
    words = [Path(argv[0]).name]
    for arg in argv[1:3]:
        if arg.startswith("-"):
            break
        words.append(arg)
    return " ".join(words)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""
