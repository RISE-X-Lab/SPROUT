"""``BtrfsStore``: the directory on btrfs where a pool keeps its subvolumes, and the check of its mount.

The user owns all of the store. Beside each subvolume, ``<name>.json`` holds the image config that
the subvolume comes from::

    <store>/bases/<image id>            read-only, one per image (Id from podman image inspect)
    <store>/sandboxes/<sandbox id>      writable, one per container: its rootfs
    <store>/snapshots/<snapshot id>     read-only, id "<sandbox id>-<counter>[-<name>]"
    <store>/sockets/<sandbox id>/runtime.sock   (socket_dir; the container sees /run/sprout)

A subvolume made inside a sandbox is not in a snapshot of the sandbox: ``btrfs subvolume snapshot``
is not recursive, and shows the nested subvolume as an empty directory. :meth:`BtrfsStore.nested_subvolumes`
finds such subvolumes, so that the pool can refuse the snapshot, and :meth:`BtrfsStore.discard_rootfs`
deletes them first. The store's directories are the catalog of its subvolumes, because
``btrfs subvolume list`` needs root.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .podman_cli import PodmanError, default_podman_bin, run_command

#: The mount option that lets the store's unprivileged owner delete its own subvolumes. Without it,
#: every spawn works, and the first destroy fails after the container is gone.
REQUIRED_MOUNT_OPTION = "user_subvol_rm_allowed"


class BtrfsStore:
    """The directory on btrfs where a pool keeps its subvolumes: ``bases``, ``sandboxes``, ``snapshots``.

    Each operation is one ``btrfs`` command that the owner runs without root, and raises
    ``PodmanError`` if the command fails. Raises ``ValueError`` at construction for a path that
    :func:`store_problem` refuses."""

    def __init__(self, path: str | Path, timeout: float = 120) -> None:
        self.path = Path(path).resolve()
        self.timeout = float(timeout)
        # Check at construction, so that a bad store is a configuration error, and not a failure of
        # the first spawn or, for a missing mount option, of the first destroy.
        problem = store_problem(self.path)
        if problem:
            raise ValueError(problem)
        self.bases = self.path / "bases"
        self.sandboxes = self.path / "sandboxes"
        self.snapshots = self.path / "snapshots"
        for directory in (self.bases, self.sandboxes, self.snapshots):
            directory.mkdir(exist_ok=True)

    async def _btrfs(self, *args: str) -> str:
        return await run_command(["btrfs", *args], timeout=self.timeout)

    async def create(self, path: Path) -> None:
        """Create the empty subvolume ``path``."""
        await self._btrfs("subvolume", "create", str(path))

    async def snapshot(self, source: Path, destination: Path, *, read_only: bool = False) -> None:
        """Snapshot the subvolume ``source`` as ``destination``, read-only if ``read_only``."""
        flags = ["-r"] if read_only else []
        await self._btrfs("subvolume", "snapshot", *flags, str(source), str(destination))

    async def set_read_only(self, path: Path, read_only: bool) -> None:
        """Make the subvolume ``path`` read-only, or writable."""
        await self._btrfs("property", "set", str(path), "ro", "true" if read_only else "false")

    async def delete(self, path: Path) -> None:
        """Delete a subvolume, read-only or not. A subvolume that is already gone is not an error."""
        if not path.is_dir():
            return
        # The unprivileged owner can delete a subvolume only while it is writable. On a writable
        # subvolume, this changes nothing.
        await self.set_read_only(path, False)
        await self._btrfs("subvolume", "delete", str(path))

    async def discard(self, path: Path) -> None:
        """Delete a subvolume of this store and the image config saved beside it."""
        await self.delete(path)
        config_path(path).unlink(missing_ok=True)

    async def nested_subvolumes(self, rootfs: Path, *, podman_bin: str | None = None) -> list[Path]:
        """The subvolumes made inside ``rootfs``, at any depth, deepest first.

        The search runs in the user namespace, so it also sees inside a directory that the container's
        user made 0700. It takes about a tenth of a second on a rootfs of 100k inodes."""
        podman_bin = podman_bin or default_podman_bin()
        # The root directory of a subvolume is inode 256, and files start at 257.
        raw = await run_command(
            [podman_bin, "unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"],
            timeout=self.timeout,
        )
        found = [Path(line) for line in raw.splitlines() if line.strip()]
        return sorted(found, key=lambda path: len(path.parts), reverse=True)

    async def discard_rootfs(self, rootfs: Path, *, podman_bin: str | None = None) -> None:
        """Delete a container's rootfs subvolume and its config, and first every subvolume made inside it.

        A subvolume that holds another cannot be deleted ("Directory not empty"). So the nested ones go
        first, deepest first, from inside the user namespace, where their owners map to."""
        podman_bin = podman_bin or default_podman_bin()
        if rootfs.is_dir():
            for nested in await self.nested_subvolumes(rootfs, podman_bin=podman_bin):
                # Writable first, for the same reason as in `delete`.
                await run_command(
                    [podman_bin, "unshare", "btrfs", "property", "set", str(nested), "ro", "false"],
                    timeout=self.timeout,
                )
                await run_command(
                    [podman_bin, "unshare", "btrfs", "subvolume", "delete", str(nested)], timeout=self.timeout
                )
        await self.discard(rootfs)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        """Delete the snapshot ``snapshot_id`` of this store, and its config. The call needs no pool.

        Raises ``ValueError`` for an id that is not one path component, and ``PodmanError`` if the store
        holds no such snapshot. Sandboxes started from the snapshot keep their files."""
        if not snapshot_id or "/" in snapshot_id or snapshot_id in (".", ".."):
            raise ValueError("not a snapshot id: %r" % snapshot_id)
        path = self.snapshots / snapshot_id
        if not path.is_dir():
            raise PodmanError("no snapshot %r in %s" % (snapshot_id, self.snapshots))
        await self.discard(path)


def store_problem(path: str | Path) -> str | None:
    """Why ``path`` cannot be a store, or ``None``. :class:`BtrfsStore` refuses the same paths.

    The directory must exist, ``btrfs`` must be on PATH, and :func:`mount_problem` must find nothing.
    The check creates nothing in the directory."""
    if not Path(path).is_dir():
        return (
            "store does not exist: %s (create one with "
            "sudo ops/btrfs_store.sh <mount point> <image file> <size>)" % path
        )
    if shutil.which("btrfs") is None:
        return (
            "btrfs-progs is not installed (no `btrfs` on PATH); "
            "the podman backend keeps its sandboxes in btrfs subvolumes"
        )
    return mount_problem(path)


def mount_problem(path: str | Path) -> str | None:
    """Why the mount of ``path`` cannot hold a store, or ``None``.

    The mount that holds ``path`` must be btrfs, with ``user_subvol_rm_allowed``. ``findmnt`` must be
    installed to tell."""
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


def config_path(subvolume: Path) -> Path:
    """The image config saved beside a subvolume."""
    return subvolume.with_name(subvolume.name + ".json")
