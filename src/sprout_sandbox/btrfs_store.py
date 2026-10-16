"""The btrfs store: its bases/sandboxes/snapshots layout, the ``<name>.json`` sidecar beside each
subvolume, the btrfs subvolume operations (nested subvolumes included) and the mount check.

**Layout of the store**, all of it owned by the user and every subvolume with
the image config it was extracted from saved beside it as ``<name>.json``::

    <store>/bases/<image id>            read-only, one per image (Id from podman image inspect)
    <store>/sandboxes/<sandbox id>      writable, one per container: its rootfs
    <store>/snapshots/<snapshot id>     read-only; the id is "<sandbox id>-<counter>[-<name>]"
    <store>/sockets/<sandbox id>/runtime.sock   (socket_dir; the container sees /run/sprout)

**A subvolume made inside a sandbox is not snapshotted.** ``btrfs subvolume
snapshot`` is not recursive: a subvolume the container created (an
unprivileged ioctl -- ``btrfs subvolume create``, podman-in-podman on btrfs,
snapper) would appear in the snapshot as an empty directory. The pool looks for
them first (``find -inum 256`` inside the user namespace, since a 0700
directory of the container's user is readable only there; about a tenth of a
second on a 100k-inode rootfs) and refuses the snapshot, so the checkpointer
records the step as not captured rather than a branch point with files
missing; ``destroy`` deletes them, deepest first, before the rootfs.

**Cleaning up.** The store's directories
are the catalog of subvolumes (``btrfs subvolume list`` needs root in
btrfs-progs 6.6; a subvolume's root directory has inode 256), and a read-only
one is deleted only after ``btrfs property set <path> ro false``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .podman_cli import PodmanError, default_podman_bin, run_command

#: The mount option that lets the store's unprivileged owner delete the
#: subvolumes it created; without it every spawn works and the first destroy
#: fails after the container is gone.
REQUIRED_MOUNT_OPTION = "user_subvol_rm_allowed"


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
        return await run_command(["btrfs", *args], timeout=self.timeout)

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
        config_path(path).unlink(missing_ok=True)

    async def nested_subvolumes(self, rootfs: Path, *, podman_bin: str | None = None) -> list[Path]:
        """Subvolumes made inside ``rootfs``, deepest first.

        A subvolume's root directory is inode 256 (files start at 257), so one
        ``find`` names every nested one, at any depth. It runs inside the user
        namespace: a directory the container's user made 0700 is readable only
        by the uid it mapped to, i.e. only from in there. About a tenth of a
        second on a 100k-inode rootfs.
        """
        podman_bin = podman_bin or default_podman_bin()
        raw = await run_command(
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
                await run_command(
                    [podman_bin, "unshare", "btrfs", "property", "set", str(nested), "ro", "false"],
                    timeout=self.timeout,
                )
                await run_command(
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


def config_path(subvolume: Path) -> Path:
    """The image config saved beside a subvolume."""
    return subvolume.with_name(subvolume.name + ".json")
