"""Images: a reference becomes an inspect record, a pinned digest and a read-only base subvolume,
extracted once with ``podman export | tar`` and published by renaming; the saved OCI config sidecar.

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

An extraction that dies half-way leaves ``bases/<image id>.partial-<token>``;
a complete base is published by renaming, so a base that exists is whole. Its
config also records the extraction format (``extract``); a base extracted by an
older format is extracted again (format 1 ran tar without ``--xattrs`` and lost
every extended attribute, file capabilities included).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from pathlib import Path

from .btrfs_store import BtrfsStore, config_path
from .podman_cli import PodmanError, default_podman_bin, remove_container, run_command

#: What a container started from a bare rootfs needs from the image's OCI config
#: (podman reads none of it for ``--rootfs``); saved beside every subvolume.
CONFIG_KEYS = ("Env", "WorkingDir", "User", "Entrypoint", "Cmd")
#: The format of an extracted base, recorded as ``extract`` in its config; a
#: base whose record says less is extracted again. Format 1 ran tar without
#: ``--xattrs`` and so lost every extended attribute -- file capabilities
#: (``ping``, ``node``, gosu-style helpers) among them.
EXTRACTION_FORMAT = 2

_HEX_ID = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


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
    await run_command([podman_bin, "pull", image], timeout=timeout, cwd=cwd)
    return await _inspect_image(image, podman_bin, cwd)


async def _inspect_image(image: str, podman_bin: str, cwd: str | Path | None = None) -> dict:
    raw = await run_command([podman_bin, "image", "inspect", "--format", "json", image], timeout=120, cwd=cwd)
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


async def ensure_base(
    store: BtrfsStore, image: str, *, podman_bin: str, timeout: float, bulk_timeout: float
) -> Path:
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
    info = await image_info(image, podman_bin=podman_bin, timeout=bulk_timeout, cwd=store.path)
    digest = _image_digest(info.get("Id") or "")
    base = store.bases / digest
    if base.is_dir() and _extraction_format(base) == EXTRACTION_FORMAT:
        return base
    if base.is_dir() or config_path(base).exists():
        await store.discard(base)
    partial = store.bases / ("%s.partial-%s" % (digest, secrets.token_hex(4)))
    await store.create(partial)
    try:
        container = _last_line(
            await run_command(
                [podman_bin, "create", "--entrypoint", "/bin/true", image], timeout=timeout, cwd=store.path
            )
        )
        try:
            await _extract(podman_bin, container, partial, cwd=store.path, timeout=bulk_timeout)
        finally:
            await remove_container(podman_bin, container, timeout=timeout, cwd=store.path)
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
        os.replace(config_path(partial), config_path(base))
        try:
            os.rename(partial, base)
        except OSError:
            if not base.is_dir():
                raise
            # Another process published the same base first; ours is a
            # duplicate.
            await store.delete(partial)
        await store.set_read_only(base, True)
    except BaseException:
        await store.discard(partial)
        raise
    return base


async def _extract(podman_bin: str, container: str, destination: Path, *, cwd: Path, timeout: float) -> None:
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
            podman_bin,
            "export",
            container,
            cwd=str(cwd),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=write_end,
            stderr=asyncio.subprocess.PIPE,
        )
        os.close(write_end)
        write_end = -1
        extract = await asyncio.create_subprocess_exec(
            podman_bin,
            "unshare",
            "tar",
            "-x",
            "--numeric-owner",
            "--xattrs",
            "--xattrs-include=*",
            "--acls",
            "-C",
            str(destination),
            cwd=str(cwd),
            stdin=read_end,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        os.close(read_end)
        read_end = -1
        (_, export_err), (_, extract_err) = await asyncio.wait_for(
            asyncio.gather(export.communicate(), extract.communicate()), timeout
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


def _write_config(subvolume: Path, payload: dict) -> None:
    config_path(subvolume).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_config(subvolume: Path) -> dict:
    """The OCI config subset saved beside a subvolume (``CONFIG_KEYS``)."""
    return json.loads(config_path(subvolume).read_text()).get("config") or {}


def _extraction_format(base: Path) -> int:
    """The format a base was extracted with; 0 for a record that says nothing (or is missing)."""
    try:
        return int(json.loads(config_path(base).read_text()).get("extract") or 0)
    except (OSError, ValueError):
        return 0


def _image_digest(text: str) -> str:
    """The bare hex id ``podman image inspect`` reports as ``Id``."""
    match = _HEX_ID.match(text.strip())
    if not match:
        raise PodmanError("expected an image id, got %r" % text)
    return match.group(1)


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else ""
