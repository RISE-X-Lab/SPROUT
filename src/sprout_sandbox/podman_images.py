"""Images: an image reference becomes an inspect record, a pinned digest and a read-only base subvolume.

:func:`ensure_base` extracts the rootfs of an image into a base once. The rootfs of each sandbox is
a writable snapshot of a base or of a snapshot, so nothing stacks and nothing needs squashing. A base
that exists is whole: an extraction that dies half-way leaves only ``bases/<image id>.partial-<token>``.
The part of the OCI config that a container needs (``CONFIG_KEYS``) is saved beside each subvolume.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
from pathlib import Path
from typing import Any

from .btrfs_store import BtrfsStore, config_path
from .podman_cli import PodmanError, default_podman_bin, remove_container, run_command

#: The keys of the image's OCI config that a container on a bare rootfs needs, because podman reads
#: no config for ``--rootfs``. They are saved beside every subvolume.
CONFIG_KEYS = ("Env", "WorkingDir", "User", "Entrypoint", "Cmd")
#: The format of an extracted base, saved as ``extract`` in its config. A base of another format is
#: extracted again. Format 1 ran tar without ``--xattrs``, and lost file capabilities.
EXTRACTION_FORMAT = 2

_HEX_ID = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")


def docker_reference(image: str) -> str:
    """``image`` as Docker reads it.

    A name whose first part is not a registry host (no ``.`` or ``:``, not ``localhost``) is on docker.io,
    and a one-part name is in its ``library/`` namespace. Harbor tasks are written for Docker, and podman
    refuses a short name when no registries.conf names a search registry."""
    first, slash, _ = image.split("@", 1)[0].partition("/")
    if slash and ("." in first or ":" in first or first == "localhost"):
        return image
    return ("docker.io/" if slash else "docker.io/library/") + image


async def image_info(
    image: str, *, podman_bin: str | None = None, timeout: float = 1800.0, cwd: str | Path | None = None
) -> dict[str, Any]:
    """The ``podman image inspect`` record of ``image``, pulled first if it is not local.

    One record gives both the key of the base (``Id``, ``Config``) and the pinned digest
    (``RepoDigests``, :func:`digest_reference`), so the two cannot disagree. ``timeout`` bounds the
    pull. A Docker short name is qualified first (:func:`docker_reference`)."""
    image = docker_reference(image)
    podman_bin = podman_bin or default_podman_bin()
    try:
        return await _inspect_image(image, podman_bin, cwd)
    except PodmanError as error:
        if "image not known" not in str(error):
            raise
    await run_command([podman_bin, "pull", image], timeout=timeout, cwd=cwd)
    return await _inspect_image(image, podman_bin, cwd)


async def _inspect_image(image: str, podman_bin: str, cwd: str | Path | None = None) -> dict[str, Any]:
    raw = await run_command([podman_bin, "image", "inspect", "--format", "json", image], timeout=120, cwd=cwd)
    parsed = json.loads(raw)
    record: dict[str, Any] = parsed[0] if isinstance(parsed, list) else parsed
    return record


def digest_reference(image: str, info: dict[str, Any]) -> str:
    """``image`` pinned to a digest, as ``<repository>@sha256:...`` from its inspect record ``info``.

    The digest of the repository that ``image`` names as Docker reads it wins: ``python`` matches only
    ``docker.io/library/python``. A tag can change, so a run records the digest that it ran. Raises
    ``PodmanError`` when ``info`` has no digest of that repository, such as for a local build."""
    repository = docker_reference(image).split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    digests: list[str] = info.get("RepoDigests") or []
    for candidate in digests:
        if candidate.split("@", 1)[0] == repository:
            return candidate
    raise PodmanError(
        "image %s has no repository digest to pin to (RepoDigests: %s); "
        "runs record the digest they ran against" % (image, info.get("RepoDigests"))
    )


async def ensure_base(
    store: BtrfsStore, image: str, *, podman_bin: str, timeout: float, bulk_timeout: float
) -> Path:
    """The read-only base subvolume of ``image``, extracted now if the store lacks it.

    The image id keys the base, so two references to one image share a base and a re-pushed tag gets
    its own. Two processes that extract one image at once both end with the one base. A base of another
    ``EXTRACTION_FORMAT`` is extracted again, and the sandboxes and snapshots from it keep their files."""
    image = docker_reference(image)
    info = await image_info(image, podman_bin=podman_bin, timeout=bulk_timeout, cwd=store.path)
    digest = _image_digest(info.get("Id") or "")
    base = store.bases / digest
    if base.is_dir() and _extraction_format(base) == EXTRACTION_FORMAT:
        return base
    if base.is_dir() or config_path(base).exists():
        await store.discard(base)
    # The extraction lands in a .partial- subvolume, and a rename publishes it, so a base that
    # exists is complete.
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
        # The export has no entry for "/", so the subvolume's mode is what the umask left (0700 under
        # 077). Then a non-root user of the image cannot reach a single file.
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
            # Another process published the same base first, so this one is a duplicate.
            await store.delete(partial)
        await store.set_read_only(base, True)
    except BaseException:
        await store.discard(partial)
        raise
    return base


async def _extract(podman_bin: str, container: str, destination: Path, *, cwd: Path, timeout: float) -> None:
    """Extract the rootfs of ``container`` into ``destination``, with ``podman export | podman unshare tar``.

    Raises ``PodmanError`` with the stderr of each command that failed, and ``TimeoutError`` after
    ``timeout`` seconds."""
    # The tarball is a whole rootfs, so it never passes through this process: export writes one end
    # of an OS pipe, and tar reads the other.
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
        # tar runs in the user namespace, so the image's root becomes this user and its other uids
        # become subordinate ids: the mapping that the container runs under.
        extract = await asyncio.create_subprocess_exec(
            podman_bin,
            "unshare",
            "tar",
            "-x",
            # The archive's numeric ids, never looked up by name in the host's passwd.
            "--numeric-owner",
            # GNU tar applies extended attributes and ACLs only when asked. These restore
            # security.capability (a setcap binary keeps its capabilities), user.* and POSIX ACLs.
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
    # Report both failures. A tar that dies breaks the export's pipe, so the cause is in the
    # other stderr.
    failures = [
        "podman %s failed (exit %s): %s"
        % (command, process.returncode, err.decode(errors="replace").strip() or "(no stderr)")
        for command, process, err in (
            ("export", export, export_err),
            ("unshare tar", extract, extract_err),
        )
        if process.returncode != 0
    ]
    if failures:
        raise PodmanError("; ".join(failures))


def _write_config(subvolume: Path, payload: dict[str, Any]) -> None:
    config_path(subvolume).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_config(subvolume: Path) -> dict[str, Any]:
    """The OCI config subset saved beside a subvolume (``CONFIG_KEYS``)."""
    config: dict[str, Any] = json.loads(config_path(subvolume).read_text()).get("config") or {}
    return config


def _extraction_format(base: Path) -> int:
    """The format that a base was extracted with, or 0 if its record is missing or says nothing."""
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
