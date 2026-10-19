"""Image references as Docker reads them, pinned digests, and the base subvolume of each image.

The medium tests extract bases with the fake ``podman`` and ``btrfs``: the fake exports an archive
of ``EXPORT_BYTES`` zero bytes, and its ``unshare tar`` writes the size it read to ``etc/extracted``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from sprout_sandbox.btrfs_store import config_path
from sprout_sandbox.podman_cli import PodmanError
from sprout_sandbox.podman_images import EXTRACTION_FORMAT, digest_reference, docker_reference
from tests.support.fakes.podman import EXPORT_BYTES, EXTRACTION_CONTAINER_ID

pytestmark = pytest.mark.anyio

IMAGE = "docker.io/library/python:3.11-slim"
#: The OCI config subset that the fake podman reports for every image.
FAKE_CONFIG = {
    "Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
    "WorkingDir": "/testbed",
    "User": "",
    "Entrypoint": None,
    "Cmd": ["python3"],
}


def fake_image_id(reference: str) -> str:
    """The image ``Id`` that the fake podman reports for ``reference``."""
    return hashlib.sha256(reference.encode()).hexdigest()


def extractions(calls: list[list[str]]) -> list[str]:
    """The image of each container that podman created to export a rootfs from."""
    return [call[-1] for call in calls if call[:3] == ["create", "--entrypoint", "/bin/true"]]


def tar_call(calls: list[list[str]]) -> list[str]:
    """The one ``podman unshare tar`` call among ``calls``."""
    [tar] = [call for call in calls if call[:2] == ["unshare", "tar"]]
    return tar


def created_subvolume(calls: list[list[str]]) -> str:
    """The path of the one ``btrfs subvolume create`` call among ``calls``."""
    [created] = [call[2] for call in calls if call[:2] == ["subvolume", "create"]]
    return created


def names_in(directory: Path) -> list[str]:
    """The names of the entries of ``directory``, sorted."""
    return sorted(path.name for path in directory.iterdir())


def container_commands(calls: list[list[str]]) -> list[list[str]]:
    """The calls among ``calls`` that start a container or make its network."""
    return [call for call in calls if call[0] in ("run", "start", "network")]


def write_base(store: Path, reference: str, record: dict | None) -> Path:
    """A base of ``reference`` with a file ``etc/old``, and ``record`` beside it unless that is None."""
    base = store / "bases" / fake_image_id(reference)
    (base / "etc").mkdir(parents=True)
    (base / "etc" / "old").write_text("from an older extraction")
    if record is not None:
        config_path(base).write_text(json.dumps(record))
    return base


@pytest.mark.parametrize(
    "image, expected",
    [
        ("python:3.11-slim", "docker.io/library/python:3.11-slim"),
        ("alexgshaw/bn-fit-modify:20251031", "docker.io/alexgshaw/bn-fit-modify:20251031"),
        ("docker.io/library/python", "docker.io/library/python"),
        ("localhost/image", "localhost/image"),
        ("registry:5000/image", "registry:5000/image"),
    ],
)
def test_a_reference_is_qualified_as_docker_reads_it(image, expected):
    """Harbor tasks name images the Docker way. Podman refuses a short name without a search registry."""
    assert docker_reference(image) == expected


@pytest.mark.parametrize(
    "image, repo_digests, expected",
    [
        pytest.param(
            "docker.io/library/python:3.11",
            ["docker.io/library/python@sha256:" + "a" * 64],
            "docker.io/library/python@sha256:" + "a" * 64,
            id="qualified-and-tagged",
        ),
        pytest.param(
            "python:3.11",
            ["docker.io/library/python@sha256:" + "a" * 64],
            "docker.io/library/python@sha256:" + "a" * 64,
            id="short-name",
        ),
        pytest.param(
            "registry:5000/image",
            ["mirror.example/image@sha256:" + "b" * 64, "registry:5000/image@sha256:" + "c" * 64],
            "registry:5000/image@sha256:" + "c" * 64,
            id="registry-port-is-not-a-tag",
        ),
        pytest.param(
            "python:3.11",
            ["docker.io/someone/python@sha256:" + "b" * 64, "docker.io/library/python@sha256:" + "c" * 64],
            "docker.io/library/python@sha256:" + "c" * 64,
            id="short-name-with-a-namesake",
        ),
    ],
)
def test_an_image_is_pinned_to_the_digest_of_the_repository_that_it_names(image, repo_digests, expected):
    assert digest_reference(image, {"RepoDigests": repo_digests}) == expected


def test_an_image_without_a_repository_digest_cannot_be_pinned():
    """A local build has no digest. A run records the digest that it ran."""
    with pytest.raises(PodmanError) as raised:
        digest_reference("localhost/build:latest", {"RepoDigests": []})

    assert str(raised.value) == (
        "image localhost/build:latest has no repository digest to pin to (RepoDigests: []); "
        "runs record the digest they ran against"
    )


async def test_each_image_is_extracted_once_and_later_uses_reuse_its_base(pool, fake_podman):
    await pool.prepare(IMAGE)
    await pool.spawn(IMAGE)
    await pool.prepare("docker.io/library/alpine:3.19")

    assert extractions(fake_podman.calls()) == [IMAGE, "docker.io/library/alpine:3.19"]


async def test_a_base_holds_the_whole_export_and_the_image_config_under_the_image_id(pool, podman_store):
    base = await pool.prepare(IMAGE)

    assert base == podman_store / "bases" / fake_image_id(IMAGE)
    assert (base / "etc" / "extracted").read_text() == str(EXPORT_BYTES)
    assert json.loads(config_path(base).read_text()) == {
        "image": IMAGE,
        "digest": fake_image_id(IMAGE),
        "extract": EXTRACTION_FORMAT,
        "repo_digests": ["docker.io/library/python@sha256:" + fake_image_id(IMAGE)],
        "config": FAKE_CONFIG,
    }
    assert names_in(podman_store / "bases") == [base.name, base.name + ".json"]


async def test_the_extraction_keeps_numeric_owners_extended_attributes_and_acls(
    pool, fake_podman, fake_btrfs
):
    """GNU tar restores xattrs, file capabilities among them, and ACLs only when asked."""
    base = await pool.prepare(IMAGE)

    partial = created_subvolume(fake_btrfs.calls())

    assert tar_call(fake_podman.calls()) == [
        "unshare", "tar", "-x", "--numeric-owner", "--xattrs", "--xattrs-include=*", "--acls", "-C", partial
    ]  # fmt: skip
    assert re.fullmatch(re.escape(str(base)) + r"\.partial-[0-9a-f]{8}", partial), (
        "a base that exists is whole"
    )


async def test_an_extraction_publishes_a_read_only_base_and_removes_its_export_container(
    pool, fake_podman, fake_btrfs
):
    base = await pool.prepare(IMAGE)

    assert ["property", "set", str(base), "ro", "true"] in fake_btrfs.calls()
    assert ["rm", "-f", EXTRACTION_CONTAINER_ID] in fake_podman.calls()


async def test_a_base_is_readable_by_every_user_of_the_image_whatever_the_umask(pool):
    """The export has no entry for "/", so the mode of the root comes from the pool. A root of 0700
    would keep a non-root user of the image from each file of its rootfs."""
    os.umask(0o077)

    base = await pool.prepare(IMAGE)

    assert oct(base.stat().st_mode & 0o777) == oct(0o755)


@pytest.mark.parametrize(
    "record",
    [
        pytest.param({"image": IMAGE, "extract": 1, "config": FAKE_CONFIG}, id="format-1"),
        pytest.param({"image": IMAGE, "config": FAKE_CONFIG}, id="no-format"),
        pytest.param(None, id="no-record"),
    ],
)
async def test_a_base_of_another_extraction_format_is_extracted_again(pool, podman_store, record):
    """A base is keyed by the image id alone. A base of format 1, which lost each extended attribute,
    would otherwise serve each sandbox of the image."""
    stale = write_base(podman_store, IMAGE, record)

    base = await pool.prepare(IMAGE)

    assert (base, names_in(base / "etc")) == (stale, ["extracted"])
    assert json.loads(config_path(base).read_text())["extract"] == EXTRACTION_FORMAT


async def test_an_image_that_is_not_local_is_pulled_first(pool, fake_podman):
    fake_podman.unpulled(IMAGE)

    await pool.prepare(IMAGE)

    inspect = ["image", "inspect", "--format", "json", IMAGE]
    assert fake_podman.calls()[:3] == [inspect, ["pull", IMAGE], inspect]


async def test_an_inspect_that_fails_for_another_reason_is_an_error_and_pulls_nothing(pool, fake_podman):
    fake_podman.fail("image")

    with pytest.raises(
        PodmanError, match=r"^podman image inspect failed \(exit 125\): Error: simulated image failure$"
    ):
        await pool.prepare(IMAGE)

    assert fake_podman.calls() == [["image", "inspect", "--format", "json", IMAGE]]


async def test_a_docker_short_name_reaches_podman_qualified(pool, fake_podman):
    """TerminalBench names its images alexgshaw/<task>:<date>, which Docker reads as docker.io."""
    qualified = "docker.io/alexgshaw/bn-fit-modify:20251031"
    fake_podman.unpulled(qualified)

    await pool.prepare("alexgshaw/bn-fit-modify:20251031")

    assert fake_podman.calls()[:4] == [
        ["image", "inspect", "--format", "json", qualified],
        ["pull", qualified],
        ["image", "inspect", "--format", "json", qualified],
        ["create", "--entrypoint", "/bin/true", qualified],
    ]


async def test_a_failed_extraction_leaves_no_partial_base_and_no_export_container(
    pool, fake_podman, podman_store
):
    fake_podman.fail("export")

    with pytest.raises(
        PodmanError, match=r"^podman export failed \(exit 125\): Error: simulated export failure$"
    ):
        await pool.prepare(IMAGE)

    assert names_in(podman_store / "bases") == []
    assert ["rm", "-f", EXTRACTION_CONTAINER_ID] in fake_podman.calls()
