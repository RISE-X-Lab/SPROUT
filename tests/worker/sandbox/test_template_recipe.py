"""The recipe of a microVM template: its name, the image's OCI config read with ``regctl``, and ripgrep.

A stub ``regctl`` is a shell script in ``tmp_path``. The ripgrep download comes from a scripted HTTP
server on a local port.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path

import pytest

import sprout.worker.sandbox.template_recipe as recipe
from sprout.worker.sandbox.template_recipe import (
    RUNTIME_PATH,
    TemplateError,
    ensure_ripgrep,
    find_regctl,
    image_config_env,
    registry_config,
    template_name,
)
from sprout_sandbox import podman_containers

IMAGE = "swebench/sweb.eval.x86_64.django__django-11848:latest"


# --- the name of a template ---------------------------------------------------------------


@pytest.mark.parametrize(
    "image, fingerprint, port, resources, salt, name",
    [
        (IMAGE, "fp1", 3000, None, "", "sprout-swebench-d6a2d935d1d42d7bb114aadb"),
        (IMAGE, "fp1", 3000, {}, "", "sprout-swebench-d6a2d935d1d42d7bb114aadb"),
        (IMAGE, "fp1", 3001, None, "", "sprout-swebench-50715e7edc8e671836d6330c"),
        (IMAGE, "fp2", 3000, None, "", "sprout-swebench-538ddb485107d61a926395c8"),
        (IMAGE + "@v2", "fp1", 3000, None, "", "sprout-swebench-dec24a9e7febc53a3caaef44"),
        (IMAGE, "fp1", 3000, {"cpu": 2, "memory_mb": 4096}, "", "sprout-swebench-ce90d5fc2f041b5f61e86c85"),
        (
            IMAGE,
            "fp1",
            3000,
            {"cpu": 2, "memory_mb": 4096, "disk_size_mb": 65536},
            "",
            "sprout-swebench-66cb82f609b798a99844da4a",
        ),
        (IMAGE, "fp1", 3000, None, "env:0123456789ab", "sprout-swebench-0091fc5234bc92eada9cc9db"),
    ],
    ids=["base", "no-shape", "port", "runtime", "image", "shape", "shape-with-disk", "salt"],
)
def test_a_template_name_is_stable_and_changes_with_each_input(
    image, fingerprint, port, resources, salt, name
):
    """The server holds templates under these names. A changed derivation rebuilds every template,
    and the templates of the reported results change."""
    assert template_name(image, fingerprint, port, resources, salt=salt) == name


def test_the_template_and_the_podman_pool_place_the_runtime_at_one_path():
    """``sprout_sandbox`` may not import from ``sprout``, so the pool keeps its own copy of the path."""
    assert (RUNTIME_PATH, podman_containers.RUNTIME_PATH) == ("/usr/local/bin/sprout-runtime",) * 2


# --- the image's OCI config ---------------------------------------------------------------


def stub_regctl(directory: Path, script: str, name: str = "regctl") -> Path:
    """An executable ``name`` in ``directory`` that runs the shell ``script``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(0o755)
    return path


def printing(config: object) -> str:
    """A script that logs its arguments to ``$REGCTL_LOG`` and prints ``config`` as JSON."""
    return 'echo "$@" >> "$REGCTL_LOG"\necho %s\n' % json.dumps(json.dumps(config))


def test_the_env_of_an_image_is_its_key_value_entries_in_order(tmp_path, monkeypatch):
    regctl = stub_regctl(
        tmp_path, printing({"Env": ["PATH=/opt/venv/bin", "NO_VALUE", "OPTS=a=b"], "User": "app"})
    )
    monkeypatch.setenv("REGCTL_LOG", str(tmp_path / "regctl.log"))

    env = image_config_env(IMAGE, regctl)

    assert env == ["PATH=/opt/venv/bin", "OPTS=a=b"]
    assert (tmp_path / "regctl.log").read_text() == "image config %s --format {{json .Config}}\n" % IMAGE


@pytest.mark.parametrize(
    "script, message",
    [
        (
            'echo "unauthorized: authentication required" >&2\nexit 1\n',
            "regctl could not read .*: unauthorized",
        ),
        ("echo not-json\n", "^regctl returned no JSON for "),
        ("echo '[1, 2]'\n", "^regctl returned no config object for "),
    ],
    ids=["regctl-fails", "not-json", "not-an-object"],
)
def test_a_config_that_regctl_cannot_give_raises(tmp_path, script, message):
    with pytest.raises(TemplateError, match=message):
        image_config_env(IMAGE, stub_regctl(tmp_path, script))


def test_a_regctl_that_cannot_run_raises(tmp_path):
    with pytest.raises(TemplateError, match="^could not read .*'s config with regctl: "):
        image_config_env(IMAGE, tmp_path / "absent-regctl")


@pytest.fixture
def no_regctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A host with no regctl: none in ``$SPROUT_REGCTL``, on PATH or among AgentENV's copies.

    Returns the directory of AgentENV's copies, ``deps/regctl/<version>/regctl``."""
    monkeypatch.delenv("SPROUT_REGCTL", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(recipe, "REGCTL_DEPS_GLOB", str(tmp_path / "deps" / "regctl" / "*" / "regctl"))
    return tmp_path / "deps" / "regctl"


def test_the_regctl_named_by_sprout_regctl_wins(tmp_path, monkeypatch, no_regctl):
    named = stub_regctl(tmp_path / "named", "")
    stub_regctl(tmp_path / "bin", "")
    stub_regctl(no_regctl / "v0.11.5", "")
    monkeypatch.setenv("SPROUT_REGCTL", str(named))

    assert find_regctl() == named


def test_without_sprout_regctl_the_regctl_on_path_wins(tmp_path, monkeypatch, no_regctl):
    """A ``$SPROUT_REGCTL`` that names no file does not count."""
    on_path = stub_regctl(tmp_path / "bin", "")
    stub_regctl(no_regctl / "v0.11.5", "")
    monkeypatch.setenv("SPROUT_REGCTL", str(tmp_path / "absent-regctl"))

    assert find_regctl() == on_path


def test_without_another_regctl_the_copy_of_agentenv_is_used(no_regctl):
    """AgentENV's copy has the server's registry logins."""
    copy = stub_regctl(no_regctl / "v0.11.5", "")

    assert find_regctl() == copy


def test_a_host_without_regctl_has_none(no_regctl):
    assert find_regctl() is None


def test_the_registry_config_of_an_image_is_read_with_the_hosts_regctl(tmp_path, monkeypatch, no_regctl):
    config = {"User": "app", "Env": ["PATH=/usr/bin"], "WorkingDir": "/srv"}
    monkeypatch.setenv("SPROUT_REGCTL", str(stub_regctl(tmp_path, printing(config))))
    monkeypatch.setenv("REGCTL_LOG", str(tmp_path / "regctl.log"))

    assert registry_config(IMAGE) == config


@pytest.mark.parametrize("image", [None, ""], ids=["none", "empty"])
def test_a_registry_config_needs_an_image_reference(no_regctl, image):
    with pytest.raises(
        TemplateError, match="^no OCI image reference names the image whose config is to be read$"
    ):
        registry_config(image)


def test_a_registry_config_without_regctl_raises_with_where_regctl_was_looked_for(no_regctl):
    with pytest.raises(
        TemplateError, match=r"^no regctl was found \(SPROUT_REGCTL, PATH, or .*\); the config of "
    ):
        registry_config(IMAGE)


# --- ripgrep for the templates ----------------------------------------------------------


def ripgrep_release() -> bytes:
    """A tarball laid out as a ripgrep release: a directory named ``rg`` before the ``rg`` binary."""
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        folder = tarfile.TarInfo("ripgrep-14.1.1-x86_64-unknown-linux-musl/complete/rg")
        folder.type = tarfile.DIRTYPE
        tar.addfile(folder)
        binary = tarfile.TarInfo("ripgrep-14.1.1-x86_64-unknown-linux-musl/rg")
        binary.size = len(b"rg binary")
        tar.addfile(binary, io.BytesIO(b"rg binary"))
    return archive.getvalue()


def test_ripgrep_is_downloaded_once_into_the_cache_as_an_executable(agentenv_stub, monkeypatch, tmp_path):
    agentenv_stub.answer("GET", "/ripgrep.tar.gz", (200, ripgrep_release()))
    monkeypatch.setattr(recipe, "RIPGREP_URL", agentenv_stub.url + "/ripgrep.tar.gz")

    first = ensure_ripgrep(tmp_path / "cache")
    second = ensure_ripgrep(tmp_path / "cache")

    assert (first, second) == (tmp_path / "cache" / "rg-14.1.1",) * 2
    assert (first.read_bytes(), first.stat().st_mode & 0o777) == (b"rg binary", 0o755)
    assert agentenv_stub.paths() == ["GET /ripgrep.tar.gz"]


def test_a_ripgrep_that_cannot_be_fetched_is_none_and_a_note_says_so(
    agentenv_stub, monkeypatch, tmp_path, capsys
):
    """Templates then still build, but their sandboxes have no ``rg`` on PATH."""
    monkeypatch.setattr(recipe, "RIPGREP_URL", agentenv_stub.url + "/ripgrep.tar.gz")

    ripgrep = ensure_ripgrep(tmp_path / "cache")

    note = capsys.readouterr().out
    assert ripgrep is None
    assert re.fullmatch(
        r"note: could not fetch ripgrep to bake into templates \(.*404 Not Found.*\); "
        r"sandboxes built from these templates have no rg on PATH\n",
        note,
        flags=re.DOTALL,
    ), note
