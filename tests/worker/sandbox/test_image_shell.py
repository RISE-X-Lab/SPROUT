"""``image_shell``: the user, the ENV and the directory that the image's own commands run with.

The sessions here are microVM sessions, which read an image's config from its registry. A stub
``regctl`` (``$SPROUT_REGCTL``) prints the config in ``$REGCTL_CONFIG``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.sandbox.image_shell import ImageShell, image_shell
from sprout.worker.sandbox.session import SandboxSession
from sprout.worker.sandbox.template_recipe import TemplateError

IMAGE = "docker.io/library/haproxy:3.0"
STUB_REGCTL = '#!/bin/sh\nprintf "%s\\n" "$REGCTL_CONFIG"\n'


@pytest.fixture
def registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Set the OCI config that the stub ``regctl`` prints for every image."""
    regctl = tmp_path / "regctl"
    regctl.write_text(STUB_REGCTL)
    regctl.chmod(0o755)
    monkeypatch.setenv("SPROUT_REGCTL", str(regctl))

    def serve(config: dict[str, Any]) -> None:
        monkeypatch.setenv("REGCTL_CONFIG", json.dumps(config))

    return serve


def microvm(**options: bool) -> SandboxSession:
    return SandboxSession(quiet=True, backend={"backend": "microvm", "microvm": options})


@pytest.mark.parametrize(
    "backend",
    [{}, {"backend": "microvm", "microvm": {}}, {"backend": "podman", "podman": {"image_env": True}}],
    ids=["default", "microvm", "image-env-alone"],
)
def test_without_image_user_the_commands_run_as_root_and_no_config_is_read(registry, backend):
    """A podman session without a sandbox raises if it is asked for a config."""
    registry({"User": "app"})

    shell = image_shell(SandboxSession(quiet=True, backend=backend), backend=backend, image=IMAGE)

    assert shell is None


@pytest.mark.parametrize(
    "config, image_env, expected",
    [
        (
            {"User": "app", "Env": ["PATH=/usr/bin", "FOO=a=b c"], "WorkingDir": "/srv"},
            False,
            ImageShell(user="app", env={}, workdir="/srv"),
        ),
        (
            {"User": "app", "Env": ["PATH=/usr/bin", "FOO=a=b c", "NO_VALUE"], "WorkingDir": "/srv"},
            True,
            ImageShell(user="app", env={"PATH": "/usr/bin", "FOO": "a=b c"}, workdir="/srv"),
        ),
        ({}, True, ImageShell(user="root", env={}, workdir="/")),
    ],
    ids=["user-and-directory", "with-the-env", "nothing-named"],
)
def test_with_image_user_the_commands_take_the_images_user_and_directory(
    registry, config, image_env, expected
):
    """The ENV comes with ``image_env`` only. An image that names no user runs as root, in ``/``."""
    registry(config)
    backend = {"backend": "microvm", "microvm": {"image_user": True, "image_env": image_env}}

    shell = image_shell(microvm(image_user=True, image_env=image_env), backend=backend, image=IMAGE)

    assert shell == expected


def test_a_microvm_image_without_a_reference_cannot_give_its_user():
    """A branch of a run without a Harbor task starts from a snapshot, which no OCI reference names."""
    backend = {"backend": "microvm", "microvm": {"image_user": True}}

    with pytest.raises(
        TemplateError, match="^no OCI image reference names the image whose config is to be read$"
    ):
        image_shell(microvm(image_user=True), backend=backend, image=None)


def run_in_sh(line: str) -> str:
    """The stdout of ``line`` in a real ``/bin/sh``, as the runtime runs a command."""
    return subprocess.run(["/bin/sh", "-c", line], cwd="/", capture_output=True, text=True, timeout=30).stdout


def test_a_command_runs_in_the_images_directory_with_the_images_env(tmp_path):
    """The user is root here, so the line has no ``runuser``."""
    shell = ImageShell(user="root", env={"GREETING": "a b"}, workdir=str(tmp_path))

    line = shell.command('pwd; printf "%s\\n" "$GREETING"', working_dir=None)

    assert run_in_sh(line) == "%s\na b\n" % tmp_path


def test_the_callers_directory_wins_over_the_images(tmp_path):
    (tmp_path / "repo").mkdir()
    shell = ImageShell(user="root", env={}, workdir="/")

    line = shell.command("pwd", working_dir=str(tmp_path / "repo"))

    assert run_in_sh(line) == "%s\n" % (tmp_path / "repo")
