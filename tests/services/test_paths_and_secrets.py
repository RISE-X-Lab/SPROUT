"""Where the services CLI keeps its files, and the secrets: made once, then read back, by their owner only."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from sprout.services import secrets
from sprout.services.state import Paths
from tests.support.paths import REPO


def paths_in(tmp_path: Path) -> Paths:
    return Paths(config=tmp_path / "config", state=tmp_path / "state", repo=tmp_path / "repo")


def mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# --- paths --------------------------------------------------------------------


@pytest.mark.parametrize(
    "variables, config, state",
    [
        (
            {
                "SPROUT_SERVICES_CONFIG_DIR": "/path/to/config",
                "XDG_CONFIG_HOME": "/path/to/xdg-config",
                "XDG_STATE_HOME": "/path/to/xdg-state",
            },
            "/path/to/config",
            "/path/to/xdg-state/sprout/services",
        ),
        (
            {"XDG_CONFIG_HOME": "/path/to/xdg-config"},
            "/path/to/xdg-config/sprout",
            "/path/to/home/.local/state/sprout/services",
        ),
        (
            {"SPROUT_SERVICES_STATE_DIR": "/path/to/state", "XDG_STATE_HOME": "/path/to/xdg-state"},
            "/path/to/home/.config/sprout",
            "/path/to/state",
        ),
    ],
)
def test_each_root_is_its_override_else_its_xdg_home_else_the_default_under_home(variables, config, state):
    paths = Paths.from_env({"HOME": "/path/to/home", **variables})

    assert (paths.config, paths.state) == (Path(config), Path(state))


def test_a_leading_tilde_in_an_override_is_the_home_of_the_given_environment(monkeypatch):
    monkeypatch.setenv("HOME", "/path/to/process-home")

    paths = Paths.from_env({"HOME": "/path/to/home", "SPROUT_SERVICES_CONFIG_DIR": "~/sprout-config"})

    assert paths.config == Path("/path/to/home/sprout-config")


def test_the_files_of_a_service_are_where_docs_services_md_says():
    paths = Paths(config=Path("/path/to/config"), state=Path("/path/to/state"), repo=Path("/path/to/repo"))

    assert (
        paths.spec,
        paths.secrets,
        paths.pid_file("db"),
        paths.log_file("db"),
        paths.runtime_binary(),
    ) == (
        Path("/path/to/config/services.yaml"),
        Path("/path/to/state/secrets"),
        Path("/path/to/state/db/pid"),
        Path("/path/to/state/db/log"),
        Path("/path/to/repo/runtime/sprout-runtime"),
    )


def test_the_repository_is_the_checkout_that_the_package_is_imported_from():
    """The templates name no checkout path: ``{repo}`` and the runtime binary come from here."""
    assert Paths.from_env({"HOME": "/path/to/home"}).repo == REPO


def test_the_default_interpreter_is_the_checkouts_venv_when_it_has_one(tmp_path):
    venv_python = tmp_path / "repo" / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.touch()

    assert paths_in(tmp_path).default_python() == str(venv_python)


def test_without_a_venv_the_default_interpreter_is_the_one_that_runs_the_cli(tmp_path):
    """An installed wheel has no ``.venv`` beside it."""
    assert paths_in(tmp_path).default_python() == sys.executable


# --- secrets ------------------------------------------------------------------


def test_a_secret_is_made_on_the_first_call_and_read_back_after_it(tmp_path):
    paths = paths_in(tmp_path)

    first = secrets.ensure(paths, "db-password")

    assert (secrets.ensure(paths, "db-password"), secrets.require(paths, "db-password")) == (first, first)


def test_a_new_secret_is_readable_by_its_owner_only_whatever_the_umask(tmp_path):
    paths = paths_in(tmp_path)
    os.umask(0)

    secrets.ensure(paths, "db-password")

    assert (mode(paths.secrets), mode(paths.secrets / "db-password")) == (0o700, 0o600)


def test_ensure_narrows_the_modes_of_an_existing_secret_again(tmp_path):
    """A copy or a umask can widen them after ``init``."""
    paths = paths_in(tmp_path)
    secrets.ensure(paths, "db-password")
    os.chmod(paths.secrets, 0o755)
    os.chmod(paths.secrets / "db-password", 0o644)

    secrets.ensure(paths, "db-password")

    assert (mode(paths.secrets), mode(paths.secrets / "db-password")) == (0o700, 0o600)


def test_require_of_a_secret_that_was_never_made_names_init_as_the_fix_and_makes_nothing(tmp_path):
    paths = paths_in(tmp_path)

    with pytest.raises(ValueError) as caught:
        secrets.require(paths, "db-password")

    assert (str(caught.value), paths.secrets.exists()) == (
        "secret 'db-password' does not exist yet: run `python -m sprout.services init`",
        False,
    )


@pytest.mark.parametrize("name", ["../etc", "db/password", "DB", ""])
def test_a_secret_name_that_is_not_a_plain_file_name_is_refused(tmp_path, name):
    with pytest.raises(ValueError, match="use lower-case letters, digits and dashes"):
        secrets.ensure(paths_in(tmp_path), name)
