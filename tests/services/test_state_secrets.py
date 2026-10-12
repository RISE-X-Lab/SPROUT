"""Where the files go, and that a secret is made once, tightly, and never again."""

from __future__ import annotations

import stat

import pytest

from sprout.services import secrets
from sprout.services.state import Paths, repo_root


def test_paths_follow_the_override_variables_and_the_xdg_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("SPROUT_SERVICES_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SPROUT_SERVICES_STATE_DIR", raising=False)
    default = Paths.from_env()
    assert default.config == tmp_path / ".config" / "sprout"
    assert default.state == tmp_path / ".local" / "state" / "sprout" / "services"
    assert default.spec == tmp_path / ".config" / "sprout" / "services.yaml"
    assert default.repo == repo_root() and (default.repo / "pyproject.toml").is_file()

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert Paths.from_env().config == tmp_path / "xdg" / "sprout"

    monkeypatch.setenv("SPROUT_SERVICES_CONFIG_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("SPROUT_SERVICES_STATE_DIR", str(tmp_path / "s"))
    moved = Paths.from_env()
    assert (moved.config, moved.state) == (tmp_path / "c", tmp_path / "s")
    assert moved.pid_file("db") == tmp_path / "s" / "db" / "pid"
    assert moved.log_file("db") == tmp_path / "s" / "db" / "log"
    assert moved.secrets == tmp_path / "s" / "secrets"


def test_secrets_are_generated_once_with_tight_modes(paths):
    assert not secrets.exists(paths, "db-password")
    with pytest.raises(ValueError, match="run `python -m sprout.services init`"):
        secrets.require(paths, "db-password")
    first = secrets.ensure(paths, "db-password")
    assert len(first) >= 32 and secrets.ensure(paths, "db-password") == first == secrets.require(paths, "db-password")
    assert secrets.ensure(paths, "runstore-token") != first
    assert stat.S_IMODE(paths.secrets.stat().st_mode) == 0o700
    for file in paths.secrets.iterdir():
        assert stat.S_IMODE(file.stat().st_mode) == 0o600, file
    assert sorted(f.name for f in paths.secrets.iterdir()) == ["db-password", "runstore-token"]
    with pytest.raises(ValueError, match="lower-case letters, digits and dashes"):
        secrets.ensure(paths, "../etc")
