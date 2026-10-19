"""Where the services CLI keeps its files, and how a test moves them somewhere harmless.

Config is what a person edits: ``services.yaml`` and the per-service config files that
``init`` writes beside it, under the XDG config home (``~/.config/sprout``). State is what
the programs produce: one directory per service with its ``pid`` and ``log``, the
``secrets/``, the proxy's venv and the database's data directory, under the XDG state home
(``~/.local/state/sprout/services``).

``SPROUT_SERVICES_CONFIG_DIR`` and ``SPROUT_SERVICES_STATE_DIR`` move the two roots. Each
test sets both to a temporary directory, so no test can touch the real home. A second
instance of every service needs the same two variables and other ports.

The repository root is the checkout that this package is imported from, so the templates
never name a checkout path. The default interpreter is the checkout's ``.venv``, and the
runtime binary is its ``runtime/sprout-runtime``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

CONFIG_DIR_VAR = "SPROUT_SERVICES_CONFIG_DIR"
STATE_DIR_VAR = "SPROUT_SERVICES_STATE_DIR"
SPEC_NAME = "services.yaml"


def repo_root() -> Path:
    """The root of the checkout that this package was imported from."""
    # This file is src/sprout/services/state.py, three directories below the root.
    return Path(__file__).resolve().parents[3]


def _expand_home(value: str, *, home: Path) -> Path:
    """``value`` as a path, with a leading ``~`` as ``home`` and a leading ``~user`` as that user's home."""
    path = Path(value)
    if path.parts[:1] == ("~",):
        return home.joinpath(*path.parts[1:])
    return path.expanduser()


@dataclass(frozen=True)
class Paths:
    """The config root, the state root and the checkout, resolved once for each command."""

    config: Path
    state: Path
    repo: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Paths:
        """The paths that ``environ`` (default ``os.environ``) gives.

        Each root is its override variable if set, else its XDG home, else the default under ``HOME``.
        A leading ``~`` of an override is the ``HOME`` of ``environ``.
        """
        env = os.environ if environ is None else environ
        home = Path(env.get("HOME") or Path.home())
        config = env.get(CONFIG_DIR_VAR)
        state = env.get(STATE_DIR_VAR)
        if config:
            config_dir = _expand_home(config, home=home)
        else:
            config_dir = Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "sprout"
        if state:
            state_dir = _expand_home(state, home=home)
        else:
            state_dir = Path(env.get("XDG_STATE_HOME") or home / ".local" / "state") / "sprout" / "services"
        return cls(config=config_dir, state=state_dir, repo=repo_root())

    @property
    def spec(self) -> Path:
        return self.config / SPEC_NAME

    @property
    def secrets(self) -> Path:
        return self.state / "secrets"

    def service(self, name: str) -> Path:
        """The state directory of the service ``name``: its pid file, its log and its working directory."""
        return self.state / name

    def pid_file(self, name: str) -> Path:
        return self.service(name) / "pid"

    def log_file(self, name: str) -> Path:
        return self.service(name) / "log"

    def default_python(self) -> str:
        """The checkout's ``.venv`` interpreter if it exists, else the interpreter that runs the CLI.

        An installed wheel has no ``.venv`` beside it, and the CLI's own interpreter can
        import ``sprout``. ``services.yaml`` can name another interpreter with ``python:``.
        """
        venv = self.repo / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def runtime_binary(self) -> Path:
        return self.repo / "runtime" / "sprout-runtime"
