"""Where the CLI keeps its files, and how a test moves them somewhere harmless.

Config lives under the XDG config home (``~/.config/sprout``: ``services.yaml``
and the per-service config files ``init`` seeds beside it), state under the XDG
state home (``~/.local/state/sprout/services``: one directory per service with
its ``pid`` and ``log``, ``secrets/`` with the generated secrets, the proxy's
venv, the database's data directory). The split follows the specification's
meaning: config is what a person edits, state is what the programs produce.

Two environment variables move the roots, ``SPROUT_SERVICES_CONFIG_DIR`` and
``SPROUT_SERVICES_STATE_DIR``. Every test sets both to a ``tmp_path``, so no
test can touch the real home even when it is wrong, and a second instance of
everything (another database, another proxy, free ports) is the same two
variables plus edited ports.

The repository root is found from this file (``src/sprout/services`` is three
levels below it), so the templates never spell a checkout path: the default
interpreter is that checkout's ``.venv`` and the runtime binary its
``runtime/sprout-runtime``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

CONFIG_DIR_VAR = "SPROUT_SERVICES_CONFIG_DIR"
STATE_DIR_VAR = "SPROUT_SERVICES_STATE_DIR"
SPEC_NAME = "services.yaml"


def repo_root() -> Path:
    """The checkout this package was imported from: ``src/sprout/services`` -> repo."""
    return Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Paths:
    """The two roots and the checkout, resolved once per command."""

    config: Path
    state: Path
    repo: Path

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Paths":
        env = os.environ if environ is None else environ
        home = Path(env.get("HOME") or Path.home())
        config = env.get(CONFIG_DIR_VAR)
        state = env.get(STATE_DIR_VAR)
        if config:
            config_dir = Path(config).expanduser()
        else:
            config_dir = Path(env.get("XDG_CONFIG_HOME") or home / ".config") / "sprout"
        if state:
            state_dir = Path(state).expanduser()
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
        """The service's own directory: pid, log, and whatever it is run from."""
        return self.state / name

    def pid_file(self, name: str) -> Path:
        return self.service(name) / "pid"

    def log_file(self, name: str) -> Path:
        return self.service(name) / "log"

    def default_python(self) -> str:
        """The checkout's venv when it exists, else the interpreter running the CLI.

        An installed wheel has no ``.venv`` beside it; whatever runs the CLI can
        import ``sprout`` and is the next best guess. ``services.yaml`` may name
        another with ``python:``.
        """
        venv = self.repo / ".venv" / "bin" / "python"
        return str(venv) if venv.exists() else sys.executable

    def runtime_binary(self) -> Path:
        return self.repo / "runtime" / "sprout-runtime"
