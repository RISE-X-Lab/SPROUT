"""The generated secrets: one file per name under ``<state>/secrets``.

The database password, the proxy's master key and the two service tokens are
random (``secrets.token_urlsafe(32)``), written once and read back ever after,
so ``init``, ``up`` and ``env`` all see the same value and nothing has to be
typed, remembered or pasted into a config file (a test scans every tracked
config for exactly that mistake). The directory is mode 0700 and every file
0600; both are re-applied on each use because a ``umask`` or a copy can widen
them. A value is never logged: the CLI prints names, and ``env`` is the one
command that prints a value.
"""

from __future__ import annotations

import os
import re
import secrets as _secrets
from pathlib import Path

from sprout.services.state import Paths

#: Lower-case words joined by dashes, so a name is also a safe file name.
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def check_name(name: str) -> str:
    if not NAME.match(name):
        raise ValueError("secret name %r: use lower-case letters, digits and dashes" % name)
    return name


def path(paths: Paths, name: str) -> Path:
    return paths.secrets / check_name(name)


def exists(paths: Paths, name: str) -> bool:
    file = path(paths, name)
    return file.is_file() and bool(file.read_text(encoding="utf-8").strip())


def require(paths: Paths, name: str) -> str:
    """The value, or a ValueError naming the fix; for commands that must not create."""
    if not exists(paths, name):
        raise ValueError("secret %r does not exist yet: run `python -m sprout.services init`" % name)
    return path(paths, name).read_text(encoding="utf-8").strip()


def ensure(paths: Paths, name: str) -> str:
    """The value, generated on first use. Idempotent: the second call reads the first's file."""
    file = path(paths, name)
    directory = file.parent
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    if file.is_file():
        existing = file.read_text(encoding="utf-8").strip()
        if existing:
            os.chmod(file, 0o600)
            return existing
    value = _secrets.token_urlsafe(32)
    # Written beside and renamed over: a crash between open and write would
    # otherwise leave an empty file that the next call reads as "exists".
    tmp = file.with_name(file.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, file)
    return value
