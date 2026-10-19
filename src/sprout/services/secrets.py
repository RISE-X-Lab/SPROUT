"""The generated secrets: one file for each name under ``<state>/secrets``.

A secret is random, written once and then read back. So ``init``, ``up`` and ``env``
see the same value, and nobody types a secret into a config file. The directory has
mode 0700 and each file mode 0600. :func:`ensure` sets both modes again on each call,
because a ``umask`` or a copy can widen them. No function here prints or logs a value.
"""

from __future__ import annotations

import os
import re
import secrets as _secrets
from pathlib import Path

from sprout.services.state import Paths

#: A valid secret name: lower-case letters, digits and dashes, so that it is also a safe file name.
NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def check_name(name: str) -> str:
    """``name`` if it is a valid secret name. Raises ``ValueError`` if it is not."""
    if not NAME.match(name):
        raise ValueError("secret name %r: use lower-case letters, digits and dashes" % name)
    return name


def path(paths: Paths, name: str) -> Path:
    """The file of the secret ``name``. Raises ``ValueError`` for an invalid name."""
    return paths.secrets / check_name(name)


def exists(paths: Paths, name: str) -> bool:
    """True if the secret ``name`` has a file with a value that is not empty."""
    file = path(paths, name)
    return file.is_file() and bool(file.read_text(encoding="utf-8").strip())


def require(paths: Paths, name: str) -> str:
    """The value of the secret ``name``, for a command that must not create one.

    Raises ``ValueError``, which names ``init`` as the fix, if the secret does not exist.
    """
    if not exists(paths, name):
        raise ValueError("secret %r does not exist yet: run `python -m sprout.services init`" % name)
    return path(paths, name).read_text(encoding="utf-8").strip()


def ensure(paths: Paths, name: str) -> str:
    """The value of the secret ``name``, generated and written on the first call.

    Each later call returns the value that the first call wrote.
    """
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
    # Write a temporary file and rename it over the target, so that no reader
    # sees a partly written value.
    tmp = file.with_name(file.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(value + "\n")
    os.chmod(tmp, 0o600)
    os.replace(tmp, file)
    return value
