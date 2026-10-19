"""Each test gives the process back the umask and the environment that the test found.

Some code under test changes both for the whole process. A Run Store child, which tests run
in-process, sets the umask to 077 and exports the environment of its profile. A leaked 077 makes each
base that a later test extracts unreadable to the non-root user of its image. ``restore_process_state``
is autouse for the whole suite, so no test sees what an earlier test changed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pytest


@contextmanager
def kept_process_state() -> Iterator[None]:
    """Set the umask and ``os.environ`` back to their values at the start when the block exits."""
    umask = os.umask(0o022)
    os.umask(umask)
    environ = dict(os.environ)
    try:
        yield
    finally:
        os.umask(umask)
        os.environ.clear()
        os.environ.update(environ)


@pytest.fixture(autouse=True)
def restore_process_state() -> Iterator[None]:
    """Each test runs in ``kept_process_state``."""
    with kept_process_state():
        yield
