"""``child.execute`` is the whole life of a Run Store child process: it sets the
umask to 077 and exports the profile's environment, which is right there and
nowhere else. These tests call it in-process, so each one gives the process
back what it had -- a leaked 077 once made every podman base extracted later in
the same session unreadable to an image's non-root user."""

import os

import pytest


@pytest.fixture(autouse=True)
def child_process_state():
    umask = os.umask(0o022)
    os.umask(umask)
    environ = dict(os.environ)
    yield
    os.umask(umask)
    os.environ.clear()
    os.environ.update(environ)
