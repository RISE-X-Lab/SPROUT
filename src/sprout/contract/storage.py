"""Whether a path for a journal or an artifact root survives a reboot.

A journal is the only record that a run leaves: its snapshot ids, every step and the grading
evidence. A journal under a directory that the OS empties on reboot is lost with it.
"""

from __future__ import annotations

from pathlib import Path

#: Directories that the OS empties on reboot, or that tmpwatch empties sooner.
VOLATILE_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run")


def volatile_reason(path: str | Path) -> str | None:
    """Why ``path`` will not survive a reboot, or ``None`` if it will.

    The test is on the prefix of the resolved path, not on the filesystem type. Only an entry point
    refuses such a path: the library accepts one, because tests journal into pytest's temporary paths.
    """
    resolved = Path(path).resolve()
    for root in VOLATILE_ROOTS:
        if resolved == Path(root) or str(resolved).startswith(root + "/"):
            return (
                "%s is under %s, which the OS empties on reboot -- a journal "
                "is the only record a run leaves" % (resolved, root)
            )
    return None
