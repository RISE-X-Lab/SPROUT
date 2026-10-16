"""Where durable run artifacts may live.

A journal or an artifact root under a volatile directory is refused.
"""

from __future__ import annotations

from pathlib import Path

#: Directories the OS empties on reboot (and tmpwatch empties sooner). A journal
#: is the ONLY record a run leaves -- snapshot ids, every step, the grading
#: evidence -- so writing one here schedules its own destruction. Learned the
#: expensive way: a 32-instance batch's journals lived in /tmp when the host
#: rebooted mid-regrade, and several hours of agent time now exist only as prose.
VOLATILE_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run")


def volatile_reason(path: str | Path) -> str | None:
    """Why this path will not survive a reboot, or None if it should.

    Entry points refuse volatile journal destinations; the library layer does
    not, because tests legitimately journal into pytest's /tmp fixtures. The
    check is a resolved-prefix test rather than a filesystem-type probe --
    simple, and it catches the mistake that was actually made.
    """
    resolved = Path(path).resolve()
    for root in VOLATILE_ROOTS:
        if resolved == Path(root) or str(resolved).startswith(root + "/"):
            return (
                "%s is under %s, which the OS empties on reboot -- a journal "
                "is the only record a run leaves" % (resolved, root)
            )
    return None
