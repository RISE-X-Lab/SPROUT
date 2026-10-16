"""What the CLI prints: export lines, the status table, doctor lines. Plain text.

No colour and no decoration, because the output is meant to be read by a
person over ssh and by a shell: ``env``'s lines are ``eval``-able as they
stand, ``status`` is a fixed-width table that greps and diffs well. Secrets
reach this module only through ``env``, by design; the status and doctor
renderers are handed probe details and pids, never an environment.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Mapping, Sequence

COLUMNS = ("name", "kind", "pid", "health", "port")


@dataclass(frozen=True)
class StatusRow:
    name: str
    kind: str
    pid: str
    health: str
    port: str


def env_lines(exports: Mapping[str, str]) -> str:
    """``export NAME=value`` per variable, quoted for the shell (``eval "$(...)"``)."""
    return "".join("export %s=%s\n" % (name, shlex.quote(value)) for name, value in exports.items())


def status_table(rows: Sequence[StatusRow]) -> str:
    cells = [COLUMNS] + [(row.name, row.kind, row.pid, row.health, row.port) for row in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(COLUMNS))]
    return "".join(
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() + "\n" for line in cells
    )


def doctor_line(word: str, subject: str, detail: str) -> str:
    return "%-5s %-16s %s" % (word, subject, detail)
