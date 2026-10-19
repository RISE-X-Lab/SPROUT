"""The text that the CLI prints: export lines, the status table and doctor lines.

The output is plain text with no colour, for a person over ssh and for a shell. The
``env`` lines can be evaluated as they are, and the status table has fixed-width
columns. Only :func:`env_lines` receives secret values. The other functions receive
probe details and pids, never an environment.
"""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

COLUMNS = ("name", "kind", "pid", "health", "port")


@dataclass(frozen=True)
class StatusRow:
    """One service's line of the status table, one cell for each of ``COLUMNS``."""

    name: str
    kind: str
    pid: str
    health: str
    port: str


def env_lines(exports: Mapping[str, str]) -> str:
    """One ``export NAME=value`` line for each variable, quoted for ``eval "$(...)"`` in a shell."""
    return "".join("export %s=%s\n" % (name, shlex.quote(value)) for name, value in exports.items())


def status_table(rows: Sequence[StatusRow]) -> str:
    """A header line and one line for each row, each column as wide as its widest cell."""
    cells = [COLUMNS] + [(row.name, row.kind, row.pid, row.health, row.port) for row in rows]
    widths = [max(len(line[i]) for line in cells) for i in range(len(COLUMNS))]
    return "".join(
        " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() + "\n" for line in cells
    )


def doctor_line(word: str, subject: str, detail: str) -> str:
    """One doctor line: the status word, the subject and the detail, in fixed-width columns."""
    return "%-5s %-16s %s" % (word, subject, detail)
