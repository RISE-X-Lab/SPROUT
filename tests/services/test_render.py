"""The text that the CLI prints: ``env`` lines that a shell evaluates as they are, and the status table."""

from __future__ import annotations

import json
import subprocess
import sys

from sprout.services.render import StatusRow, env_lines, status_table

#: Prints, as JSON, the value of each variable named in its arguments that it finds in its environment.
PRINT_ENVIRONMENT = "import json, os, sys; print(json.dumps({k: os.environ.get(k) for k in sys.argv[1:]}))"


def exported_by_sh(lines: str, *names: str) -> dict[str, str | None]:
    """The values that a child of ``sh`` sees after ``sh`` evaluates ``lines``, like an experiment shell."""
    script = 'eval "$LINES"\nexec "$PYTHON" -c "$PRINT" "$@"'
    done = subprocess.run(
        ["/bin/sh", "-c", script, "sh", *names],
        env={"LINES": lines, "PYTHON": sys.executable, "PRINT": PRINT_ENVIRONMENT},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return json.loads(done.stdout)


def test_a_shell_that_evaluates_the_env_lines_exports_each_value_exactly():
    exports = {
        "PLAIN": "http://127.0.0.1:4000/v1",
        "SPACES": "host=127.0.0.1 port=15432 dbname=sprout_runstore",
        "QUOTES": 'it\'s "quoted"',
        "SHELL_WORDS": "$HOME `id` $(id) ; exit 1",
        "LINES_TWO": "first\nsecond",
        "EMPTY": "",
    }

    seen = exported_by_sh(env_lines(exports), *exports)

    assert seen == exports


def test_the_status_table_pads_each_column_to_its_widest_cell():
    rows = [
        StatusRow("db", "podman", "4242", "ok: 127.0.0.1:15432 - accepting connections", "15432"),
        StatusRow("runstore-worker", "process", "-", "down: no pid file", "-"),
    ]

    table = status_table(rows)

    assert table == (
        "name            | kind    | pid  | health                                      | port\n"
        "db              | podman  | 4242 | ok: 127.0.0.1:15432 - accepting connections | 15432\n"
        "runstore-worker | process | -    | down: no pid file                           | -\n"
    )
