"""A stand-in for the attempt process: it writes one turn, waits until the worker persisted it, and ends.

The worker starts it as ``<python> -m sprout.worker.jobs.rollout <attempt directory> <digest>``, through
a launcher that a test names as its profile's ``python``. It writes turn 1 to the journal and the native
history, and each file appears whole. Then it waits up to 30 s for the file ``persisted`` in the attempt
directory, and writes a completed ``outcome.json``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from tests.worker.checkpoint.native_records import SESSION, add_turn, attempt_files


def main(directory: Path) -> None:
    staging = directory / "staging"
    journal, native = attempt_files(staging)
    add_turn(journal, native, 1)
    (directory / "native-home").mkdir()
    os.replace(native, directory / "native-home" / ("%s.jsonl" % SESSION))
    os.replace(journal, directory / "trajectory.jsonl")
    deadline = time.monotonic() + 30
    while not (directory / "persisted").exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    (directory / "outcome.json").write_text(json.dumps({"status": "completed"}))


if __name__ == "__main__":
    main(Path(sys.argv[3]))
