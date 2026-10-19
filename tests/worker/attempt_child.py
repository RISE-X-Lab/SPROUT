"""A stand-in for the attempt process, which the worker starts through the ``python`` of a profile.

The worker runs ``<python> -m sprout.worker.jobs.rollout <attempt directory> <payload digest> <worker pid>``
and writes the payload to its stdin. ``child_python`` writes an executable that the worker can run as
that ``python``. The stand-in records what it got in the attempt directory, and reads the payload
with the checks of the attempt process. Then it acts as its behavior says. It starts no sandbox and
calls no model. The behaviors are these, after the payload:

- ``complete`` writes a completed outcome, and ``slow`` does the same after 1.5 s.
- ``hold`` waits until a signal stops it.
- ``exit`` exits without an outcome.
- ``crash`` prints ``LAST_WORDS``, writes a journal of the snapshots ``snap-1`` and ``snap-2`` of its
  podman store, and exits without an outcome.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from sprout.runstore.payload import receive_payload
from sprout.worker.attempts import process_identity
from sprout.worker.files import write_json

#: What the stand-in writes before it reads stdin: its argv, its process identity and its environment.
STARTED = "child.json"
#: The payload that the stand-in read, written once the payload passed the checks.
RECEIVED = "received.json"
#: What the ``crash`` behavior prints before it exits.
LAST_WORDS = "the stand-in child crashes now"
_LAUNCHER = (
    "#!%(python)s\nfrom tests.worker.attempt_child import main\nraise SystemExit(main(%(behavior)r))\n"
)


def child_python(directory: Path, behavior: str = "complete") -> str:
    """The path of an executable in ``directory`` that runs the stand-in with ``behavior``.

    The module docstring gives the behaviors."""
    launcher = directory / ("python-%s" % behavior)
    launcher.write_text(_LAUNCHER % {"python": sys.executable, "behavior": behavior})
    launcher.chmod(0o755)
    return str(launcher)


def main(behavior: str) -> int:
    directory, fingerprint = Path(sys.argv[3]), sys.argv[4]
    identity = process_identity(os.getpid())
    write_json(directory / STARTED, {"argv": sys.argv[1:], "identity": identity, "env": dict(os.environ)})
    try:
        payload = receive_payload(sys.stdin.fileno(), fingerprint, directory.parent.name, directory.name)
    except (OSError, ValueError):
        return 2
    write_json(directory / RECEIVED, payload)
    if behavior == "hold":
        time.sleep(600)
    if behavior == "slow":
        time.sleep(1.5)
    if behavior == "crash":
        _crash(directory, payload["effective_spec"]["backend"]["podman"]["store"])
    if behavior not in ("exit", "crash"):
        write_json(directory / "outcome.json", {"status": "completed"})
    return 0


def _crash(directory: Path, store: str) -> None:
    records = [
        {"seq": 1, "type": "sandbox.created", "sandbox_id": "sprout-1", "backend": "podman", "store": store},
        {"seq": 2, "type": "checkpoint.captured", "step": 1, "snapshot_id": "snap-1"},
        {"seq": 3, "type": "checkpoint.captured", "step": 2, "snapshot_id": "snap-2"},
    ]
    (directory / "trajectory.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))
    print(LAST_WORDS, flush=True)
