"""The attempt child, ``main`` of ``sprout.worker.jobs.rollout``, on the sandboxes of ``sandboxes.py``.

``python -m tests.worker.jobs.sandboxed_child [options] <root> <attempt directory> <payload digest>
<worker pid>`` runs the child as the worker runs it, with the payload on stdin. Its sandboxes are under
``<root>/sandboxes``, as ``run_sandboxes`` makes them, and start from the image ``base`` in
``<root>/images/base``. These options apply:

- ``--hold-destroys``: each removal of a sandbox creates ``<root>/DESTROY_HELD``, and then waits until
  ``<root>/DESTROY_RELEASED`` exists.
- ``--teardown-s <seconds>``: the ``teardown_s`` of ``main``, ``TEARDOWN_S`` by default.
- ``--worker-check-s <seconds>``: the ``worker_check_s`` of ``main``, ``WORKER_CHECK_S`` by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sprout.worker.sandbox.session
from sprout.worker.jobs.rollout import TEARDOWN_S, WORKER_CHECK_S, main
from tests.worker.jobs.sandboxes import DirectorySandboxes

#: The file under the root that a held removal of a sandbox creates once it waits.
DESTROY_HELD = "destroy-held"
#: The file under the root whose creation lets each held removal of a sandbox go on.
DESTROY_RELEASED = "destroy-released"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-destroys", action="store_true")
    parser.add_argument("--teardown-s", type=float, default=TEARDOWN_S)
    parser.add_argument("--worker-check-s", type=float, default=WORKER_CHECK_S)
    parser.add_argument("root", type=Path)
    # main reads the attempt directory, the digest and the worker pid from sys.argv.
    options, sys.argv[1:] = parser.parse_known_args()
    sandboxes = DirectorySandboxes(
        options.root / "sandboxes", images={"base": options.root / "images" / "base"}
    )
    if options.hold_destroys:
        sandboxes.hold_destroys(held=options.root / DESTROY_HELD, released=options.root / DESTROY_RELEASED)
    # The runner makes its session where it needs one, from this module attribute.
    sprout.worker.sandbox.session.SandboxSession = sandboxes
    raise SystemExit(main(teardown_s=options.teardown_s, worker_check_s=options.worker_check_s))
