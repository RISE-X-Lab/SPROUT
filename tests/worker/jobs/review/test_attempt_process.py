"""A review in the attempt process: stopped as the worker stops it, it exits within the worker's grace.

The test is medium. It starts ``python -m sprout.worker.jobs.rollout`` with a review payload, against an
endpoint on 127.0.0.1 that accepts each request and never answers it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys

from sprout.runstore.payload import encode_payload, send_payload
from sprout.runstore.specs import digest
from tests.support.http import silent_endpoint
from tests.support.paths import REPO
from tests.worker.jobs.review.review_model import failed_roots, review_spec

#: How long ``Worker.run_claimed`` waits for a stopped attempt process to exit, in seconds.
WORKER_GRACE_S = 5


def test_a_stopped_review_writes_its_outcome_and_exits_while_its_requests_are_still_out(tmp_path):
    """The worker sends SIGTERM to the process group, and then waits 5 s before it gives up on the attempt."""
    root = tmp_path / "artifacts"
    directory = tmp_path / "jobs" / "review-job" / "attempt-1"
    directory.mkdir(parents=True)
    log = tmp_path / "worker.log"

    with silent_endpoint(connections=2) as (url, both_out), log.open("w") as output:
        spec = review_spec(root, failed_roots(root, 2), model_endpoint=url, points=1, points_per_attempt=True)
        envelope = {"version": 1, "job_id": "review-job", "attempt_id": "attempt-1", "kind": "review"}
        envelope.update(effective_spec=spec, profile_config={}, recovery=None)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sprout.worker.jobs.rollout",
                str(directory),
                digest(envelope),
                str(os.getpid()),
            ],
            cwd=REPO,
            env={**os.environ, "REVIEW_KEY": "review-secret"},
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            send_payload(process.stdin, encode_payload(envelope, "review-job", "attempt-1"), timeout_s=30)
            assert both_out.wait(60), log.read_text()

            os.killpg(process.pid, signal.SIGTERM)
            returncode = process.wait(timeout=WORKER_GRACE_S)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    outcome = json.loads((directory / "outcome.json").read_text())
    assert (returncode, outcome["status"], outcome["error"]) == (
        0,
        "error",
        "KeyboardInterrupt: Worker requested shutdown",
    )
    assert "Traceback" in log.read_text(), "the attempt's error reaches worker.log in full"
