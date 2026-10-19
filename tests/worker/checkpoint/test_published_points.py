"""The recovery points that a worker publishes for an attempt whose journal grows while the worker reads it.

A medium test: a Run Store on the PostgreSQL test database, and a stand-in attempt process
(``turn_writer.py``) that the worker starts and polls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.worker.attempts import Worker
from tests.support.runstore import registered
from tests.worker.checkpoint.native_records import SESSION, add_turn

TURN_WRITER = Path(__file__).resolve().parent / "turn_writer.py"
LAUNCHER = "#!%(python)s\nimport runpy\nrunpy.run_path(%(script)r, run_name='__main__')\n"


class WriterAdvancingStore(Store):
    """A Run Store whose first ``append_events`` lets the attempt's process append turn 2 to its files.

    So the writer advances after the worker read the events that it persists. At each ``append_events``,
    the store also records the depth of each published point whose tool call it does not hold."""

    def __init__(self, dsn: str, *, artifact_root: Path) -> None:
        super().__init__(dsn)
        self._artifact_root = artifact_root
        self._advanced = False
        #: The tool depths of the points that were published without their tool call, at each persist.
        self.unheld_points: list[int] = []

    def append_events(self, job_id: str, token: str, events: list[dict[str, Any]]) -> None:
        attempt = self.get(job_id)["active_attempt"]
        held = sum(event["type"] == "tool.started" for event in self.events(attempt))
        self.unheld_points += [p["tool_depth"] for p in Index(self).points(attempt) if p["tool_depth"] > held]
        if not self._advanced:
            self._advanced = True
            directory = self._artifact_root / job_id / attempt
            add_turn(directory / "trajectory.jsonl", directory / "native-home" / ("%s.jsonl" % SESSION), 2)
            (directory / "persisted").touch()
        super().append_events(job_id, token, events)


def launcher(tmp_path: Path) -> Path:
    """An executable that runs ``turn_writer.py`` in the place of the profile's interpreter."""
    path = tmp_path / "python"
    path.write_text(LAUNCHER % {"python": sys.executable, "script": str(TURN_WRITER)})
    path.chmod(0o755)
    return path


def test_a_worker_publishes_only_the_points_of_the_journal_frame_that_it_persisted(tmp_path, run_store):
    """A reader of the Run Store must find the tool call of each point that the worker published."""
    root = (tmp_path / "artifacts").resolve()
    profile = {"python": str(launcher(tmp_path)), "run_defaults": {"sandbox_image": "image"}}
    store = WriterAdvancingStore(run_store.dsn, artifact_root=root)
    job = store.submit(JobSpec("rollout", {"prompt": "fixture"}), "frame")
    worker = Worker(
        store,
        {"artifact_root": str(root), "profiles": {"default": profile}},
        worker_id="worker-1",
        snapshot_valid=lambda point: True,
    )
    registered(store, "worker-1")
    claimed = store.claim("worker-1")

    worker.run_claimed(claimed)

    points = Index(store).points(claimed["active_attempt"])
    assert (store.get(job["id"])["state"], store.unheld_points) == ("succeeded", [])
    assert [(point["tool_depth"], point["valid"]) for point in points] == [(1, True), (2, True)]
