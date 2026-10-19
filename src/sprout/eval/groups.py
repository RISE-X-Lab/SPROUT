"""The groups of jobs of one evaluation or gate run: kept in its output, run by the search coordinator.

``FileGroups`` is a ``GroupStore`` (``sprout.search.coordinator``) whose documents are
JSON files in one directory, so an output keeps the record of every Run Store job it
made. ``Groups`` hands a group request to the coordinator and follows the group until
every one of its jobs has settled, or one of them lost its data with its host. Nothing
here runs a job: the Run Store's workers do.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sprout.search.client import RunStoreClient
from sprout.search.coordinator import Coordinator
from sprout.search.group import Submitter, initial_document, validate_request

#: Where the Run Store listens unless a client is told otherwise (``python -m sprout.runstore serve``).
DEFAULT_RUNSTORE_URL = "http://127.0.0.1:18110"


class FileGroups:
    """A group store in a directory: a ``<group id>.json`` for each group, rewritten whole on each change."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def create(self, group_id: str, document: dict[str, Any]) -> None:
        """Store a new active group. Raises FileExistsError when the directory holds that group id."""
        path = self._path(group_id)
        if path.exists():
            raise FileExistsError(path)
        self._write(
            group_id,
            {"document": document, "cancel_requested": False, "terminal": False, "created_at": time.time()},
        )

    def active(self) -> list[str]:
        """The ids of the groups that are not terminal, oldest first."""
        rows = [(json.loads(path.read_text()), path.stem) for path in self._directory.glob("*.json")]
        return [
            group_id
            for row, group_id in sorted(rows, key=lambda item: item[0]["created_at"])
            if not row["terminal"]
        ]

    def get(self, group_id: str) -> dict[str, Any]:
        """The group's ``document``, ``cancel_requested`` and ``terminal``. Raises KeyError for no group."""
        path = self._path(group_id)
        if not path.is_file():
            raise KeyError(group_id)
        row: dict[str, Any] = json.loads(path.read_text())
        return row

    def save(self, group_id: str, document: dict[str, Any], *, terminal: bool = False) -> None:
        """Replace the group's document. A terminal group is no longer active."""
        row = self.get(group_id)
        row.update(document=document, terminal=terminal)
        self._write(group_id, row)

    def release(self, group_id: str) -> None:
        """Record a request to cancel the group."""
        row = self.get(group_id)
        row["cancel_requested"] = True
        self._write(group_id, row)

    def _path(self, group_id: str) -> Path:
        return self._directory / f"{group_id}.json"

    def _write(self, group_id: str, row: dict[str, Any]) -> None:
        # A reader, ops/report among them, never sees half a row.
        self._directory.mkdir(parents=True, exist_ok=True)
        temporary = self._path(group_id).with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(self._path(group_id))


class Groups:
    """The group requests of one run: submitted through the coordinator and followed until each is ready.

    ``pause`` runs between two passes over the groups that are not ready: a wait, while a worker runs
    the jobs. The groups live in ``directory``, and each job's idempotency key has ``submitter``'s prefix.
    A group whose job lost its data with its host is ready at once, and failed. Nothing runs it again."""

    def __init__(
        self, *, client: RunStoreClient, directory: Path, submitter: Submitter, pause: Callable[[], None]
    ) -> None:
        self._store = FileGroups(directory)
        self._context_key = submitter.context_key
        self._pause = pause
        # The evaluation and the gate run a lost search again themselves, under a new key (ADR-0007).
        self._coordinator = Coordinator(
            client=client,
            groups=self._store,
            submitter=submitter,
            stopping=threading.Event(),
            runs_again=False,
        )

    def start(self, request: dict[str, Any], *, affinity: str) -> str:
        """Store the group of a group request, whose jobs carry the key ``affinity``, and return its id.

        The next ``tick`` submits its jobs. Raises ValueError or TypeError for a request that
        ``validate_request`` refuses, and FileExistsError for a group id already in use.
        """
        checked = validate_request(request)
        group_id: str = checked["rollout_job_id"]
        document = initial_document(checked, context_key=self._context_key, affinity=affinity)
        self._store.create(group_id, document)
        return group_id

    def tick(self) -> None:
        """Advance every group that is not ready once (``Coordinator.tick``)."""
        self._coordinator.tick()

    def pause(self) -> None:
        """Wait between two passes over the groups, with the ``pause`` that the constructor got."""
        self._pause()

    def document(self, group_id: str) -> dict[str, Any]:
        """The group's document as the coordinator last saved it."""
        document: dict[str, Any] = self._store.get(group_id)["document"]
        return document

    def settle(self, request: dict[str, Any], *, affinity: str) -> dict[str, Any]:
        """The document of the group that ``request`` makes under the key ``affinity``, once it is ready.

        It blocks until then, with no time limit of its own: each job of the group ends in the Run Store.
        """
        group_id = self.start(request, affinity=affinity)
        while True:
            self.tick()
            document = self.document(group_id)
            if document["ready"]:
                return document
            self._pause()
