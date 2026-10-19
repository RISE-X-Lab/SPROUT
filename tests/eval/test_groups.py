"""``FileGroups``: the group store of a gate or an evaluation, one JSON file per group in its output.

The search coordinator relies on it as a ``GroupStore`` (``sprout.search.coordinator``).
"""

import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from sprout.eval import groups
from sprout.eval.groups import FileGroups


def document(name: str) -> dict[str, object]:
    """A group document, as ``sprout.search.group.initial_document`` makes one."""
    return {"rollout_job_id": name, "status": "queued", "ready": False, "samples": []}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each group that a test creates is one second younger than the one before it."""
    ticks = itertools.count(1_000_000)
    monkeypatch.setattr(groups, "time", SimpleNamespace(time=lambda: float(next(ticks))))


def test_the_active_groups_are_the_ones_not_terminal_oldest_first(tmp_path: Path, clock: None) -> None:
    """The coordinator advances the groups in this order, so the order is by age and not by name."""
    store = FileGroups(tmp_path / "groups")
    for name in ("round-2", "parent", "review-1"):
        store.create(name, document(name))

    store.save("parent", document("parent"), terminal=True)

    assert store.active() == ["round-2", "review-1"]


def test_a_group_id_in_use_cannot_be_created_again(tmp_path: Path) -> None:
    """A second evaluation in one output must not take over the jobs of the first."""
    store = FileGroups(tmp_path / "groups")
    store.create("parent", document("parent"))

    with pytest.raises(FileExistsError):
        store.create("parent", {**document("parent"), "status": "running"})

    assert store.get("parent")["document"] == document("parent")


def test_an_unknown_group_is_a_key_error(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        FileGroups(tmp_path / "groups").get("parent")


def test_a_cancel_request_outlasts_each_later_save_of_the_document(tmp_path: Path) -> None:
    """The coordinator saves the document after each change of a stage. A save must not drop the request."""
    store = FileGroups(tmp_path / "groups")
    store.create("parent", document("parent"))
    store.release("parent")

    store.save("parent", {**document("parent"), "status": "running"})

    row = store.get("parent")
    assert (row["cancel_requested"], row["terminal"], row["document"]["status"]) == (True, False, "running")
