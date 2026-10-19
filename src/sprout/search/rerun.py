"""A group that runs again from its roots, after a lost host took the data of one of its jobs.

The Run Store fails each unfinished job of a lost host with the reason ``data_lost``
(ADR-0007). ``lost`` says whether a job of a group document's current run ended so.
``run_again`` gives the document of the group's next run: the current run kept in
``earlier_runs``, the affinity key ``<key>-r<n>`` (``again_key``) for the n-th run again,
and each stage planned again, so that each job gets a new idempotency key. A group runs
again at most ``RUNS_AGAIN`` times, and then fails. ``give_up`` fails the group at once,
for a caller that runs its search again itself. docs/RL_DRIVER.md describes it.
"""

from collections.abc import Iterator
from typing import Any

from sprout.search.group import planned_again
from sprout.search.spec import search_again

#: How many times a search runs again after a lost host (ADR-0007). The next loss fails the search.
RUNS_AGAIN = 2


def again_key(key: str, *, runs: int) -> str:
    """The affinity key of the ``runs``-th run again of a search whose first run had the key ``key``."""
    return f"{key}-r{runs}"


def lost(document: dict[str, Any]) -> bool:
    """Whether a job of the group's current run failed with the reason ``data_lost``."""
    return any(
        stage["state"] == "failed" and (stage.get("result") or {}).get("reason") == "data_lost"
        for stage in _stages(document)
    )


def run_again(document: dict[str, Any]) -> dict[str, Any]:
    """The document of a group whose current run lost a job: the group's next run, or the failed group.

    The next run keeps the current one as the last of ``earlier_runs``, has a new ``affinity``, and has
    only the roots, each stage planned. After ``RUNS_AGAIN`` runs again, the group gets an ``error`` instead.
    """
    earlier = document.get("earlier_runs", [])
    if len(earlier) >= RUNS_AGAIN:
        keys = ", ".join([*(run["affinity"] for run in earlier), document["affinity"]])
        return {
            **document,
            "error": f"A host that held the group's jobs was lost in each of its {len(earlier) + 1} runs:"
            f" {keys}. A group runs again at most {RUNS_AGAIN} times.",
        }
    search = document.get("search")
    slots = [sample["sample_slot_id"] for sample in document["samples"]]
    roots = set(search["root_slots"]) if search else set(slots)
    first = earlier[0]["affinity"] if earlier else document["affinity"]
    again = {
        **document,
        "affinity": again_key(first, runs=len(earlier) + 1),
        "earlier_runs": [*earlier, _run(document)],
        "samples": [
            planned_again(sample) for sample in document["samples"] if sample["sample_slot_id"] in roots
        ],
    }
    if search:
        again["search"] = search_again(search, branch_slots=[slot for slot in slots if slot not in roots])
    return again


def give_up(document: dict[str, Any]) -> dict[str, Any]:
    """The document of a group whose current run lost a job, failed with an ``error``, without a run again."""
    return {
        **document,
        "error": f"A host that held the jobs of the key {document['affinity']} was lost,"
        " and this coordinator runs no group again.",
    }


def _run(document: dict[str, Any]) -> dict[str, Any]:
    """What a group document keeps of a run that it ended: its key, its samples and its search."""
    run = {"affinity": document["affinity"], "samples": document["samples"]}
    if document.get("search"):
        run["search"] = document["search"]
    return run


def _stages(document: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Each stage of the group's current run: each actor and grade, and each review of its search."""
    for sample in document["samples"]:
        yield from (stage for stage in (sample["actor"], sample["grade"]) if stage is not None)
    search = document.get("search")
    if search:
        reviews = (search["review"], *search.get("chain_reviews", []))
        yield from (stage for stage in reviews if stage is not None)
