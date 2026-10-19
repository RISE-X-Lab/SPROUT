"""The rules by which a worker derives what an attempt runs from its job: backend and budget."""

from __future__ import annotations

import pytest

from sprout.worker.attempts import apply_branch_budget, owned_backend


@pytest.mark.parametrize(
    "effective, backend",
    [
        ({"sandbox_image": "image"}, "podman"),
        ({"sandbox_image": "image", "backend": {"backend": "microvm"}}, "microvm"),
    ],
    ids=["no backend named", "microvm"],
)
def test_a_queued_rollout_runs_on_podman_unless_it_names_microvm(effective, backend):
    assert owned_backend("rollout", effective) == backend


def test_a_branch_budget_replaces_only_the_limits_that_it_names_in_the_inherited_contract():
    effective = {
        "prompt": "fix the failing test",
        "extra": {"rollout_contract": {"max_turns": 50, "deadline_at": 1700000000.0, "message_export": True}},
    }

    bounded = apply_branch_budget(effective, {"max_turns": 10})

    assert bounded == {
        "prompt": "fix the failing test",
        "extra": {"rollout_contract": {"max_turns": 10, "deadline_at": 1700000000.0, "message_export": True}},
    }


def test_a_branch_budget_without_a_contract_to_bound_is_refused():
    with pytest.raises(ValueError, match="^A branch budget bounds a rollout contract; this run has none$"):
        apply_branch_budget({"prompt": "fix the failing test"}, {"max_turns": 10})
