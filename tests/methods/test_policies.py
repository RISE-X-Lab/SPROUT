"""The exact selectors (eval.policies) refuse invalid input and never invent entropy."""

import pytest

from sprout.eval.policies import bpo_select, shepherd_select, token_entropy


def test_exact_selectors_reject_invalid_and_do_not_invent_entropy():
    with pytest.raises(ValueError):
        token_entropy({})
    with pytest.raises(ValueError):
        shepherd_select({"checkpoint_step": 6, "reason": "test"}, [1, 2], 7)
    selected = bpo_select([{"step": 1, "token_position": 10, "entropy_lower_bound_nats": 1}], 7)
    assert len(selected) == 7 and {s["step"] for s in selected} == {1}
