"""``wait_until`` fails at its deadline, so that a test cannot pass on a condition that never held."""

from __future__ import annotations

import pytest

from tests.support.waiting import wait_until


def test_a_condition_that_never_holds_fails_naming_what_was_awaited():
    with pytest.raises(AssertionError, match=r"^the claim did not hold within 0\.0 s"):
        wait_until(lambda: False, timeout_s=0, what="the claim")
