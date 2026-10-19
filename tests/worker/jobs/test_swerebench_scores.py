"""The scores of a SWE-rebench grade, read from the status of each test that the log parser found.

``resolved`` is the verdict, ``progress`` the share of FAIL_TO_PASS tests fixed, and
``expected_pass_share`` the share of all expected tests that passed. The RL driver reports all three.
"""

from __future__ import annotations

import pytest

from sprout.worker.jobs.grade import expected_pass_share, progress, resolved

TWO_TARGETS = {"FAIL_TO_PASS": ["t::f1", "t::f2"], "PASS_TO_PASS": ["t::p1", "t::p2"]}
ONLY_KEPT = {"FAIL_TO_PASS": [], "PASS_TO_PASS": ["t::p1", "t::p2"]}
NO_TESTS = {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}


@pytest.mark.parametrize(
    "statuses, task, scores",
    [
        pytest.param(
            {"t::f1": "PASSED", "t::f2": "PASSED", "t::p1": "PASSED", "t::p2": "PASSED"},
            TWO_TARGETS,
            (True, 1.0, 1.0),
            id="every-expected-test-passed",
        ),
        pytest.param(
            {"t::f1": "PASSED", "t::f2": "FAILED", "t::p1": "PASSED", "t::p2": "PASSED"},
            TWO_TARGETS,
            (False, 0.5, 0.75),
            id="one-of-two-targets-fixed",
        ),
        pytest.param(
            {"t::f1": "PASSED", "t::f2": "PASSED", "t::p1": "PASSED", "t::p2": "FAILED"},
            TWO_TARGETS,
            (False, 0.0, 0.75),
            id="targets-fixed-but-a-kept-test-broke",
        ),
        pytest.param(
            {"t::f1": "PASSED", "t::p1": "PASSED", "t::p2": "PASSED"},
            TWO_TARGETS,
            (False, 0.5, 0.75),
            id="a-target-that-the-log-lacks-did-not-pass",
        ),
        pytest.param(
            {"t::f1": "PASSED", "t::f2": "SKIPPED", "t::p1": "PASSED", "t::p2": "ERROR"},
            TWO_TARGETS,
            (False, 0.0, 0.5),
            id="only-PASSED-is-a-pass",
        ),
        pytest.param(
            {"t::p1": "PASSED", "t::p2": "PASSED"},
            ONLY_KEPT,
            (True, 1.0, 1.0),
            id="no-targets-and-every-kept-test-passed",
        ),
        pytest.param(
            {"t::p1": "PASSED", "t::p2": "FAILED"},
            ONLY_KEPT,
            (False, 0.0, 0.5),
            id="no-targets-and-a-kept-test-broke",
        ),
        pytest.param({"t::p1": "PASSED"}, NO_TESTS, (False, 0.0, 0.0), id="no-expected-tests"),
    ],
)
def test_the_scores_count_the_expected_tests_that_passed(statuses, task, scores):
    """Without targets, progress is the verdict. A broken PASS_TO_PASS test makes progress 0."""
    assert (resolved(statuses, task), progress(statuses, task), expected_pass_share(statuses, task)) == scores


# A test runner can append a duration to a test's name, in one of three forms. The log of a
# run and the dataset can disagree about it, on either side.
@pytest.mark.parametrize(
    "logged, expected",
    [
        ("suite renders [12 ms]", "suite renders"),
        ("suite renders [ 1.5 s ]", "suite renders"),
        ("suite renders in 0.5 sec", "suite renders"),
        ("suite renders in 12 msec", "suite renders"),
        ("suite renders (3 ms)", "suite renders"),
        ("suite renders (3MS)", "suite renders"),
        ("suite renders", "suite renders [40 ms]"),
    ],
)
def test_a_test_name_matches_without_the_duration_that_a_runner_appends(logged, expected):
    assert resolved({logged: "PASSED"}, {"FAIL_TO_PASS": [expected]}) is True


def test_a_name_that_differs_beyond_its_duration_is_another_test():
    assert resolved({"suite renders [12 ms]": "PASSED"}, {"FAIL_TO_PASS": ["suite updates"]}) is False
