"""The example worker config, ``resources/runstore/config.example.json``, which docs/RUNSTORE.md starts from.

A default that its job spec does not have fails each job of the profile at the worker. So does a
backend that the worker does not own, or a sandbox without the runtime binary.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from sprout.contract.run_spec import RunSpec
from sprout.runstore.config import read_config
from sprout.runstore.specs import GradeSpec
from sprout.worker.attempts import owned_backend
from tests.support.paths import REPO

EXAMPLE = REPO / "resources" / "runstore" / "config.example.json"


def test_the_example_has_the_profiles_and_the_default_sections_that_the_tests_below_check():
    """A new profile or a new section of defaults needs a row in the tests below."""
    profiles = read_config(EXAMPLE)["profiles"]

    assert {name: sorted(profile) for name, profile in profiles.items()} == {
        "mini": ["python", "run_defaults", "worker_env"],
        "grade": ["grade_defaults", "python"],
        "mini-microvm": ["python", "run_defaults", "snapshot_repository", "worker_env"],
        "grade-microvm": ["grade_defaults", "python"],
        "review": ["python", "worker_env"],
    }


@pytest.mark.parametrize(
    "profile, section, spec",
    [
        ("mini", "run_defaults", RunSpec),
        ("mini-microvm", "run_defaults", RunSpec),
        ("grade", "grade_defaults", GradeSpec),
        ("grade-microvm", "grade_defaults", GradeSpec),
    ],
)
def test_each_default_of_an_example_profile_is_a_field_of_its_job_spec(profile, section, spec):
    defaults = read_config(EXAMPLE)["profiles"][profile][section]

    assert set(defaults) - {field.name for field in fields(spec)} == set()


@pytest.mark.parametrize(
    "profile, section, backend",
    [
        ("mini", "run_defaults", "podman"),
        ("mini-microvm", "run_defaults", "microvm"),
        ("grade", "grade_defaults", "podman"),
        ("grade-microvm", "grade_defaults", "microvm"),
    ],
)
def test_each_sandbox_profile_of_the_example_runs_on_an_owned_backend_with_the_runtime_binary(
    profile, section, backend
):
    defaults = read_config(EXAMPLE)["profiles"][profile][section]

    name = owned_backend("grade", defaults)

    assert (name, defaults["backend"][name]["runtime_bin"]) == (
        backend,
        "/path/to/sprout/runtime/sprout-runtime",
    )
