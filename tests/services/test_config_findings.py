"""What the doctor and the setup find in a runstore.json that makes jobs fail, and the fix of each problem.

Each case changes one value of the file that ``init`` writes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sprout.services.setup.findings import Finding, host_findings
from sprout.services.setup.templates import HostValues, json_text, runstore_config

#: A shell that sets the key variable of the template's profiles.
KEYED = {"SPROUT_MODEL_API_KEY": "set"}


@pytest.fixture
def config(tmp_path: Path) -> dict[str, Any]:
    """The runstore.json that ``init`` writes on a host whose files are under ``tmp_path``."""
    host = HostValues(
        artifact_root=str(tmp_path / "artifacts"),
        runtime_bin=str(tmp_path / "sprout-runtime"),
        python=str(tmp_path / "python"),
        store=str(tmp_path / "store"),
    )
    return runstore_config(host)


def summary(findings: list[Finding]) -> list[tuple[str, str, str]]:
    return [(finding.status, finding.problem, finding.fix) for finding in findings]


def test_the_runstore_json_that_init_writes_has_no_finding(config):
    assert host_findings(runstore=json_text(config), environ=KEYED) == []


def test_an_example_path_of_the_template_fails_with_the_profiles_that_hold_it(config):
    config["profiles"]["mini"]["run_defaults"]["backend"]["podman"]["store"] = "/path/to/sprout-btrfs"
    config["profiles"]["grade"]["grade_defaults"]["backend"]["podman"]["store"] = "/path/to/sprout-btrfs"

    findings = host_findings(runstore=json_text(config), environ=KEYED)

    assert summary(findings) == [
        (
            "fail",
            "`store` is the example path /path/to/sprout-btrfs of the template, in profiles grade, mini",
            "set it to a path on this host",
        )
    ]


def test_a_worker_env_variable_unset_in_this_shell_warns_with_its_name_and_its_profiles(config):
    """The worker resolves ``worker_env`` for each job, and an unset variable raises KeyError."""
    findings = host_findings(runstore=json_text(config), environ={})

    assert summary(findings) == [
        (
            "warn",
            "SPROUT_MODEL_API_KEY is not set in this shell, and the worker_env of profiles mini, "
            "mini-microvm, review names it: each job of these profiles fails with KeyError unless the "
            "shell of the worker sets it",
            "export SPROUT_MODEL_API_KEY in the shell that starts the worker, or remove the reference "
            "from runstore.json",
        )
    ]


def test_a_runstore_json_that_is_not_json_fails():
    findings = host_findings(runstore="{", environ=KEYED)

    assert summary(findings) == [
        (
            "fail",
            "runstore.json is not valid JSON: Expecting property name enclosed in double quotes: "
            "line 1 column 2 (char 1)",
            "fix the file, or move it away and run `python -m sprout.services init`",
        )
    ]
