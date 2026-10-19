"""The check of a worker's profiles at its start, and the grades that its grade profiles let it make.

A profile that fails the check would fail each of its jobs at the worker, so ``start`` refuses it. A grade
profile names the dataset and the parser of its ``swe-rebench-v2`` grades. docs/RUNSTORE.md ("Config")
gives the rules.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from sprout.runstore.config import read_config
from sprout.worker.profiles import checked_grades
from tests.support.paths import REPO


def grade_files(
    directory: Path, *, rows: str = '{"instance_id": "a"}\n', parser: str = "NAME_TO_PARSER = {}\n"
):
    """The ``grade_defaults`` that name a dataset of ``rows`` and a parser of ``parser`` in ``directory``."""
    dataset, parsers = directory / "tasks.jsonl", directory / "log_parsers.py"
    dataset.write_text(rows)
    parsers.write_text(parser)
    return {
        "dataset_path": str(dataset),
        "dataset_sha256": hashlib.sha256(rows.encode()).hexdigest(),
        "parser_path": str(parsers),
        "grader_revision": "sha256:" + hashlib.sha256(parser.encode()).hexdigest(),
    }


def test_each_grade_profile_that_names_its_dataset_and_parser_gives_their_checksums(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, second = grade_files(tmp_path / "a"), grade_files(tmp_path / "b", rows='{"instance_id": "b"}\n')
    profiles = {
        "rebench-b": {"grade_defaults": second},
        "rebench-a": {"grade_defaults": {**first, "timeout_s": 600}},
        "mini": {"run_defaults": {"slot": "mini-swe-agent"}},
    }

    grades = checked_grades(profiles)

    assert grades == tuple(
        sorted(
            [
                (first["dataset_sha256"], first["grader_revision"]),
                (second["dataset_sha256"], second["grader_revision"]),
            ]
        )
    )


def test_the_example_config_passes_the_check_and_names_no_grades():
    """``python -m sprout.services init`` writes the example as the config of a new host."""
    profiles = read_config(REPO / "resources" / "runstore" / "config.example.json")["profiles"]

    assert checked_grades(profiles) == ()


def changed(defaults: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """``defaults`` with ``changes`` over them. A change to None removes the key."""
    merged = {**defaults, **changes}
    return {key: value for key, value in merged.items() if value is not None}


@pytest.mark.parametrize(
    "change, message",
    [
        pytest.param(
            {"dataset_sha256": "0" * 64},
            "^Profile grade: the SHA-256 of .*tasks.jsonl is not its dataset_sha256$",
            id="dataset-checksum",
        ),
        pytest.param(
            {"grader_revision": "sha256:" + "0" * 64},
            "^Profile grade: the SHA-256 of .*log_parsers.py is not its grader_revision$",
            id="parser-revision",
        ),
        pytest.param(
            {"dataset_sha256": None},
            "^Profile grade: grade_defaults names dataset_path, dataset_sha256, parser_path and "
            "grader_revision together, or none of them$",
            id="dataset-without-checksum",
        ),
        pytest.param(
            {"dataset_path": None, "dataset_sha256": None},
            "^Profile grade: grade_defaults names dataset_path, dataset_sha256, parser_path and "
            "grader_revision together, or none of them$",
            id="parser-without-dataset",
        ),
        pytest.param(
            {"dataset_path": "/path/to/missing.jsonl"},
            "^Profile grade: .*No such file or directory: '/path/to/missing.jsonl'$",
            id="missing-dataset",
        ),
    ],
)
def test_a_grade_profile_whose_files_fail_their_checksums_is_refused(tmp_path, change, message):
    profiles = {"grade": {"grade_defaults": changed(grade_files(tmp_path), **change)}}

    with pytest.raises(ValueError, match=message):
        checked_grades(profiles)


@pytest.mark.parametrize(
    "profile, message",
    [
        pytest.param(
            {"run_defaults": {"slot": "mini-swe-agent", "flavour": "x"}},
            "^Profile mini: run_defaults sets flavour, which RunSpec does not have$",
            id="run-default",
        ),
        pytest.param(
            {"grade_defaults": {"slot": "mini-swe-agent"}},
            "^Profile mini: grade_defaults sets slot, which GradeSpec does not have$",
            id="grade-default",
        ),
        pytest.param(
            {"review_defaults": {"backend": {}}},
            "^Profile mini: review_defaults sets backend, which ReviewSpec does not have$",
            id="review-default",
        ),
        pytest.param(
            {"run_defaults": {"backend": {"backend": "local"}}},
            "^Profile mini: run_defaults names the backend 'local', and queued jobs run only on podman "
            "or microvm$",
            id="run-backend",
        ),
        pytest.param(
            {"grade_defaults": {"backend": {"backend": "docker"}}},
            "^Profile mini: grade_defaults names the backend 'docker', and queued jobs run only on "
            "podman or microvm$",
            id="grade-backend",
        ),
        pytest.param(
            {"run_defaults": ["slot"]},
            "^Profile mini: run_defaults is not a JSON object$",
            id="not-an-object",
        ),
    ],
)
def test_a_profile_whose_defaults_would_fail_each_of_its_jobs_is_refused(profile, message):
    with pytest.raises(ValueError, match=message):
        checked_grades({"mini": profile})
