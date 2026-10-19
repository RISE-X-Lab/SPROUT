"""The check of a worker's profiles at its start, and the grades that its grade profiles let it make.

``checked_grades`` refuses a profile that would fail each of its jobs at the worker. A grade profile
can name the dataset and the parser of its ``swe-rebench-v2`` grades in ``grade_defaults``, each
pinned by its checksum. The worker claims only the grades of these checksums. docs/RUNSTORE.md
("Checks at the start" and "Grade files") gives the rules.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from sprout.contract.backend import backend_name
from sprout.contract.run_spec import RunSpec
from sprout.runstore.specs import GradeSpec, ReviewSpec
from sprout.worker.attempts import OWNED_BACKENDS
from sprout.worker.jobs.grade import dataset_sha256, parser_revision

#: The job spec whose fields each section of a profile's defaults can set.
_SPECS = {"run_defaults": RunSpec, "grade_defaults": GradeSpec, "review_defaults": ReviewSpec}
#: The sections whose jobs run in a sandbox, on the backend that the section names.
_SANDBOX_SECTIONS = ("run_defaults", "grade_defaults")
#: The ``grade_defaults`` that name the files of a grade. A grade profile names all four or none.
_GRADE_FILES = ("dataset_path", "dataset_sha256", "parser_path", "grader_revision")


def checked_grades(profiles: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """The ``(dataset_sha256, grader_revision)`` pairs of the grade profiles in ``profiles``, sorted.

    Raises ValueError, which names the profile, for a default that its job spec does not have, a sandbox
    backend other than podman or microvm, grade files named in part, or a file without its checksum.
    """
    grades = set()
    for name, profile in sorted(profiles.items()):
        try:
            _check_defaults(profile)
            grade = _grade_files(profile.get("grade_defaults", {}))
        except (OSError, ValueError) as error:
            raise ValueError(f"Profile {name}: {error}") from None
        if grade is not None:
            grades.add(grade)
    return tuple(sorted(grades))


def _check_defaults(profile: Any) -> None:
    if not isinstance(profile, dict):
        raise ValueError("a profile is a JSON object")
    for section, spec in _SPECS.items():
        defaults = profile.get(section, {})
        if not isinstance(defaults, dict):
            raise ValueError(f"{section} is not a JSON object")
        unknown = sorted(set(defaults) - {field.name for field in fields(spec)})
        if unknown:
            raise ValueError(f"{section} sets {', '.join(unknown)}, which {spec.__name__} does not have")
        backend = backend_name(defaults.get("backend") or {})
        if section in _SANDBOX_SECTIONS and backend not in OWNED_BACKENDS:
            raise ValueError(
                f"{section} names the backend {backend!r}, and queued jobs run only on podman or microvm"
            )


def _grade_files(defaults: dict[str, Any]) -> tuple[str, str] | None:
    """The checksums of the dataset and the parser that ``defaults`` names, once both files have them.

    None if ``defaults`` names no grade file.
    """
    named = [defaults.get(key) for key in _GRADE_FILES]
    if not any(named):
        return None
    texts = [value for value in named if isinstance(value, str) and value]
    if len(texts) != len(_GRADE_FILES):
        raise ValueError(
            f"grade_defaults names {', '.join(_GRADE_FILES[:-1])} and {_GRADE_FILES[-1]} "
            "together, or none of them"
        )
    dataset, rows, parser, revision = texts
    if dataset_sha256(Path(dataset).read_bytes()) != rows:
        raise ValueError(f"the SHA-256 of {dataset} is not its dataset_sha256")
    if parser_revision(Path(parser).read_bytes()) != revision:
        raise ValueError(f"the SHA-256 of {parser} is not its grader_revision")
    return rows, revision
