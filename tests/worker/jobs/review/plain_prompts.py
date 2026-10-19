"""A prompts file of short templates, so that a test can state the whole prompt that a request sends.

Each template names its placeholders in order after a label: ``REPORTS {reports}``, for example. The
packaged prompts change with the research. These templates change only with the placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS_PATH, Prompts, load_prompts

ANALYST = (
    "PROBLEM {problem}\nVERDICT {verdict}\nTRANSCRIPT {transcript}\n"
    "STEPS {checkpoint_steps} OF {lo}..{hi} AT MOST {candidate_limit}"
)
FEEDBACK = "\nREJECTED {error}"
REVIEWERS = {
    "point": "POINTS {problem}\nREPORTS {reports}\nCOUNT {count_rule}",
    "user_hint": "HINTS {problem}\nREPORTS {reports}\nCOUNT {count_rule}",
    "assistant_turn": "TURNS {problem}\nREPORTS {reports}\nCOUNT {count_rule}",
}
FORMAT_HINT = "FORMAT HINT."
JSON_DIRECTIVE = "\nJSON ONLY."


def write_plain_prompts(path: Path) -> Path:
    """The packaged prompts file at ``path``, with every analyst, reviewer and transport template short."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["analyst"] = {"case": ANALYST, "feedback": FEEDBACK}
    data["reviewer"] = {
        **REVIEWERS,
        "assistant_format_hint": FORMAT_HINT,
        "count_rule": {"adaptive": "AT MOST {count}", "fixed": "EXACTLY {count}"},
    }
    data["transport"] = {"json_directive": JSON_DIRECTIVE}
    path.write_text(yaml.safe_dump(data))
    return path


def plain_prompts(path: Path) -> Prompts:
    return load_prompts(write_plain_prompts(path))


def prompts_record(directory: Path) -> dict[str, str]:
    """The ``prompts`` of a review spec that names the short templates, in ``directory/prompts.yaml``."""
    prompts = plain_prompts(directory / "prompts.yaml")
    return {"path": str(prompts.path), "sha256": prompts.sha256}


def reports_in(prompt: str) -> list[dict[str, Any]]:
    """The reviewer rows that a reviewer request built from these templates sends."""
    return json.loads(prompt.split("\nREPORTS ", 1)[1].rsplit("\nCOUNT ", 1)[0])
