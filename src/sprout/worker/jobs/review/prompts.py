"""The prompts of SPROUT's analyst, reviewer and Shepherd selector, loaded from one YAML file.

The packaged ``resources/review/prompts.yaml`` is the default, and ``eval.evaluate --prompts PATH``
names another, so a prompt can change without a code change. Every key is required. Each value
is a ``str.format`` template (a literal brace doubled), or a verbatim text where
``REQUIRED_PLACEHOLDERS`` gives None. ``load_prompts`` checks that each template has exactly its
required placeholders. So an edited template that dropped ``{reports}`` fails when it is loaded, not
at the first reviewer request. ``Prompts.sha256`` lets a run record which prompts it used.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from string import Formatter

import yaml

from sprout.resource import resource

DEFAULT_PROMPTS_PATH = resource("review/prompts.yaml")

#: Dotted key in the YAML -> the placeholders that its template must have, exactly. None marks
#: a text sent verbatim, never through ``str.format``, so that its braces are literal.
REQUIRED_PLACEHOLDERS: dict[str, frozenset[str] | None] = {
    "analyst.case": frozenset(
        {"problem", "verdict", "transcript", "checkpoint_steps", "lo", "hi", "candidate_limit"}
    ),
    "analyst.feedback": frozenset({"error"}),
    "reviewer.assistant_turn": frozenset({"problem", "reports", "count_rule"}),
    "reviewer.assistant_format_hint": frozenset(),
    "reviewer.point": frozenset({"problem", "reports", "count_rule"}),
    "reviewer.user_hint": frozenset({"problem", "reports", "count_rule"}),
    "reviewer.count_rule.adaptive": frozenset({"count"}),
    "reviewer.count_rule.fixed": frozenset({"count"}),
    "shepherd.system": None,
    "transport.json_directive": frozenset(),
}


@dataclass(frozen=True)
class Prompts:
    """The prompts of one file, one attribute per key (``analyst.case`` becomes ``analyst_case``).

    ``path`` is the file, and ``sha256`` is the digest of its bytes."""

    analyst_case: str
    analyst_feedback: str
    reviewer_assistant_turn: str
    reviewer_assistant_format_hint: str
    reviewer_point: str
    reviewer_user_hint: str
    reviewer_count_rule_adaptive: str
    reviewer_count_rule_fixed: str
    shepherd_system: str
    transport_json_directive: str
    path: Path
    sha256: str


def placeholders(template: str) -> set[str]:
    """The named fields ``str.format`` would fill in ``template``."""
    return {field for _, field, _, _ in Formatter().parse(template) if field is not None}


def _lookup(data: dict[str, object], key: str, path: Path) -> str:
    node: object = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{path}: missing prompt {key}")
        node = node[part]
    if not isinstance(node, str) or not node.strip():
        raise ValueError(f"{path}: prompt {key} must be a non-empty string, not {type(node).__name__}")
    return node


def _leaves(node: object, prefix: str = "") -> list[str]:
    if not isinstance(node, dict):
        return [prefix]
    return [
        leaf
        for name, child in node.items()
        for leaf in _leaves(child, f"{prefix}.{name}" if prefix else str(name))
    ]


def load_prompts(path: Path | None = None) -> Prompts:
    """Read and validate a prompts file, the packaged one when ``path`` is None.

    Raises ``OSError`` if the file cannot be read. Raises ``ValueError``, naming the key, for a missing
    or non-string prompt, a template that does not parse, a template with the wrong placeholders, or a
    key that no code reads."""
    path = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the prompts file must be a mapping, not {type(data).__name__}")
    # A misspelled key in an edited file would otherwise change nothing, and nobody would notice.
    unknown = sorted(set(_leaves(data)) - set(REQUIRED_PLACEHOLDERS))
    if unknown:
        raise ValueError(
            f"{path}: unknown prompt keys {unknown}; the known ones are {sorted(REQUIRED_PLACEHOLDERS)}"
        )
    values: dict[str, str] = {}
    for key, required in REQUIRED_PLACEHOLDERS.items():
        text = _lookup(data, key, path)
        if required is not None:
            try:
                found = placeholders(text)
            except ValueError as error:
                # Formatter's message ("Single '}' encountered in format string") names no file or key,
                # and a stray single brace is the likeliest slip in a file whose literal braces are doubled.
                raise ValueError(
                    f"{path}: prompt {key} is not a valid str.format template: {error}"
                ) from error
            if found != required:
                raise ValueError(
                    f"{path}: prompt {key} has the wrong placeholders: missing {sorted(required - found)}, "
                    f"unexpected {sorted(found - required)}; it needs exactly {sorted(required)}"
                )
        values[key.replace(".", "_")] = text
    return Prompts(**values, path=path, sha256=hashlib.sha256(raw).hexdigest())


DEFAULT_PROMPTS = load_prompts()
