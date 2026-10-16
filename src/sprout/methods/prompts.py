"""The prompts ``sprout.methods`` sends, loaded from one YAML file.

``resources/methods/prompts.yaml`` is the default copy; ``methods.evaluate
--prompts PATH`` loads another, so a prompt can be edited without touching
code. Every key is required, every value is a ``str.format`` template (a
literal brace doubled), and the loader checks that each template names
exactly the placeholders ``REQUIRED_PLACEHOLDERS`` lists for it -- an edited
template that dropped ``{reports}`` or added ``{hint}`` would otherwise fail
at the first reviewer request, or silently send a prompt with a field missing.
The file's sha256 travels with the loaded prompts so a run can record which
prompts it used (``summary.json``, ``plan-roundN.json``, the runner's manifest).
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from string import Formatter
from typing import Optional

import yaml

from sprout.resource import resource

DEFAULT_PROMPTS_PATH = resource("methods/prompts.yaml")

#: Dotted key in the YAML -> the placeholders its template must have, exactly.
#: None marks a text that is sent verbatim, never through ``str.format``: its
#: braces are literal (shepherd.system shows the JSON object it wants back), so
#: it is not parsed as a template.
REQUIRED_PLACEHOLDERS: dict[str, Optional[frozenset[str]]] = {
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
    """One attribute per key of the file (``analyst.case`` -> ``analyst_case``),
    plus where it came from and the sha256 of its bytes."""

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


def _lookup(data: dict, key: str, path: Path) -> str:
    node = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"{path}: missing prompt {key}")
        node = node[part]
    if not isinstance(node, str) or not node.strip():
        raise ValueError(f"{path}: prompt {key} must be a non-empty string, not {type(node).__name__}")
    return node


def _leaves(node, prefix: str = "") -> list[str]:
    if not isinstance(node, dict):
        return [prefix]
    return [
        leaf
        for name, child in node.items()
        for leaf in _leaves(child, f"{prefix}.{name}" if prefix else str(name))
    ]


def load_prompts(path: Path | None = None) -> Prompts:
    """Read and validate a prompts file (the packaged one when ``path`` is None).

    Raises ``ValueError`` naming the key for a missing or non-string prompt, a
    template that does not parse (a single brace where a literal one was meant),
    a template whose placeholders differ from ``REQUIRED_PLACEHOLDERS`` (the
    missing and the unexpected ones are listed), or a key the code never reads
    (a misspelled override would otherwise be a silent no-op); ``OSError`` when
    the file cannot be read.
    """
    path = Path(path) if path is not None else DEFAULT_PROMPTS_PATH
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"{path}: not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: the prompts file must be a mapping, not {type(data).__name__}")
    unknown = sorted(set(_leaves(data)) - set(REQUIRED_PLACEHOLDERS))
    if unknown:
        raise ValueError(
            f"{path}: unknown prompt keys {unknown}; the known ones are {sorted(REQUIRED_PLACEHOLDERS)}"
        )
    values = {}
    for key, required in REQUIRED_PLACEHOLDERS.items():
        text = _lookup(data, key, path)
        if required is not None:
            try:
                found = placeholders(text)
            except ValueError as error:
                # Formatter's own message ("Single '}' encountered in format
                # string") names neither the file nor the key; a stray single
                # brace is the likeliest slip in a file whose literal braces
                # are doubled.
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
