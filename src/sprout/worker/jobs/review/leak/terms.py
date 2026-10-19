"""Which hidden test names a reviewer-written turn gives away: its ``leak_terms``.

The analyst reads the grade, and the reviewer reads the analyst's report, so a repair turn can
name a hidden test that the actor never saw. If the history the actor kept at that point (h_s)
never showed the name, a policy trained on the turn learns to name what it could not have seen.
This module only finds such names. The trainer decides whether a candidate may still be credited.

The check is lexical: a term counts as named when it occurs in the text as a substring.
``hidden_terms`` takes the vocabulary from two facts that the grade records:

- each test_patch path of at least ``MIN_TERM`` characters, and its basename when that is as
  long and not in ``GENERIC_BASENAMES``
- each FAIL_TO_PASS name of at least ``MIN_IDENTIFIER`` characters, and each ``[A-Za-z0-9_]+``
  token of it that long and not all digits. A test id is rarely retyped whole, but its
  function name is.

PASS_TO_PASS names are no terms: those tests already pass in the repository that the actor reads.
Every function is pure and uses only the standard library. See the Review section of ``docs/RUNSTORE.md``.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterable
from typing import Any

MIN_TERM = 6
MIN_IDENTIFIER = 8
#: Names any repository may hold. Naming one reveals nothing about the hidden tests.
GENERIC_BASENAMES = frozenset(
    {
        "__init__.py",
        "conftest.py",
        "setup.py",
        "setup.cfg",
        "pytest.ini",
        "tox.ini",
        "pyproject.toml",
        "package.json",
        "Cargo.toml",
        "go.mod",
        "Makefile",
        "test.py",
        "tests.py",
        "utils.py",
        "helpers.py",
        "common.py",
        "main.py",
        "index.js",
        "index.ts",
        "mod.rs",
        "lib.rs",
    }
)
_SEPARATORS = re.compile(r"[^A-Za-z0-9_]+")


def _names(value: object, key: str) -> list[str]:
    # Fail closed: a list arriving as a JSON string would iterate as single
    # characters, none of them a term, and every turn would look clean.
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def hidden_terms(*, fail_to_pass: object, test_patch_paths: object) -> frozenset[str]:
    """The hidden vocabulary of a grade's two facts, as the module docstring describes it.

    Each fact is None or a list of strings. Raises ``ValueError`` for any other value."""
    terms = set()
    for path in _names(test_patch_paths, "test_patch_paths"):
        if len(path) >= MIN_TERM and path not in GENERIC_BASENAMES:
            terms.add(path)
        base = posixpath.basename(path)
        if base != path and len(base) >= MIN_TERM and base not in GENERIC_BASENAMES:
            terms.add(base)
    for name in _names(fail_to_pass, "fail_to_pass"):
        name = name.strip()
        if len(name) >= MIN_IDENTIFIER:
            terms.add(name)
        terms.update(
            token for token in _SEPARATORS.split(name) if len(token) >= MIN_IDENTIFIER and not token.isdigit()
        )
    return frozenset(terms)


def hidden_terms_of(grade: dict[str, Any]) -> frozenset[str] | None:
    """The grade's hidden vocabulary, or None when the grade has neither fact."""
    fail_to_pass, test_patch_paths = grade.get("fail_to_pass"), grade.get("test_patch_paths")
    if fail_to_pass is None and test_patch_paths is None:
        return None
    return hidden_terms(fail_to_pass=fail_to_pass, test_patch_paths=test_patch_paths)


def _text(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def turn_text(turn: dict[str, Any]) -> str:
    """The text that one message shows: its content, its reasoning and its tool calls.

    Each call gives its raw arguments and then the command they decode to. A term that JSON escapes
    in the raw arguments (a quote or a non-ASCII character) still matches in the decoded command."""
    parts = [_text(turn.get("content")), _text(turn.get("reasoning_content"))]
    for call in turn.get("tool_calls") or ():
        arguments = (call.get("function") or {}).get("arguments")
        parts.append(_text(arguments))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:  # a call the schema check rejected: the raw text is all it shows
                continue
        if isinstance(arguments, dict) and isinstance(arguments.get("command"), str):
            parts.append(arguments["command"])
    return "\n".join(part for part in parts if part)


def _observation(content: object) -> list[str]:
    # mini renders an observation as JSON through Jinja's tojson, which escapes quotes, <, >, & and
    # non-ASCII text. A test id that the actor's run printed is then whole only in the decoded strings.
    try:
        document = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return []
    return (
        [value for value in document.values() if isinstance(value, str)] if isinstance(document, dict) else []
    )


def visible_text(messages: Iterable[dict[str, Any]]) -> str:
    """The ``turn_text`` of every message in a history, joined into one text.

    A tool output is its tool message's content. The text has that content as
    sent and, when it is a JSON document, its decoded strings as well.
    """
    parts = []
    for message in messages:
        parts.append(turn_text(message))
        if message.get("role") == "tool":
            parts.extend(_observation(message.get("content")))
    return "\n".join(part for part in parts if part)


def leak_terms(turn: dict[str, Any], hidden: Iterable[str], visible: str) -> list[str]:
    """The hidden terms ``turn`` names that ``visible`` (h_s) does not show, sorted.

    The result is ``[]`` for a clean turn and for an empty vocabulary.
    """
    text = turn_text(turn)
    return sorted(term for term in hidden if term in text and term not in visible)
