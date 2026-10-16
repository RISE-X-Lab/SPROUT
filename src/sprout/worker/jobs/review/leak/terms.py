"""Which hidden test names a reviewer-written turn gives away.

A repair turn is written in hindsight. The analyst read the grade, including
the test log and the names of the tests that failed, and the reviewer who
writes the turn read the analyst's report. Those tests are hidden from the
actor. If the turn names one of them, and the history the actor kept at that
point (h_s) never showed the name, a policy trained on the turn learns to
name things it could not have seen. This module only finds such names (a turn's
``leak_terms``). Whether a candidate may still be credited is the trainer's
decision.

The check is lexical. A term counts as named when it occurs in the text as a
substring. The hidden vocabulary comes from the two facts the grade records
(``hidden_terms``):

- each test_patch path of at least ``MIN_TERM`` characters, and its basename
  when the basename is that long too and is not one every repository has
  (``GENERIC_BASENAMES``). A root-level generic file, such as a
  ``conftest.py`` the patch edits, is not a term either.
- each FAIL_TO_PASS name of at least ``MIN_IDENTIFIER`` characters, and each
  ``[A-Za-z0-9_]+`` token of it that long and not all digits. A test id
  (``tests/test_x.py::test_y[1]``) is rarely retyped whole, but its function
  name is.

PASS_TO_PASS names are not terms: those tests already pass in the repository
the actor reads. The module uses the standard library only, and every function
is pure.
"""

from __future__ import annotations

import json
import posixpath
import re
from collections.abc import Iterable

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


def _names(value, key: str) -> list[str]:
    # Fail closed: a list arriving as a JSON string would iterate as single
    # characters, none of them a term, and every turn would look clean.
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be a list of strings")
    return list(value)


def hidden_terms(*, fail_to_pass, test_patch_paths) -> frozenset[str]:
    """The hidden vocabulary of a grade's two facts (see the module docstring)."""
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


def hidden_terms_of(grade: dict) -> frozenset[str] | None:
    """The grade's hidden vocabulary, or None when the grade has neither fact."""
    fail_to_pass, test_patch_paths = grade.get("fail_to_pass"), grade.get("test_patch_paths")
    if fail_to_pass is None and test_patch_paths is None:
        return None
    return hidden_terms(fail_to_pass=fail_to_pass, test_patch_paths=test_patch_paths)


def _text(value) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def turn_text(turn: dict) -> str:
    """The text one message shows: its content, its reasoning, and for each
    tool call the raw arguments followed by the command they decode to.

    Both forms of a call are kept. A term that JSON escapes inside the raw
    arguments (a quote, or a non-ASCII character) still matches in the
    decoded command, and the other way round.
    """
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


def _observation(content) -> list[str]:
    # mini renders an observation as a JSON document ({"returncode": 1, "output":
    # "..."}, through Jinja's tojson). That escapes quotes, <, >, &, ' and
    # non-ASCII characters, so a test id the actor's own run printed whole may
    # not appear whole in the raw text. The decoded strings are what it printed.
    try:
        document = json.loads(content) if isinstance(content, str) else None
    except ValueError:
        return []
    return (
        [value for value in document.values() if isinstance(value, str)] if isinstance(document, dict) else []
    )


def visible_text(messages: Iterable[dict]) -> str:
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


def leak_terms(turn: dict, hidden: Iterable[str], visible: str) -> list[str]:
    """The hidden terms ``turn`` names that ``visible`` (h_s) does not show, sorted.

    The result is ``[]`` for a clean turn and for an empty vocabulary.
    """
    text = turn_text(turn)
    return sorted(term for term in hidden if term in text and term not in visible)
