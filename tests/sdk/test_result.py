"""ToolResult parsing (sprout_sandbox/result.py).

`from_response` is the one parser for a tools/call result — a hand-rolled copy
in the backend drifted the moment the shape grew fields. It also unpacks the
runtime's CommandOutcome schema, the shape `shell` answers in when there is more
to say than bare stdout.

Covered:
- envelope: text slot, isError
- a CommandOutcome in the text slot becomes first-class fields
- detection is by shape: a JSON document a *command* printed is not an outcome
- exit_code None vs 0 stay distinguishable
- a plain success carries no outcome fields
"""

from __future__ import annotations

import json

import pytest

from sprout_sandbox.result import ToolResult


def _payload(text: str = "hello", **overrides) -> dict:
    base = {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }
    base.update(overrides)
    return base


def _outcome(**fields) -> str:
    base = {"stdout": "", "stderr": "", "exit_code": 0,
            "stdout_bytes": 0, "stderr_bytes": 0,
            "stdout_truncated": False, "stderr_truncated": False}
    base.update(fields)
    return json.dumps(base)


# --------------------------------------------------------------------------- #
#  Envelope
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("envelope, text", [
    pytest.param(_payload(isError=True), "hello", id="text-slot"),
    pytest.param({"isError": True}, "", id="no-content"),
])
def test_envelope_text_and_error_flag(envelope, text):
    """The text slot and isError are read off the envelope; one with no
    `content` at all is empty output, not an IndexError."""
    result = ToolResult.from_response(envelope)
    assert result.output == text and result.is_error


# --------------------------------------------------------------------------- #
#  CommandOutcome unpacking
# --------------------------------------------------------------------------- #

def test_an_outcome_becomes_first_class_fields():
    text = _outcome(stdout="collected 42\n", stderr="AssertionError\n",
                    exit_code=1, stdout_bytes=13, stderr_bytes=15)
    result = ToolResult.from_response(_payload(text, isError=True))
    assert result.exit_code == 1
    assert result.stdout == "collected 42\n"
    assert result.stderr == "AssertionError\n"          # streams stay separate
    assert result.stderr_bytes == 15
    assert not result.truncated


def test_truncation_and_timeout_are_flags_not_prose():
    text = _outcome(stdout="partial\n", exit_code=137, timed_out=True,
                    stdout_bytes=419430400, stdout_truncated=True)
    result = ToolResult.from_response(_payload(text, isError=True))
    assert result.timed_out and result.truncated
    assert result.stdout_bytes == 419430400


def test_a_running_process_has_no_exit_code_yet():
    """None and 0 are different answers, so `is None` is the right test."""
    text = _outcome(stdout="so far\n", exit_code=None, running=True)
    result = ToolResult.from_response(_payload(text))
    assert result.running and result.exit_code is None

    done = ToolResult.from_response(_payload(_outcome(exit_code=0)))
    assert done.exit_code == 0 and not done.running


# --------------------------------------------------------------------------- #
#  Detection is by shape, not by tool name
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("text", [
    pytest.param("hello_ci\n", id="plain-success"),
    pytest.param("plain output", id="plain-text"),
    pytest.param("", id="empty"),
    pytest.param("{not json", id="malformed-json"),
    pytest.param("[1,2,3]", id="json-array"),
    pytest.param('{"pid":"a1b2c3"}', id="no-outcome-fields"),
    pytest.param(json.dumps({"exit_code": 0}), id="valid-field-no-streams"),
    pytest.param(json.dumps({"stdout": "x", "exit_code": 0}), id="one-stream"),
    pytest.param(json.dumps({"stdout": "x", "stderr": "y", "name": "my-package"}),
                 id="both-streams-foreign-key"),
])
def test_not_an_outcome_passes_through(text):
    """Whatever is not a CommandOutcome is passed through as `output`, with the
    outcome fields left unset.

    `echo hello` has nothing to report beyond its stdout, so the runtime sends
    bare text (plain-success). The rest is what a *command* printed: `cat
    package.json` returns JSON as its output, and treating that as the runtime's
    own report would silently replace the model's view of it. An outcome always
    reports both streams and nothing but outcome fields, so a payload borrowing
    a field name is not one — accepting it would fabricate empty streams for a
    command whose real output was that JSON."""
    result = ToolResult.from_response(_payload(text))
    assert result.stdout is None and result.exit_code is None   # not unpacked
    assert result.output == text                                # passed through


def test_every_backend_parses_through_from_response():
    """The backend once hand-rolled this parse and drifted when the shape grew
    fields — the outcome silently stopped arriving while every unit test passed,
    because they construct ToolResult directly. Grep is the cheap guard: one
    from_response per backend class, however many there are."""
    import re
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "src" /
              "sprout_sandbox" / "backends.py").read_text()
    backends = re.findall(r"^class (\w+Backend)\b", source, re.M)
    assert backends, "no concrete backend found"
    assert source.count("ToolResult.from_response") == len(backends), backends
    assert 'result["content"][0]["text"]' not in source, \
        "a backend is parsing the envelope by hand again"
