"""A Qwen3.5 turn as the model sampled it: its raw text, its strict parse, and the text it renders to.

Qwen3.5's chat template opens each assistant turn with ``<|im_start|>assistant\\n<think>\\n``, so the
model writes its reasoning first. A turn that the harness runs is, exactly::

    {reasoning}\\n</think>\\n\\n<tool_call>\\n<function={name}>\\n
    (<parameter={key}>\\n{value}\\n</parameter>\\n)*</function>\\n</tool_call><|im_end|>

``check`` rejects everything else, with the code of the first flaw: the 13 malformed turns of
Arctic-SWE's Table 6 (arXiv 2610.24914) and the harness's rules of its Appendix I.3 (one call,
non-blank reasoning, no prose before the call). The server's tool-call parser never decides what
runs, because it repairs some of these flaws. ``render`` writes the chat template's text of a turn.
A turn that parses but renders to other text still runs, and the caller records the difference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
FUNCTION_OPEN, FUNCTION_CLOSE = "<function=", "</function>"
PARAMETER_OPEN, PARAMETER_CLOSE = "<parameter=", "</parameter>"
#: The special tokens that end a completion. Log-probs include the one that the model sampled.
END_TOKENS = ("<|im_end|>", "<|endoftext|>")
_CALL_TAGS = (CALL_OPEN, CALL_CLOSE, FUNCTION_OPEN, FUNCTION_CLOSE, PARAMETER_OPEN, PARAMETER_CLOSE)
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Call:
    """One function call of a turn: its name and its parameters as written, in order."""

    name: str
    parameters: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Turn:
    """A parsed turn: the reasoning, the prose between the reasoning and the calls, and the calls."""

    reasoning: str
    content: str
    calls: tuple[Call, ...]


class FormatViolation(ValueError):
    """A turn that breaks the format. ``code`` names the flaw, ``message`` tells the model what to fix."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def check(raw: str, *, finish_reason: str | None, require_reasoning: bool = True) -> Turn:
    """The turn that ``raw`` holds, if the harness runs it. Raises ``FormatViolation`` if it does not.

    A response that the token limit cut off is rejected whatever it holds."""
    if finish_reason == "length":
        raise FormatViolation(
            "response_truncated",
            "Your response was cut off at the output token limit before it ended. Reason more briefly, "
            "and end every response with exactly one tool call.",
        )
    turn = parse(raw)
    if not turn.calls:
        raise FormatViolation(
            "missing_tool_call",
            "Your response has no tool call. Every response must end with exactly one tool call; "
            "call submit when the task is done.",
        )
    if len(turn.calls) > 1:
        raise FormatViolation(
            "multiple_calls",
            "Your response has %d tool calls. Make exactly one tool call per response, and wait for "
            "its result before the next." % len(turn.calls),
        )
    if require_reasoning and not turn.reasoning:
        raise FormatViolation(
            "empty_reasoning",
            "Your reasoning block is empty. Think about the next step before the tool call.",
        )
    if turn.content:
        raise FormatViolation(
            "prose_before_call",
            "Your response has text between </think> and the tool call. Put all of your reasoning "
            "inside the reasoning block, and nothing but the tool call after it.",
        )
    return turn


def parse(raw: str) -> Turn:
    """The reasoning, prose and calls of ``raw``, the text after the generation prompt.

    Whitespace around the reasoning, the prose and the call blocks is not checked here: ``render``
    shows whether the turn is canonical. Raises ``FormatViolation`` for every structural flaw."""
    close = raw.find(THINK_CLOSE)
    if close < 0:
        raise FormatViolation(
            "unclosed_reasoning",
            "Your reasoning block was never closed. End your reasoning with </think>, then make "
            "the tool call.",
        )
    reasoning, rest = raw[:close], raw[close + len(THINK_CLOSE) :]
    if THINK_OPEN in reasoning or THINK_OPEN in rest or THINK_CLOSE in rest:
        raise FormatViolation(
            "repeated_think_tag",
            "Your response has more than one <think> or </think> tag. The reasoning block is already "
            "open when you start; close it once with </think>.",
        )
    if any(tag in reasoning for tag in _CALL_TAGS):
        raise FormatViolation(
            "call_in_reasoning",
            "Your tool call is inside the reasoning block, so it did not run. Close the reasoning "
            "with </think> first, then write the tool call.",
        )
    start = rest.find(CALL_OPEN)
    if start < 0:
        if any(tag in rest for tag in _CALL_TAGS):
            raise _envelope("the call is not wrapped in <tool_call> and </tool_call>")
        return Turn(reasoning.strip(), rest.strip(), ())
    return Turn(reasoning.strip(), rest[:start].strip(), _calls(rest, start))


def render(turn: Turn) -> str:
    """The text that the chat template writes for ``turn`` after the generation prompt, without end token."""
    text = turn.reasoning + "\n" + THINK_CLOSE + "\n\n"
    if turn.content:
        text += turn.content + "\n\n"
    for index, call in enumerate(turn.calls):
        text += ("\n" if index else "") + CALL_OPEN + "\n" + FUNCTION_OPEN + call.name + ">\n"
        for key, value in call.parameters:
            text += PARAMETER_OPEN + key + ">\n" + value + "\n" + PARAMETER_CLOSE + "\n"
        text += FUNCTION_CLOSE + "\n" + CALL_CLOSE
    return text


def raw_completion(choice: Mapping[str, Any]) -> str | None:
    """The text that the model sampled in ``choice``, from the bytes of its log-probs, or None without them.

    The bytes of one character can span two tokens, so they are joined before decoding."""
    logprobs = choice.get("logprobs")
    tokens = logprobs.get("content") if isinstance(logprobs, Mapping) else None
    if not isinstance(tokens, list) or not tokens:
        return None
    data = bytearray()
    for token in tokens:
        chunk = token.get("bytes") if isinstance(token, Mapping) else None
        if isinstance(chunk, list):
            data.extend(chunk)
        else:
            data.extend(str(token.get("token", "") if isinstance(token, Mapping) else "").encode())
    text = data.decode("utf-8", errors="replace")
    for end in END_TOKENS:
        if text.endswith(end):
            return text[: -len(end)]
    return text


def structured_completion(message: Mapping[str, Any]) -> str:
    """The canonical text of a response that a server returned already parsed, without log-probs.

    The text is what the chat template renders for the parsed fields, so it always round-trips. Raises
    ``FormatViolation`` for a call whose fields are not a name and an object of arguments."""
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") if isinstance(call, Mapping) else None
        arguments = function.get("arguments") if isinstance(function, Mapping) else None
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            decoded = None
        if not isinstance(function, Mapping) or not isinstance(decoded, Mapping):
            raise _envelope("the call's arguments are not a JSON object")
        calls.append(
            Call(str(function.get("name", "")), tuple((k, _value_text(v)) for k, v in decoded.items()))
        )
    reasoning = message.get("reasoning_content") or ""
    content = message.get("content") or ""
    return render(Turn(str(reasoning).strip(), str(content).strip(), tuple(calls)))


def _calls(rest: str, start: int) -> tuple[Call, ...]:
    """The calls of ``rest`` from the first ``<tool_call>`` at ``start``, up to the end of the response."""
    calls = []
    position = start
    while True:
        end = rest.find(CALL_CLOSE, position)
        if end < 0:
            raise FormatViolation(
                "unclosed_tool_call", "Your <tool_call> block was never closed with </tool_call>."
            )
        block = rest[position + len(CALL_OPEN) : end]
        if CALL_OPEN in block:
            raise FormatViolation(
                "nested_tool_call", "Your response has a <tool_call> block inside another one."
            )
        calls.append(_call(block))
        tail = rest[end + len(CALL_CLOSE) :]
        following = tail.lstrip()
        if not following:
            return tuple(calls)
        if not following.startswith(CALL_OPEN):
            raise FormatViolation(
                "text_after_call",
                "Your response has text after </tool_call>. End the response right after the tool call.",
            )
        position = len(rest) - len(following)


def _call(block: str) -> Call:
    """The one function call of a ``<tool_call>`` block. Raises ``FormatViolation`` for a malformed one."""
    text = block.lstrip()
    if not text.startswith(FUNCTION_OPEN):
        raise FormatViolation(
            "missing_function", "Your <tool_call> block has no <function=NAME> tag naming the tool to call."
        )
    name, position = _tag_name(text, len(FUNCTION_OPEN), "function")
    seen: set[str] = set()
    parameters = []
    while True:
        remainder = text[position:]
        stripped = remainder.lstrip()
        position += len(remainder) - len(stripped)
        if stripped.startswith(PARAMETER_OPEN):
            key, value_start = _tag_name(text, position + len(PARAMETER_OPEN), "parameter")
            value_end = text.find(PARAMETER_CLOSE, value_start)
            value = text[value_start:value_end] if value_end >= 0 else ""
            if value_end < 0 or any(tag in value for tag in (PARAMETER_OPEN, FUNCTION_OPEN, FUNCTION_CLOSE)):
                raise FormatViolation(
                    "unclosed_parameter", "Your parameter %r was never closed with </parameter>." % key
                )
            if key in seen:
                raise FormatViolation(
                    "duplicate_parameter", "Your call sets the parameter %r more than once." % key
                )
            seen.add(key)
            parameters.append((key, _trim_newlines(value)))
            position = value_end + len(PARAMETER_CLOSE)
        elif stripped.startswith(FUNCTION_CLOSE):
            after = stripped[len(FUNCTION_CLOSE) :].strip()
            if after.startswith(FUNCTION_OPEN):
                raise _multiple_functions()
            if after:
                raise _envelope("there is text after </function>")
            return Call(name, tuple(parameters))
        elif not stripped:
            raise FormatViolation(
                "unclosed_function", "Your <function=%s> was never closed with </function>." % name
            )
        elif stripped.startswith(FUNCTION_OPEN):
            raise _multiple_functions()
        else:
            raise _envelope("there is text between the parameters")


def _tag_name(text: str, start: int, kind: str) -> tuple[str, int]:
    """The name of the tag whose name starts at ``start``, and the position after its ``>``."""
    end = text.find(">", start)
    name = text[start:end] if end >= 0 else text[start:]
    if end < 0 or not _NAME.fullmatch(name):
        written = text[start - len("<%s=" % kind) :].split("\n", 1)[0][:60]
        raise FormatViolation(
            "malformed_tag",
            "Your <%s=...> tag is malformed: you wrote %r. Write <%s=NAME> with a closing > and nothing "
            "else on the line, for example <%s=%s>." % (kind, written, kind, kind, _example(kind)),
        )
    return name, end + 1


def _example(kind: str) -> str:
    return "execute_bash" if kind == "function" else "command"


def _trim_newlines(value: str) -> str:
    """``value`` without the one newline that the template writes on each side of it."""
    value = value.removeprefix("\n")
    return value.removesuffix("\n")


def _value_text(value: object) -> str:
    """How Qwen3.5's chat template writes an argument value: a string as it is, else its text."""
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _envelope(problem: str) -> FormatViolation:
    return FormatViolation(
        "malformed_call",
        "Your tool call is malformed: %s. Write <tool_call>, then <function=NAME>, then one "
        "<parameter=KEY>...</parameter> per argument, then </function> and </tool_call>." % problem,
    )


def _multiple_functions() -> FormatViolation:
    return FormatViolation(
        "multiple_functions",
        "Your <tool_call> block has more than one <function=...>; call one tool per response.",
    )
