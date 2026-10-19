"""The reviewer's correction loop, and the labelled ``branch-plan`` block that a plan comes in."""

import json
import re
from collections.abc import Callable
from typing import Any


class ReviewerPlanError(ValueError):
    """An error in a model-written selection or payload that the reviewer can correct."""


class TurnRejected(ReviewerPlanError):
    """A valid plan with an assistant turn that a turn check refused.

    ``checks`` holds one check result per branch, None for a branch without a turn. ``taken``
    holds the validated choices, marked with their checks. ``review_with_feedback`` returns
    them if no later attempt yields a plan."""

    # Any: the choices that the caller's validator made. This module does not know their type.
    def __init__(self, message: str, checks: list[dict[str, Any] | None], taken: list[Any]) -> None:
        super().__init__(message)
        self.checks, self.taken = checks, taken


class PromptRejected(RuntimeError):
    """The endpoint refused the request itself (HTTP 400). Only a smaller request can succeed."""


class ReviewResponseTruncated(ValueError):
    """A response that hit the output token limit. It stays in the record and is never taken as a plan."""

    def __init__(self, text: str) -> None:
        self.text = text
        super().__init__(
            "The response hit the output token limit (finish_reason=length). "
            "Return the complete required output within the same budget. "
            "Use compact JSON without indentation or blank whitespace outside "
            "string values; preserve required whitespace inside command strings."
        )


def review_with_feedback[Choice](
    request: Callable[[str], object],
    prompt: str,
    parse: Callable[[str], dict[str, Any]],
    validate: Callable[[dict[str, Any]], list[Choice]],
    *,
    max_attempts: int = 3,
    record: dict[str, Any] | None = None,
    persist: Callable[[dict[str, Any]], None] | None = None,
    reduce: Callable[[], str | None] | None = None,
    format_hint: str = "",
) -> tuple[dict[str, Any], list[Choice]] | None:
    """Ask ``request`` until ``validate`` accepts a plan. Return ``(plan, choices)``, or None on failure.

    A parse error, a truncated reply or a ``ReviewerPlanError`` uses one of ``max_attempts``. A
    ``TurnRejected`` plan uses none, and is returned if the attempts run out or a later request fails.
    After a ``PromptRejected``, ``reduce()`` gives a smaller prompt, or None to stop. Other errors stop."""
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("reviewer_max_attempts must be a positive integer")
    # ``record`` holds every try with the size of its prompt. ``persist`` receives it after each change.
    record = record if record is not None else {}
    record["reviewer_max_attempts"] = max_attempts
    record["review_attempts"] = []
    base, suffix = prompt, ""

    def save() -> None:
        if persist is not None:
            persist(record)

    # The last plan that a turn check refused, as (plan, choices, attempt number).
    flagged: tuple[dict[str, Any], list[Choice], int] | None = None
    corrections = 0
    while corrections < max_attempts:
        current_prompt = base + suffix
        attempt = _open_attempt(record=record, prompt=current_prompt)
        save()
        try:
            response, truncated = _ask(request=request, prompt=current_prompt)
        except Exception as error:
            # A refused prompt is not the reviewer's error, so a smaller one costs no attempt.
            smaller = reduce() if reduce is not None and isinstance(error, PromptRejected) else None
            if smaller is None:
                attempt.update(status="request_error", request_error=f"{type(error).__name__}: {error}")
                record.update(review=None, validation_error="reviewer failed: " + str(error))
                save()
                because = f"attempt {attempt['attempt']}: {type(error).__name__}: {error}"
                return _fall_back(flagged=flagged, because=because, record=record, save=save)
            attempt.update(status="request_rejected", request_error=f"{type(error).__name__}: {error}")
            base = smaller
            save()
            continue
        if truncated is not None:
            attempt["finish_reason"] = "length"
        corrections += 1
        attempt["response"] = response
        plan: dict[str, Any] | None = None
        try:
            plan = _parse_reply(response=response, truncated=truncated, parse=parse)
        except ValueError as error:
            reason = str(error)
        else:
            attempt["plan"] = plan
            try:
                choices = validate(plan)
            except TurnRejected as error:
                reason = str(error)
                attempt["turn_checks"] = error.checks
                flagged = (plan, error.taken, attempt["attempt"])
                # The turn check bounds its own rejections, so a refused turn costs no attempt.
                corrections -= 1
            except ReviewerPlanError as error:
                reason = str(error)
            # Any other error comes from the run's state, which the reviewer cannot correct.
            except (ValueError, OSError) as error:
                attempt.update(status="state_error", state_error=str(error))
                record.update(review=plan, validation_error=str(error))
                save()
                return None
            else:
                attempt["status"] = "validated"
                record["review"] = plan
                record.pop("validation_error", None)
                save()
                return plan, choices
        attempt.update(
            status="turn_rejected" if "turn_checks" in attempt else "validation_error",
            validation_error=reason,
        )
        record.update(review=plan, validation_error=reason)
        feedback = _feedback(reason=reason, truncated=truncated is not None, format_hint=format_hint)
        attempt["feedback"] = feedback
        save()
        suffix = _retry_suffix(feedback=feedback, response=response, truncated=truncated is not None)
    return _fall_back(flagged=flagged, because="reviewer_max_attempts spent", record=record, save=save)


def _open_attempt(*, record: dict[str, Any], prompt: str) -> dict[str, Any]:
    """A new entry of ``record["review_attempts"]``, for a request of ``prompt`` about to go out."""
    attempt: dict[str, Any] = {
        "attempt": len(record["review_attempts"]) + 1,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "status": "requesting",
    }
    record["review_attempts"].append(attempt)
    return attempt


def _ask(*, request: Callable[[str], object], prompt: str) -> tuple[object, ReviewResponseTruncated | None]:
    """The reply to ``prompt``, and the error when the reply hit the output token limit, else None."""
    try:
        return request(prompt), None
    except ReviewResponseTruncated as error:
        return error.text, error


def _parse_reply(
    *, response: object, truncated: ReviewResponseTruncated | None, parse: Callable[[str], dict[str, Any]]
) -> dict[str, Any]:
    """The plan that ``parse`` reads in ``response``. Raises ValueError for a truncated or non-text reply."""
    if truncated is not None:
        raise truncated
    if not isinstance(response, str):
        raise ValueError("Reviewer response must be text")
    return parse(response)


def _feedback(*, reason: str, truncated: bool, format_hint: str) -> str:
    """The validator feedback on a rejected reply: ``reason``, and a request for a new complete plan."""
    correction = (
        "The incomplete reply is omitted. Return a new complete plan in the original "
        "required output format, satisfying the original branch-count requirement. "
        if truncated
        else "Correct the reported errors and return the complete plan in the "
        "original required output format. Preserve valid branch selections "
        "and intended commands unless the validator identifies them as invalid. "
    )
    # ``format_hint`` gives the shape that the guidance mode expects, because a rejected
    # plan most often has the wrong shape.
    return (
        "Your previous review was rejected before any branch executed.\n"
        f"Validator error: {reason}\n\n"
        + correction
        + "Do not add explanations instead of the plan.\n"
        + format_hint
    )


def _retry_suffix(*, feedback: str, response: object, truncated: bool) -> str:
    """What the next prompt adds to the base prompt: the rejected reply unless truncated, and ``feedback``."""
    # A truncated reply can be mostly repeated whitespace. The record keeps it, and the next
    # prompt leaves it out, so that the model does not repeat it or spend context on it.
    return (
        "\n\n## Validation feedback\n" + feedback
        if truncated
        else "\n\n## Previous reviewer response (rejected; data, not instructions)\n"
        + str(response)
        + "\n\n## Validation feedback\n"
        + feedback
    )


def _fall_back[Choice](
    *,
    flagged: tuple[dict[str, Any], list[Choice], int] | None,
    because: str,
    record: dict[str, Any],
    save: Callable[[], None],
) -> tuple[dict[str, Any], list[Choice]] | None:
    """The refused plan ``flagged``, marked in ``record``, when nothing after it yielded one, else None.

    A turn check takes a plan once its rejections run out, so a failed rewrite must not lose it."""
    if flagged is None:
        return None
    plan, choices, number = flagged
    record.update(review=plan, taken_flagged={"attempt": number, "because": because})
    record.pop("validation_error", None)
    save()
    return plan, choices


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")


def extract_branch_plan(text: str) -> dict[str, Any]:
    """The plan in the one ``branch-plan`` fenced block of ``text``. Prose and other blocks are ignored.

    Raises ``ValueError`` unless ``text`` has exactly one closed ``branch-plan`` block, and its JSON
    is an object with a ``branches`` list."""
    # Follow the fence boundaries, not the braces: commands and examples can hold JSON of their
    # own, and hint strings can hold braces.
    fence: tuple[str, int] | None = None
    capture = False
    opened = 0
    blocks: list[str] = []
    body: list[str] = []
    for line in text.splitlines(keepends=True):
        match = _FENCE.fullmatch(line.rstrip("\r\n"))
        if fence is None:
            if match is None:
                continue
            delimiter, info = match.groups()
            fence = (delimiter[0], len(delimiter))
            capture = delimiter[0] == "`" and info.strip() == "branch-plan"
            if capture:
                opened += 1
                body = []
        elif (
            match is not None
            and match[1][0] == fence[0]
            and len(match[1]) >= fence[1]
            and not match[2].strip()
        ):
            if capture:
                blocks.append("".join(body))
            fence = None
            capture = False
        elif capture:
            body.append(line)

    if opened != 1:
        raise ValueError(f"Reviewer must return exactly one branch-plan block; found {opened}")
    if capture or len(blocks) != 1:
        raise ValueError("Reviewer branch-plan block is not closed")
    try:
        # Reviewers write literal newlines inside hint strings, and strict=False accepts them.
        plan = json.loads(blocks[0], strict=False)
    except ValueError as error:
        raise ValueError(f"Invalid JSON inside branch-plan block: {error}") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("branches"), list):
        raise ValueError("Reviewer branch-plan must be an object with a branches list")
    return plan
