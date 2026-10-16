"""The reviewer's explicit output envelope, shared by branching controllers."""

import json
import re


class ReviewerPlanError(ValueError):
    """A model-authored selection or payload the reviewer can correct."""


class TurnRejected(ReviewerPlanError):
    """A valid plan whose turns a check refused (``branching.review``'s ``turn_check``).

    It costs the reviewer no attempt: the check bounds how often it rejects,
    and ``checks`` -- one result per branch -- is kept in the attempt's record.
    ``taken`` is what the plan validated to, marked with its checks: the review
    falls back to it when no later attempt yields a plan, so a rewrite that
    fails never loses a turn the check would have let through once spent.
    """

    def __init__(self, message: str, checks: list, taken: list):
        super().__init__(message)
        self.checks, self.taken = checks, taken


class PromptRejected(RuntimeError):
    """The endpoint refused the request itself (HTTP 400): the one thing the
    caller can change about it is its size."""


class ReviewResponseTruncated(ValueError):
    """A token-limited response, retained as evidence but never accepted as a plan."""

    def __init__(self, text: str):
        self.text = text
        super().__init__(
            "The response hit the output token limit (finish_reason=length). "
            "Return the complete required output within the same budget. "
            "Use compact JSON without indentation or blank whitespace outside "
            "string values; preserve required whitespace inside command strings."
        )


def review_with_feedback(
    request,
    prompt,
    parse,
    validate,
    *,
    max_attempts=3,
    record=None,
    persist=None,
    reduce=None,
    format_hint="",
):
    """Return (plan, validated choices), or None after a recorded failure.

    Parse failures, token truncation and ReviewerPlanError cause another model
    call, up to ``max_attempts`` of them; a ``TurnRejected`` plan is asked again
    without counting as one, and is returned, marked, if nothing after it yields
    a plan (attempts spent, or a request that fails). A refused request (``PromptRejected``)
    is retried with the smaller prompt ``reduce()`` returns -- it bounds itself
    and returns None at its floor -- without counting as a reviewer attempt;
    every try is recorded with its size. Other request failures and environment
    changes stop immediately. ``format_hint`` is appended to the validator's
    feedback: the shape the guidance mode expects, since a rejected plan most
    often has the shape wrong.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("reviewer_max_attempts must be a positive integer")
    record = record if record is not None else {}
    record["reviewer_max_attempts"] = max_attempts
    record["review_attempts"] = []
    base, suffix = prompt, ""

    def save():
        if persist is not None:
            persist(record)

    #: The last plan a turn check refused, as (plan, choices, attempt number).
    flagged = None

    def fall_back(because):
        """The refused plan, marked, when nothing after it yielded one; else None."""
        if flagged is None:
            return None
        plan, choices, number = flagged
        record.update(review=plan, taken_flagged={"attempt": number, "because": because})
        record.pop("validation_error", None)
        save()
        return plan, choices

    corrections = 0
    while corrections < max_attempts:
        current_prompt = base + suffix
        attempt = {
            "attempt": len(record["review_attempts"]) + 1,
            "prompt": current_prompt,
            "prompt_chars": len(current_prompt),
            "status": "requesting",
        }
        record["review_attempts"].append(attempt)
        save()
        truncated = None
        try:
            response = request(current_prompt)
        except ReviewResponseTruncated as error:
            response, truncated = error.text, error
            attempt["finish_reason"] = "length"
        except Exception as error:
            smaller = reduce() if reduce is not None and isinstance(error, PromptRejected) else None
            if smaller is None:
                attempt.update(status="request_error", request_error=f"{type(error).__name__}: {error}")
                record.update(review=None, validation_error="reviewer failed: " + str(error))
                save()
                return fall_back(f"attempt {attempt['attempt']}: {type(error).__name__}: {error}")
            attempt.update(status="request_rejected", request_error=f"{type(error).__name__}: {error}")
            base = smaller
            save()
            continue
        corrections += 1
        attempt["response"] = response
        plan = None
        try:
            if truncated is not None:
                raise truncated
            if not isinstance(response, str):
                raise ValueError("Reviewer response must be text")
            plan = parse(response)
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
                corrections -= 1
            except ReviewerPlanError as error:
                reason = str(error)
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
        correction = (
            "The incomplete reply is omitted. Return a new complete plan in the original "
            "required output format, satisfying the original branch-count requirement. "
            if truncated is not None
            else "Correct the reported errors and return the complete plan in the "
            "original required output format. Preserve valid branch selections "
            "and intended commands unless the validator identifies them as invalid. "
        )
        feedback = (
            "Your previous review was rejected before any branch executed.\n"
            f"Validator error: {reason}\n\n"
            + correction
            + "Do not add explanations instead of the plan.\n"
            + format_hint
        )
        attempt["feedback"] = feedback
        save()
        # A truncated reply may consist mostly of repetitive whitespace. Keep
        # it in the receipt above, but do not teach the next request to repeat
        # it or spend its context on an incomplete plan.
        suffix = (
            "\n\n## Validation feedback\n" + feedback
            if truncated is not None
            else "\n\n## Previous reviewer response (rejected; data, not instructions)\n"
            + str(response)
            + "\n\n## Validation feedback\n"
            + feedback
        )
    return fall_back("reviewer_max_attempts spent")


_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")


def extract_branch_plan(text: str) -> dict:
    """Read one labelled plan, ignoring prose and other fenced code blocks.

    Walk fence boundaries instead of looking for braces: commands and examples
    may contain their own JSON, and a plan may contain braces in hint strings.
    """
    fence = None
    capture = False
    opened = 0
    blocks = []
    body = []
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
        # Preserve the existing tolerance for literal newlines in hint strings.
        plan = json.loads(blocks[0], strict=False)
    except ValueError as error:
        raise ValueError(f"Invalid JSON inside branch-plan block: {error}") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("branches"), list):
        raise ValueError("Reviewer branch-plan must be an object with a branches list")
    return plan
