"""The BPO and Shepherd branch selectors, which pick restart points on a failed parent.

Both choose only among the parent's exact checkpoints. BPO ranks them by the policy's
first-token entropy at each point, and Shepherd asks a meta-agent that reads the trajectory.
Neither writes guidance: each sibling samples from the selected state. Every model request,
with its response, is written under ``args.output`` for an audit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx

from sprout.contract.mini_history import load_prefix
from sprout.contract.sampling import endpoint
from sprout.worker.agents.mini_swe.history import actor_tools_at, reference_at
from sprout.worker.checkpoint.rollback import Checkpoint
from sprout.worker.gateway.auth import bearer_headers
from sprout.worker.gateway.completions import choice_error_status
from sprout.worker.jobs.review.branching import extract_json, render_transcript
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, Prompts
from sprout.worker.run.journal import read_journal


def token_entropy(token: dict[str, Any]) -> dict[str, float]:
    """A lower bound on the entropy of one token of ``logprobs.content``, from its top tokens and their tail.

    Raises ValueError when it has no ``top_logprobs``, a log probability is not finite or is positive,
    a token repeats, or the mass of the top tokens exceeds one.
    """
    entries = token.get("top_logprobs")
    if not entries:
        raise ValueError("BPO requires top_logprobs; this provider omitted them")
    probabilities: dict[bytes, float] = {}
    for entry in entries:
        logprob = float(entry["logprob"])
        if not math.isfinite(logprob) or logprob > 0:
            raise ValueError("Non-finite or positive token log probability")
        raw = entry.get("bytes")
        identity = bytes(raw) if raw is not None else entry["token"].encode("utf-8")
        if identity in probabilities:
            raise ValueError("Duplicate token in top_logprobs")
        probabilities[identity] = math.exp(logprob)
    mass = math.fsum(probabilities.values())
    if mass > 1 + 1e-6:
        raise ValueError("Reported top-token probability mass exceeds one")
    tail = max(0.0, 1.0 - mass)
    entropy = -math.fsum(p * math.log(p) for p in [*probabilities.values(), tail] if p)
    return {
        "entropy_lower_bound_nats": entropy,
        "top_mass": mass,
        "tail_mass": tail,
        "candidate_count": len(probabilities),
    }


def bpo_select(
    scores: list[dict[str, Any]], budget: int = 7, min_spacing: int = 64, max_points: int = 7
) -> list[dict[str, Any]]:
    """``budget`` branches over up to ``max_points`` of ``scores``, at least ``min_spacing`` tokens apart.

    The checkpoints with the highest entropy go first, and the branches are dealt over them in turn. Raises
    ValueError for a negative budget or spacing, no point allowed, a repeated step or no eligible checkpoint.
    """
    if budget < 0 or min_spacing < 0 or max_points < 1:
        raise ValueError("Invalid BPO budget, spacing, or point limit")
    if budget == 0:
        return []
    if len({row["step"] for row in scores}) != len(scores):
        raise ValueError("Duplicate backbone checkpoint")
    ranked = sorted(scores, key=lambda row: (-row["entropy_lower_bound_nats"], row["step"]))
    selected: list[dict[str, Any]] = []
    for score in ranked:
        if all(abs(score["token_position"] - other["token_position"]) >= min_spacing for other in selected):
            selected.append(score)
            if len(selected) >= min(budget, max_points):
                break
    if not selected:
        raise ValueError("No eligible probability-scored branch points")
    return [
        dict(selected[index % len(selected)], sibling_index=2 + index // len(selected))
        for index in range(budget)
    ]


def shepherd_select(proposal: dict[str, Any], eligible: list[int], budget: int) -> list[dict[str, Any]]:
    """``budget`` branches at the checkpoint step that the Shepherd ``proposal`` names, with its reason.

    Raises ValueError unless the proposal names a step in ``eligible`` and a nonempty reason.
    """
    step = proposal.get("checkpoint_step")
    if type(step) is not int or step not in eligible:
        raise ValueError("Shepherd selected a checkpoint outside the exact eligible set")
    reason = proposal.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Shepherd selection needs a nonempty reason")
    return [{"step": step, "sibling_index": index + 2, "reason": reason} for index in range(budget)]


def write_json(path: Path, value: object) -> None:
    """Write ``value`` to ``path`` as JSON, through a temporary file, so that no reader sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


class ModelClient:
    """A Chat Completions client that retries transient failures and keeps a record of each request."""

    def __init__(self, base_url: str, api_key: str | None, timeout: float = 1800.0) -> None:
        base = endpoint(base_url).rstrip("/")
        self.url = base + "/v1/chat/completions"
        self.api_key = api_key
        self.timeout = timeout

    def complete(self, payload: dict[str, Any], audit: Path) -> dict[str, Any]:
        """The response body of one completion of ``payload``, in up to 8 tries. ``audit`` records each try.

        A timeout, a transport error, a retryable status, a body without one choice or a failed choice is
        tried again, after a backoff of up to 30 s. Any other error, and the last retryable one, is raised.
        """
        record: dict[str, Any] = {"request": payload, "started_at": time.time(), "attempts": []}
        write_json(audit, record)
        max_attempts = 8
        retry_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(1, max_attempts + 1):
            record.pop("response", None)
            attempt_record: dict[str, Any] = {"attempt": attempt, "started_at": time.time()}
            record["attempts"].append(attempt_record)
            try:
                response = httpx.post(
                    self.url, json=payload, headers=bearer_headers(self.api_key), timeout=self.timeout
                )
                attempt_record["http_status"] = response.status_code
                if response.status_code != 200:
                    attempt_record["response_text"] = response.text[:4000]
                    error = RuntimeError(f"model endpoint returned HTTP {response.status_code}")
                    if response.status_code not in retry_statuses:
                        raise error
                    raise httpx.TransportError(str(error))
                body: dict[str, Any] = response.json()
                record["response"] = body
                attempt_record["usage"] = body.get("usage")
                if body.get("error") or len(body.get("choices") or []) != 1:
                    raise httpx.TransportError("model endpoint returned no unique completion")
                if choice_error_status(body) is not None:
                    raise httpx.TransportError("model endpoint answered 200 with a failed choice")
                record["http_status"] = response.status_code
                record["response"] = body
                return body
            except (httpx.TimeoutException, httpx.TransportError) as error:
                attempt_record["error_type"] = type(error).__name__
                attempt_record["error"] = str(error)
                if attempt == max_attempts:
                    record["error_type"] = type(error).__name__
                    record["error"] = str(error)
                    raise
                delay = min(30, 2 ** (attempt - 1))
                attempt_record["retry_delay_seconds"] = delay
                time.sleep(delay)
            except Exception as error:
                attempt_record["error_type"] = type(error).__name__
                attempt_record["error"] = str(error)
                record["error_type"] = type(error).__name__
                record["error"] = str(error)
                raise
            finally:
                attempt_record["finished_at"] = time.time()
                attempt_record["seconds"] = attempt_record["finished_at"] - attempt_record["started_at"]
                record["finished_at"] = time.time()
                record["seconds"] = record["finished_at"] - record["started_at"]
                write_json(audit, record)
        raise AssertionError("the last try returns or raises")


def request_positions(journal: Path, points: dict[int, Checkpoint]) -> dict[int, int]:
    """For each step of ``points``, the completion tokens that the run sampled before its next request.

    A step that no request follows is absent. Raises ValueError for a response whose usage has no
    completion-token count.
    """
    positions = {}
    cumulative = 0
    previous_turn_step = None
    for row in read_journal(journal):
        if row.get("type") == "raw.mini-swe-agent":
            if previous_turn_step in points and previous_turn_step not in positions:
                positions[previous_turn_step] = cumulative
            usage = (row.get("response") or {}).get("usage") or {}
            count = usage.get("completion_tokens")
            if type(count) is not int or count < 0:
                raise ValueError("BPO requires original completion-token accounting")
            cumulative += count
        elif row.get("type") == "model.turn.completed":
            step = row.get("step")
            if type(step) is int:
                previous_turn_step = step
    return positions


def score_point(
    args: argparse.Namespace, client: ModelClient, journal: Path, point: Checkpoint, token_position: int
) -> dict[str, Any]:
    """The first-token entropy of the policy at ``point``, written to ``entropy/step-<N>.json``.

    Raises ValueError when the journal has no exact mini prefix at the point, or the endpoint
    returns no log probabilities.
    """
    reference = reference_at(journal, point.step, point.session_ckpt) if point.session_ckpt else None
    if reference is None:
        raise ValueError(f"No exact mini prefix for step {point.step}")
    prefix = load_prefix(reference)
    messages = [row["message"] for row in prefix if row["type"] == "mini.message"]
    payload = {
        "model": args.model,
        "messages": messages,
        "tools": actor_tools_at(journal, point.step),
        "reasoning_effort": args.reasoning_effort,
        "stream": False,
        "logprobs": True,
        "top_logprobs": args.bpo_top_k,
        # A high-reasoning Qwen spends reasoning tokens before its first visible one, so a cap of one
        # token returns no content and no logprobs. Only the first visible token is scored below.
        "max_tokens": 64,
    }
    response = client.complete(payload, args.output / "entropy" / f"step-{point.step}.request.json")
    tokens = (response["choices"][0].get("logprobs") or {}).get("content") or []
    if not tokens:
        raise ValueError("BPO endpoint omitted requested first-token logprobs")
    result = {
        "step": point.step,
        "token_position": token_position,
        "snapshot_id": point.snapshot_id,
        "scoring_usage": response.get("usage"),
        "first_token": tokens[0],
        "entropy_kind": "first-content-token-top-k-plus-tail-lower-bound",
    }
    result.update(token_entropy(tokens[0]))
    write_json(args.output / "entropy" / f"step-{point.step}.json", result)
    return result


def select_points(
    args: argparse.Namespace,
    problem: str,
    journal: Path,
    points: dict[int, Checkpoint],
    client: ModelClient,
    prompts: Prompts = DEFAULT_PROMPTS,
) -> list[dict[str, Any]]:
    """The branches that ``args.method`` (``bpo`` or ``shepherd``) selects over the parent's ``points``.

    Each has the ``step`` it starts from and its ``sibling_index``. The selector's requests, scores and
    proposal are written under ``args.output``. Raises ValueError when the selector makes no valid selection.
    """
    if args.method == "bpo":
        positions = request_positions(journal, points)
        eligible = [point for step, point in sorted(points.items()) if step in positions]
        if not eligible:
            raise ValueError("No exact checkpoint maps to a following provider request")
        with ThreadPoolExecutor(max_workers=args.bpo_scoring_workers) as pool:
            scores = list(
                pool.map(
                    lambda point: score_point(args, client, journal, point, positions[point.step]),
                    eligible,
                )
            )
        selected = bpo_select(scores, args.max_rollouts - 1, args.bpo_min_spacing, args.bpo_max_points)
        write_json(args.output / "scores.json", scores)
        return selected
    transcript, lo, hi = render_transcript(journal, token_budget=args.meta_transcript_tokens)
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": prompts.shepherd_system},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": problem,
                        "reward": 0,
                        "eligible_checkpoint_steps": sorted(points),
                        "transcript_visible_step_range": [lo, hi],
                        "trajectory": transcript,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "reasoning_effort": args.reasoning_effort,
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "max_tokens": args.meta_max_tokens,
    }
    response = client.complete(payload, args.output / "meta-request.json")
    if response["choices"][0].get("finish_reason") != "stop":
        raise ValueError("Incomplete Shepherd selector response")
    proposal = extract_json(response["choices"][0]["message"].get("content") or "")
    write_json(args.output / "proposal.json", proposal)
    return shepherd_select(proposal, sorted(points), args.max_rollouts - 1)
