"""The BPO and Shepherd selectors of ``sprout.eval.policies``, and the model client that they call.

The selection rules are small tests. The model client and the selectors call a scripted chat
endpoint on a local port (``tests.support.http.chat_server``), and read a parent journal that the
fake worker writes (``tests.eval.worker.write_rollout``): three turns of one ``ls`` call each.
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import yaml

from sprout.eval import policies
from sprout.eval.policies import (
    ModelClient,
    bpo_select,
    request_positions,
    score_point,
    select_points,
    shepherd_select,
    token_entropy,
)
from sprout.worker.agents.mini_swe.history import available_branch_points
from sprout.worker.agents.mini_swe.tools import mini_tool_schema
from sprout.worker.checkpoint.rollback import Checkpoint
from sprout.worker.jobs.review.branching import render_transcript
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS_PATH, load_prompts
from sprout.worker.run.journal import JournalWriter
from tests.eval.worker import snapshot_id, turn_messages, write_rollout
from tests.support.http import chat_server

LN_HALF = math.log(0.5)
PROBLEM = "Write `ready` followed by a newline into `/app/answer.txt`.\n"


def top(*probabilities: float, text: str = "t") -> list[dict[str, Any]]:
    """The ``top_logprobs`` of one token: a distinct token for each of ``probabilities``."""
    return [{"token": f"{text}{index}", "logprob": math.log(p)} for index, p in enumerate(probabilities)]


# --- token entropy ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries, entropy, top_mass, tail_mass, count",
    [
        (top(0.5, 0.5), math.log(2), 1.0, 0.0, 2),
        (top(0.25), 0.5623351446188083, 0.25, 0.75, 1),
        (top(1.0), 0.0, 1.0, 0.0, 1),
        (top(0.50000005, 0.50000005), 0.6931471805599453, 1.0000001, 0.0, 2),
    ],
    ids=["no-tail", "a-tail", "certain", "mass-within-rounding-of-one"],
)
def test_the_entropy_lower_bound_counts_the_mass_outside_the_top_tokens_as_one_more_outcome(
    entries: list[dict[str, Any]], entropy: float, top_mass: float, tail_mass: float, count: int
) -> None:
    """The tail is one outcome, so the bound is below the entropy of any split of the tail."""
    assert token_entropy({"top_logprobs": entries}) == pytest.approx(
        {
            "entropy_lower_bound_nats": entropy,
            "top_mass": top_mass,
            "tail_mass": tail_mass,
            "candidate_count": count,
        }
    )


def test_two_top_tokens_with_one_text_and_different_bytes_are_two_tokens() -> None:
    """A token that is part of a UTF-8 character prints as U+FFFD. Its bytes say which token it is."""
    entries = [
        {"token": "�", "logprob": LN_HALF, "bytes": [0xE4]},
        {"token": "�", "logprob": LN_HALF, "bytes": [0xE5]},
    ]

    assert token_entropy({"top_logprobs": entries})["candidate_count"] == 2


@pytest.mark.parametrize(
    "token, message",
    [
        ({}, "BPO requires top_logprobs; this provider omitted them"),
        ({"top_logprobs": []}, "BPO requires top_logprobs; this provider omitted them"),
        ({"top_logprobs": [{"token": "a", "logprob": 0.1}]}, "Non-finite or positive token log probability"),
        (
            {"top_logprobs": [{"token": "a", "logprob": math.nan}]},
            "Non-finite or positive token log probability",
        ),
        (
            {"top_logprobs": [{"token": "a", "logprob": -math.inf}]},
            "Non-finite or positive token log probability",
        ),
        (
            {"top_logprobs": [{"token": "a", "logprob": LN_HALF}, {"token": "a", "logprob": LN_HALF}]},
            "Duplicate token in top_logprobs",
        ),
        (
            {
                "top_logprobs": [
                    {"token": "a", "logprob": -1.0, "bytes": [97]},
                    {"token": "b", "logprob": -1.0, "bytes": [97]},
                ]
            },
            "Duplicate token in top_logprobs",
        ),
        ({"top_logprobs": top(0.6, 0.6)}, "Reported top-token probability mass exceeds one"),
    ],
    ids=[
        "missing",
        "empty",
        "positive",
        "nan",
        "minus-infinity",
        "same-text",
        "same-bytes",
        "mass-above-one",
    ],
)
def test_bpo_scores_no_token_whose_log_probabilities_it_cannot_trust(
    token: dict[str, Any], message: str
) -> None:
    """No entropy is invented for a provider that omits or garbles the log probabilities."""
    with pytest.raises(ValueError, match=message):
        token_entropy(token)


# --- the BPO selection ----------------------------------------------------------------------


def score(step: int, position: int, entropy: float) -> dict[str, Any]:
    """The score of the checkpoint at ``step``, ``position`` completion tokens into the parent."""
    return {"step": step, "token_position": position, "entropy_lower_bound_nats": entropy}


def branches(selected: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Each selected branch as its step and its sibling index."""
    return [(row["step"], row["sibling_index"]) for row in selected]


SCORES = [score(1, 0, 0.5), score(2, 50, 2.0), score(3, 120, 1.0), score(4, 130, 1.5), score(5, 300, 1.0)]


@pytest.mark.parametrize(
    "budget, max_points, expected",
    [
        (7, 7, [(2, 2), (4, 2), (5, 2), (2, 3), (4, 3), (5, 3), (2, 4)]),
        (3, 2, [(2, 2), (4, 2), (2, 3)]),
        (2, 7, [(2, 2), (4, 2)]),
        (0, 7, []),
    ],
    ids=["dealt-over-the-points", "at-most-max-points", "budget-below-the-points", "no-budget"],
)
def test_bpo_deals_the_branches_over_the_most_uncertain_checkpoints_that_are_far_enough_apart(
    budget: int, max_points: int, expected: list[tuple[int, int]]
) -> None:
    """Step 3 is 10 tokens from the more uncertain step 4, and step 1 is 50 from step 2: neither is taken.

    Each pass over the points gives each one sibling more. The parent is sibling 1."""
    selected = bpo_select(SCORES, budget, min_spacing=64, max_points=max_points)

    assert branches(selected) == expected


@pytest.mark.parametrize("position, expected", [(64, [(1, 2), (2, 2)]), (63, [(1, 2), (1, 3)])])
def test_two_checkpoints_exactly_min_spacing_apart_are_both_eligible(
    position: int, expected: list[tuple[int, int]]
) -> None:
    selected = bpo_select([score(1, 0, 2.0), score(2, position, 1.0)], 2, min_spacing=64, max_points=7)

    assert branches(selected) == expected


def test_checkpoints_of_equal_entropy_rank_by_step() -> None:
    selected = bpo_select([score(7, 0, 1.0), score(3, 500, 1.0)], 1, min_spacing=64, max_points=1)

    assert branches(selected) == [(3, 2)]


@pytest.mark.parametrize(
    "scores, budget, spacing, max_points, message",
    [
        (SCORES, -1, 64, 7, "Invalid BPO budget, spacing, or point limit"),
        (SCORES, 7, -1, 7, "Invalid BPO budget, spacing, or point limit"),
        (SCORES, 7, 64, 0, "Invalid BPO budget, spacing, or point limit"),
        ([score(1, 0, 1.0), score(1, 99, 2.0)], 7, 64, 7, "Duplicate backbone checkpoint"),
        ([], 7, 64, 7, "No eligible probability-scored branch points"),
    ],
    ids=["negative-budget", "negative-spacing", "no-point-allowed", "repeated-step", "no-checkpoint"],
)
def test_bpo_refuses_a_selection_it_cannot_make(
    scores: list[dict[str, Any]], budget: int, spacing: int, max_points: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        bpo_select(scores, budget, min_spacing=spacing, max_points=max_points)


# --- the Shepherd selection ------------------------------------------------------------------


def test_shepherd_samples_every_branch_from_the_one_checkpoint_that_its_proposal_names() -> None:
    proposal = {"checkpoint_step": 2, "reason": "The listing at step 2 missed the file."}

    selected = shepherd_select(proposal, [1, 2, 3], 3)

    assert selected == [
        {"step": 2, "sibling_index": 2, "reason": "The listing at step 2 missed the file."},
        {"step": 2, "sibling_index": 3, "reason": "The listing at step 2 missed the file."},
        {"step": 2, "sibling_index": 4, "reason": "The listing at step 2 missed the file."},
    ]


@pytest.mark.parametrize(
    "proposal, message",
    [
        ({"checkpoint_step": 4, "reason": "late"}, "outside the exact eligible set"),
        ({"checkpoint_step": True, "reason": "late"}, "outside the exact eligible set"),
        ({"checkpoint_step": "2", "reason": "late"}, "outside the exact eligible set"),
        ({"reason": "late"}, "outside the exact eligible set"),
        ({"checkpoint_step": 2, "reason": "  "}, "needs a nonempty reason"),
        ({"checkpoint_step": 2}, "needs a nonempty reason"),
    ],
    ids=["not-eligible", "a-boolean", "a-string", "no-step", "blank-reason", "no-reason"],
)
def test_shepherd_refuses_a_proposal_outside_the_eligible_checkpoints_or_without_a_reason(
    proposal: dict[str, Any], message: str
) -> None:
    """``True == 1``, so a boolean step would pass a plain membership test on ``[1, 2, 3]``."""
    with pytest.raises(ValueError, match=message):
        shepherd_select(proposal, [1, 2, 3], 3)


# --- request positions ----------------------------------------------------------------------


def journal_of(tmp_path: Path, records: list[tuple[str, dict[str, Any]]]) -> Path:
    """A journal of ``records``, each an event type and its payload."""
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path) as writer:
        for kind, payload in records:
            writer.emit(kind, **payload)
    return path


def response(tokens: object) -> tuple[str, dict[str, Any]]:
    """A model response record whose usage counts ``tokens`` completion tokens."""
    return ("raw.mini-swe-agent", {"response": {"usage": {"completion_tokens": tokens}}})


def completed(step: int) -> tuple[str, dict[str, Any]]:
    return ("model.turn.completed", {"turn_id": f"turn-{step}", "step": step})


def checkpoints(*steps: int) -> dict[int, Checkpoint]:
    return {
        step: Checkpoint(step=step, seq=0, snapshot_id=f"snap-{step}", session_ckpt="ses-1") for step in steps
    }


def test_a_checkpoints_position_is_the_completion_tokens_sampled_before_the_next_request(
    tmp_path: Path,
) -> None:
    """The last checkpoint has no request after it. A turn that is not a checkpoint gets no position."""
    journal = journal_of(
        tmp_path,
        [
            response(30),
            completed(1),
            response(50),
            completed(2),
            response(20),
            completed(3),
            response(40),
            completed(4),
        ],
    )

    assert request_positions(journal, checkpoints(1, 3, 4)) == {1: 30, 3: 100}


@pytest.mark.parametrize("tokens", [None, -1, True, 1.5], ids=["missing", "negative", "boolean", "fraction"])
def test_bpo_refuses_a_parent_whose_responses_do_not_count_their_completion_tokens(
    tmp_path: Path, tokens: object
) -> None:
    journal = journal_of(tmp_path, [response(30), completed(1), response(tokens)])

    with pytest.raises(ValueError, match="BPO requires original completion-token accounting"):
        request_positions(journal, checkpoints(1))


# --- the model client -----------------------------------------------------------------------


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """The backoffs of the model client, each recorded instead of slept."""
    recorded: list[float] = []
    monkeypatch.setattr(policies, "time", SimpleNamespace(time=time.time, sleep=recorded.append))
    return recorded


def completion(
    content: str = "Run", *, finish_reason: str = "stop", usage: int = 7, **choice: Any
) -> dict[str, Any]:
    """A chat completion with one choice."""
    message = {"role": "assistant", "content": content}
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message, **choice}],
        "usage": {"prompt_tokens": 11, "completion_tokens": usage},
    }


def tries(audit: Path) -> list[tuple[int, int | None, int | None]]:
    """Each try that the audit file records: its number, its HTTP status and the backoff after it."""
    record = json.loads(audit.read_text())
    return [
        (row["attempt"], row.get("http_status"), row.get("retry_delay_seconds")) for row in record["attempts"]
    ]


@pytest.mark.parametrize(
    "failure",
    [
        (503, {"error": {"message": "overloaded"}}),
        {**completion(), "choices": [completion()["choices"][0], completion()["choices"][0]]},
        {**completion(), "error": {"message": "upstream reset"}},
    ],
    ids=["retryable-status", "two-choices", "error-body"],
)
def test_a_transient_failure_is_tried_again_after_a_backoff(
    tmp_path: Path, sleeps: list[float], failure: Any
) -> None:
    with chat_server([failure, completion("Done")]) as server:
        body = ModelClient(server.url + "/v1", "fixture-key").complete(
            {"model": "policy"}, tmp_path / "audit.json"
        )

    assert (body["choices"][0]["message"]["content"], len(server.requests), sleeps) == ("Done", 2, [1])
    assert tries(tmp_path / "audit.json")[0][2] == 1


def test_a_status_that_no_retry_can_fix_fails_at_once(tmp_path: Path, sleeps: list[float]) -> None:
    with chat_server([(400, {"error": {"message": "bad request"}}), completion()]) as server:
        with pytest.raises(RuntimeError, match="model endpoint returned HTTP 400"):
            ModelClient(server.url + "/v1", None).complete({"model": "policy"}, tmp_path / "audit.json")

    assert (len(server.requests), sleeps, tries(tmp_path / "audit.json")) == (1, [], [(1, 400, None)])


def test_the_client_gives_up_after_eight_tries_with_backoffs_that_double_up_to_30_seconds(
    tmp_path: Path, sleeps: list[float]
) -> None:
    """The scripted endpoint answers each request with 500."""
    with chat_server([]) as server:
        with pytest.raises(httpx.TransportError, match="model endpoint returned HTTP 500"):
            ModelClient(server.url + "/v1", None).complete({"model": "policy"}, tmp_path / "audit.json")

    assert (len(server.requests), sleeps) == (8, [1, 2, 4, 8, 16, 30, 30])


# --- the selectors --------------------------------------------------------------------------


def logprob_completion(*probabilities: float) -> dict[str, Any]:
    """A completion whose first visible token has a top token for each of ``probabilities``."""
    first = {"token": "t0", "logprob": math.log(probabilities[0]), "top_logprobs": top(*probabilities)}
    return completion("t0", finish_reason="length", logprobs={"content": [first]})


def options(tmp_path: Path, **changes: Any) -> argparse.Namespace:
    """The selector settings of an evaluation (``eval.evaluate.parse_args``), with ``changes``."""
    settings = {
        "method": "bpo",
        "model": "policy",
        "reasoning_effort": "high",
        "output": tmp_path / "out",
        "max_rollouts": 4,
        "bpo_top_k": 5,
        "bpo_min_spacing": 64,
        "bpo_max_points": 7,
        "bpo_scoring_workers": 1,
        "meta_max_tokens": 1000,
        "meta_transcript_tokens": 60000,
    }
    return argparse.Namespace(**{**settings, **changes})


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    """The journal of a parent rollout of three turns, each sampled with 100 completion tokens."""
    directory = tmp_path / "parent"
    directory.mkdir()
    return write_rollout(directory, job_id="job-0")


def test_bpo_scores_a_checkpoint_by_the_first_token_that_the_policy_samples_there(
    tmp_path: Path, parent: Path, sleeps: list[float]
) -> None:
    """The request continues the native history up to the checkpoint, with the actor's tools."""
    args = options(tmp_path)
    point = available_branch_points(parent)[2]

    with chat_server([logprob_completion(0.5, 0.5)]) as server:
        result = score_point(args, ModelClient(server.url + "/v1", None), parent, point, 100)

    assert server.requests == [
        {
            "model": "policy",
            "messages": turn_messages(1) + turn_messages(2),
            "tools": mini_tool_schema(),
            "reasoning_effort": "high",
            "stream": False,
            "logprobs": True,
            "top_logprobs": 5,
            "max_tokens": 64,
        }
    ]
    expected = {"step": 2, "token_position": 100, "snapshot_id": snapshot_id("job-0", 2)}
    assert {key: result[key] for key in expected} == expected
    assert result["entropy_lower_bound_nats"] == pytest.approx(math.log(2))
    assert json.loads((tmp_path / "out" / "entropy" / "step-2.json").read_text()) == result


def test_bpo_refuses_an_endpoint_that_returns_no_first_token_log_probabilities(
    tmp_path: Path, parent: Path, sleeps: list[float]
) -> None:
    point = available_branch_points(parent)[2]

    with chat_server([completion("Run")]) as server:
        with pytest.raises(ValueError, match="BPO endpoint omitted requested first-token logprobs"):
            score_point(options(tmp_path), ModelClient(server.url + "/v1", None), parent, point, 100)


def test_bpo_selects_among_the_checkpoints_that_a_later_request_follows(
    tmp_path: Path, parent: Path, sleeps: list[float]
) -> None:
    """Step 3 is the last turn: no request follows it, so it has no position and no score.

    Step 2 is more uncertain than step 1, and 100 tokens after it, so it ranks first."""
    args = options(tmp_path, max_rollouts=4)

    with chat_server([logprob_completion(0.9), logprob_completion(0.5, 0.5)]) as server:
        selected = select_points(
            args, PROBLEM, parent, available_branch_points(parent), ModelClient(server.url + "/v1", None)
        )

    assert branches(selected) == [(2, 2), (1, 2), (2, 3)]
    scores = json.loads((tmp_path / "out" / "scores.json").read_text())
    assert [(row["step"], row["token_position"]) for row in scores] == [(1, 100), (2, 200)]


def test_bpo_refuses_a_parent_without_a_checkpoint_that_a_request_follows(
    tmp_path: Path, parent: Path, sleeps: list[float]
) -> None:
    last = {3: available_branch_points(parent)[3]}

    with pytest.raises(ValueError, match="No exact checkpoint maps to a following provider request"):
        select_points(options(tmp_path), PROBLEM, parent, last, ModelClient("http://model.test/v1", None))


@pytest.fixture
def edited_prompts(tmp_path: Path) -> Path:
    """A prompts file whose Shepherd system prompt is not the packaged one."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["shepherd"]["system"] = "Pick the one checkpoint to restart from."
    path = tmp_path / "prompts.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


SHEPHERD_ANSWER = '```json\n{"checkpoint_step": 2, "reason": "The listing at step 2 missed the file."}\n```'


def test_shepherd_asks_the_meta_agent_with_the_prompts_it_is_given_and_samples_from_its_choice(
    tmp_path: Path, parent: Path, edited_prompts: Path, sleeps: list[float]
) -> None:
    args = options(tmp_path, method="shepherd", max_rollouts=3)
    transcript, first, last = render_transcript(parent, token_budget=60000)

    with chat_server([completion(SHEPHERD_ANSWER)]) as server:
        selected = select_points(
            args,
            PROBLEM,
            parent,
            available_branch_points(parent),
            ModelClient(server.url + "/v1", None),
            prompts=load_prompts(edited_prompts),
        )

    (request,) = server.requests
    assert {**request, "messages": None} == {
        "model": "policy",
        "messages": None,
        "reasoning_effort": "high",
        "temperature": 1.0,
        "top_p": 0.95,
        "stream": False,
        "max_tokens": 1000,
    }
    system, user = request["messages"]
    assert system == {"role": "system", "content": "Pick the one checkpoint to restart from."}
    assert (user["role"], json.loads(user["content"])) == (
        "user",
        {
            "task": PROBLEM,
            "reward": 0,
            "eligible_checkpoint_steps": [1, 2, 3],
            "transcript_visible_step_range": [first, last],
            "trajectory": transcript,
        },
    )
    assert branches(selected) == [(2, 2), (2, 3)]
    assert json.loads((tmp_path / "out" / "proposal.json").read_text()) == {
        "checkpoint_step": 2,
        "reason": "The listing at step 2 missed the file.",
    }


def test_shepherd_refuses_a_meta_agent_answer_that_was_cut_off(
    tmp_path: Path, parent: Path, sleeps: list[float]
) -> None:
    with chat_server([completion(SHEPHERD_ANSWER, finish_reason="length")]) as server:
        with pytest.raises(ValueError, match="Incomplete Shepherd selector response"):
            select_points(
                options(tmp_path, method="shepherd"),
                PROBLEM,
                parent,
                available_branch_points(parent),
                ModelClient(server.url + "/v1", None),
            )
