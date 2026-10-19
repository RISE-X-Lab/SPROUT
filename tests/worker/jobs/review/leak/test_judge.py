"""The Jev leak judge: candidate spans found in code, one Jev question each, and both checks of one turn."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from sprout.worker.jobs.review.leak.judge import (
    QUESTION,
    HiddenText,
    JevError,
    JevLeakJudge,
    LeakJudgement,
    added_lines,
    candidate_spans,
    check_turn,
)

PATCH = """diff --git a/tests/test_links.py b/tests/test_links.py
--- a/tests/test_links.py
+++ b/tests/test_links.py
@@ -1,2 +1,5 @@
 import links
+def test_balanced_parentheses():
+    assert convert("![Alt](<my image (1).png>)") == "![[my image (1).png|Alt]]"
+    assert width(1234) == 1234
"""
STATEMENT = "Convert Markdown links to wiki links. `[Doc](doc.md)` becomes `[[doc.md|Doc]]`."
FIXTURE = "![Alt](<my image (1).png>)"


def reviewer_turn(command: str, content: str = "Check the parenthesised case.") -> dict[str, Any]:
    call = {
        "id": "reviewer-r1-b1-c1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": content, "tool_calls": [call]}


class FakeJev:
    """Jev behind ``httpx.MockTransport``. It answers each question with ``p``, after ``statuses``."""

    def __init__(self, p: float = 0.9, statuses: tuple[int, ...] = ()) -> None:
        self.p, self.statuses, self.requests = p, list(statuses), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(
            {"url": str(request.url), "authorization": request.headers["authorization"], "body": body}
        )
        if self.statuses:
            return httpx.Response(
                self.statuses.pop(0), json={"detail": "refused"}, headers={"retry-after": "0"}
            )
        answers = {name: {"type": "noul", "noul": self.p} for name in body["questions"]}
        return httpx.Response(
            200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 100}, "answers": answers}
        )


def judge_with(fake: FakeJev, **settings: Any) -> JevLeakJudge:
    return JevLeakJudge("typesafe-key", client=httpx.Client(transport=httpx.MockTransport(fake)), **settings)


def test_the_hidden_tests_are_the_lines_that_the_patch_adds():
    assert added_lines(PATCH) == (
        "def test_balanced_parentheses():\n"
        '    assert convert("![Alt](<my image (1).png>)") == "![[my image (1).png|Alt]]"\n'
        "    assert width(1234) == 1234"
    )


@pytest.mark.parametrize(
    "turn, public, spans",
    [
        pytest.param(f'echo "{FIXTURE}"', STATEMENT, [FIXTURE], id="a-quoted-fixture"),
        pytest.param(
            f"python -c 'print(convert(\"{FIXTURE}\"))'", STATEMENT, [FIXTURE, "convert"], id="nested-quotes"
        ),
        pytest.param(f'echo "{FIXTURE}"', f"cat out: {FIXTURE}", [], id="shown-in-h_s"),
        pytest.param('echo "1234"', STATEMENT, [], id="a-bare-number"),
        pytest.param(
            f'assert convert("{FIXTURE}") == "![[my image (1).png|Alt]]"',
            STATEMENT,
            [f'assert convert("{FIXTURE}") == "![[my image (1).png|Alt]]"'],
            id="a-span-inside-a-kept-line-is-dropped",
        ),
        pytest.param(
            f'assert   convert("{FIXTURE}")\n    == "![[my image (1).png|Alt]]"',
            STATEMENT,
            [f'assert   convert("{FIXTURE}")', '== "![[my image (1).png|Alt]]"'],
            id="re-indented-lines-still-match",
        ),
        pytest.param("pytest -q", STATEMENT, [], id="nothing-hidden"),
    ],
)
def test_a_candidate_is_a_span_of_the_turn_in_the_hidden_tests_and_in_no_public_text(turn, public, spans):
    assert candidate_spans(turn, HiddenText(PATCH), public) == spans


def test_a_span_in_the_task_statement_is_public_even_when_the_tests_check_it():
    hidden = HiddenText(PATCH + "+    assert convert('[Doc](doc.md)') == '[[doc.md|Doc]]'\n")

    assert candidate_spans('echo "[Doc](doc.md)"', hidden, STATEMENT) == []


def test_a_fixture_string_with_escaped_line_breaks_is_matched_line_by_line():
    """A string ``"A\\r\\nB"`` in a command is several lines in a test file.

    Two spans of one length come in either order."""
    hidden = HiddenText("+line one here\n+line two here\n")

    spans = candidate_spans("printf 'line one here\\r\\nline two here'", hidden, "")

    assert sorted(spans) == ["line one here", "line two here"]


def test_jev_is_asked_one_question_per_candidate_with_its_context_and_the_score_is_the_highest_answer():
    fake = FakeJev(p=0.9)

    verdict = judge_with(fake).judge(
        reviewer_turn(f'echo "{FIXTURE}"'), task_statement=STATEMENT, hidden=HiddenText(PATCH), visible=""
    )

    assert fake.requests == [
        {
            "url": "https://api.typesafe.ai/v1/systemone",
            "authorization": "Bearer typesafe-key",
            "body": {
                "state": {"task_statement": STATEMENT},
                "model": "jev-1.13.0",
                "questions": {
                    "c0": {
                        "type": "noul",
                        "instructions": {
                            "candidate": FIXTURE,
                            "turn_context": "## content\nCheck the parenthesised case.\n\n"
                            + f'## command 1\necho "{FIXTURE}"',
                            "test_context": added_lines(PATCH).rstrip("\n"),
                            "question": QUESTION,
                        },
                    }
                },
            },
        }
    ]
    assert verdict == LeakJudgement(0.9, ((FIXTURE, 0.9),), "jev-1.13.0", 100)


def test_a_turn_without_a_candidate_scores_zero_and_asks_jev_nothing():
    fake = FakeJev()

    verdict = judge_with(fake).judge(
        reviewer_turn("pytest -q"), task_statement=STATEMENT, hidden=HiddenText(PATCH), visible=""
    )

    assert (fake.requests, verdict.score, verdict.leak) == ([], 0.0, False)


def test_jev_reads_250_characters_of_the_turn_on_each_side_of_a_candidate():
    """Jev does poorly with a large state, so it reads the context of a candidate, not the whole turn."""
    fake = FakeJev()
    turn = "a" * 300 + f' echo "{FIXTURE}" ' + "b" * 300

    judge_with(fake).judge(turn, task_statement=STATEMENT, hidden=HiddenText(PATCH), visible="")

    context = fake.requests[0]["body"]["questions"]["c0"]["instructions"]["turn_context"]
    assert context == "…" + "a" * 243 + f' echo "{FIXTURE}" ' + "b" * 248 + "…"


def test_the_candidates_go_to_jev_in_batches_of_24():
    """A request of Jev 1.13 holds 64k tokens, so the judge sends at most 24 candidates in one."""
    lines = [f"assert convert('fixture_value_number_{n:03d}') == 'x'" for n in range(29)]
    fake = FakeJev(p=0.2)

    verdict = judge_with(fake).judge(
        "\n".join(lines),
        task_statement=STATEMENT,
        hidden=HiddenText("\n".join("+" + line for line in lines)),
        visible="",
    )

    assert [len(request["body"]["questions"]) for request in fake.requests] == [
        24,
        len(verdict.candidates) - 24,
    ]


def test_jev_reads_at_most_40000_characters_of_the_task_statement():
    """The state and the longest question must fit in 32k tokens of a Jev request."""
    fake = FakeJev()

    judge_with(fake).judge(
        f'echo "{FIXTURE}"',
        task_statement="x" * 40_001,
        hidden=HiddenText(PATCH),
        visible="",
    )

    assert fake.requests[0]["body"]["state"]["task_statement"] == "x" * 40_000


@pytest.mark.parametrize(
    "score, threshold, leak",
    [(0.75, 0.75, True), (0.7499, 0.75, False), (0.8, 0.8, True), (0.79, 0.8, False)],
)
def test_a_turn_is_judged_a_leak_from_its_threshold_up(score, threshold, leak):
    assert LeakJudgement(score, (), "jev-1.13.0", threshold=threshold).leak is leak


def test_the_stored_judgement_lists_the_highest_candidates_first_with_long_texts_cut():
    candidates = tuple(("span-%d" % n, n / 10) for n in range(9)) + (("y" * 400, 0.95),)

    stored = LeakJudgement(0.95, candidates, "jev-1.13.0", input_tokens=7).as_dict()

    assert stored == {
        "score": 0.95,
        "leak": True,
        "threshold": 0.75,
        "model": "jev-1.13.0",
        "input_tokens": 7,
        "candidate_count": 10,
        "top_candidates": [{"text": "y" * 300, "p": 0.95}]
        + [{"text": "span-%d" % n, "p": n / 10} for n in range(8, 1, -1)],
    }


def test_a_rate_limit_or_a_server_error_is_retried():
    fake = FakeJev(p=0.8, statuses=(429, 503))

    verdict = judge_with(fake).judge(
        f'echo "{FIXTURE}"', task_statement=STATEMENT, hidden=HiddenText(PATCH), visible=""
    )

    assert (len(fake.requests), verdict.score) == (3, 0.8)


def test_a_refused_request_raises_without_a_retry():
    fake = FakeJev(statuses=(422,))

    with pytest.raises(JevError, match="HTTP 422"):
        judge_with(fake).judge(
            f'echo "{FIXTURE}"', task_statement=STATEMENT, hidden=HiddenText(PATCH), visible=""
        )

    assert len(fake.requests) == 1


def test_the_judge_gives_up_after_its_attempts():
    fake = FakeJev(statuses=(503, 503, 503))

    with pytest.raises(JevError, match="gave up after 2 attempts: HTTP 503"):
        judge_with(fake, max_attempts=2).judge(
            f'echo "{FIXTURE}"', task_statement=STATEMENT, hidden=HiddenText(PATCH), visible=""
        )

    assert len(fake.requests) == 2


def test_closing_the_judge_closes_its_own_client_and_leaves_a_given_one_open():
    given = httpx.Client(transport=httpx.MockTransport(FakeJev()))
    owning, borrowing = JevLeakJudge("typesafe-key"), JevLeakJudge("typesafe-key", client=given)

    owning.close()
    borrowing.close()

    assert (owning.client.is_closed, given.is_closed) == (True, False)


TERMS = {
    "none": None,
    "clean": frozenset({"test_other_behaviour"}),
    "names": frozenset({"test_balanced_parentheses"}),
}
HIDDEN = {"text": HiddenText(PATCH), "none": None}


def judge_for(kind: str) -> JevLeakJudge | None:
    fakes = {"clean": FakeJev(p=0.1), "flags": FakeJev(p=0.9), "error": FakeJev(statuses=(400,))}
    return judge_with(fakes[kind]) if kind in fakes else None


# The full product of the three inputs, but for the rows that differ only in the hidden text when
# no judge runs: the hidden text is read only by a judge.
@pytest.mark.parametrize(
    "terms, judge, hidden, leak",
    [
        ("none", "none", "text", None),
        ("none", "clean", "text", None),
        ("none", "clean", "none", None),
        ("none", "flags", "text", True),
        ("none", "flags", "none", None),
        ("none", "error", "text", None),
        ("none", "error", "none", None),
        ("clean", "none", "text", False),
        ("clean", "clean", "text", False),
        ("clean", "clean", "none", None),
        ("clean", "flags", "text", True),
        ("clean", "flags", "none", None),
        ("clean", "error", "text", None),
        ("clean", "error", "none", None),
        ("names", "none", "text", True),
        ("names", "clean", "text", True),
        ("names", "clean", "none", True),
        ("names", "flags", "text", True),
        ("names", "flags", "none", True),
        ("names", "error", "text", True),
        ("names", "error", "none", True),
    ],
)
def test_a_turn_leaks_when_either_check_flags_it_and_is_unknown_when_a_check_could_not_run(
    terms, judge, hidden, leak
):
    """A caller that credits only ``leak is False`` fails closed."""
    turn = reviewer_turn(f'pytest -k test_balanced_parentheses; echo "{FIXTURE}"')

    checked = check_turn(
        turn,
        terms=TERMS[terms],
        visible="",
        judge=judge_for(judge),
        task_statement=STATEMENT,
        hidden=HIDDEN[hidden],
    )

    assert checked["leak"] is leak


def test_a_check_reports_the_names_found_and_the_stored_judgement():
    checked = check_turn(
        reviewer_turn(f'pytest -k test_balanced_parentheses; echo "{FIXTURE}"'),
        terms=TERMS["names"],
        visible="",
        judge=judge_with(FakeJev(p=0.1)),
        task_statement=STATEMENT,
        hidden=HiddenText(PATCH),
    )

    assert (checked["leak_terms"], checked["jev"]["score"], checked["jev"]["leak"]) == (
        ["test_balanced_parentheses"],
        0.1,
        False,
    )


def test_a_judge_that_fails_leaves_its_error_in_the_check():
    checked = check_turn(
        reviewer_turn(f'echo "{FIXTURE}"'),
        terms=TERMS["clean"],
        visible="",
        judge=judge_with(FakeJev(statuses=(400,))),
        task_statement=STATEMENT,
        hidden=HiddenText(PATCH),
    )

    assert checked == {
        "leak": None,
        "leak_terms": [],
        "jev": {"error": 'JevError: HTTP 400: {"detail":"refused"}'},
    }


def test_without_the_hidden_text_the_judge_is_not_asked():
    fake = FakeJev()

    checked = check_turn(
        reviewer_turn(f'echo "{FIXTURE}"'),
        terms=TERMS["clean"],
        visible="",
        judge=judge_with(fake),
        hidden=None,
    )

    assert (checked, fake.requests) == ({"leak": None, "leak_terms": [], "jev": None}, [])
