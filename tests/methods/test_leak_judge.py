"""The Jev leak judge: candidates found in code, one Noul each, and the RL filter over a sprout run."""

import json

import httpx
import pytest

from sprout.worker.jobs.review.leak import judge as leak_judge
from sprout.worker.jobs.review.leak.judge import (BATCH, HiddenText, JevError, JevLeakJudge, added_lines, candidate_spans,
                                        check_turn, main, sprout_branches)

REAL_CLIENT = httpx.Client
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


def call(identifier, command):
    return {"id": identifier, "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def reviewer_turn(command, content="Check the parenthesised case."):
    return {"role": "assistant", "content": content, "tool_calls": [call("reviewer-r1-b1-c1", command)]}


def test_added_lines_keeps_only_what_the_patch_adds():
    lines = added_lines(PATCH).split("\n")
    assert lines[0] == "def test_balanced_parentheses():"
    assert "import links" not in lines and not any(line.startswith("++") for line in lines)


def test_a_hidden_fixture_absent_from_public_text_is_a_candidate():
    turn = 'python -c \'print(convert("![Alt](<my image (1).png>)"))\''
    spans = candidate_spans(turn, HiddenText(PATCH), STATEMENT)
    assert "![Alt](<my image (1).png>)" in spans


def test_public_spans_numbers_and_contained_spans_are_not_candidates():
    hidden = HiddenText(PATCH)
    # In the task statement: public, even though the tests check it too.
    assert candidate_spans('echo "[Doc](doc.md)"', HiddenText(PATCH + "+[Doc](doc.md)\n"), STATEMENT) == []
    # A bare number is never asked about.
    assert candidate_spans("width 1234", hidden, STATEMENT) == []
    # Already in the actor's history: public.
    assert candidate_spans('"![Alt](<my image (1).png>)"', hidden, 'cat out: ![Alt](<my image (1).png>)') == []
    # One fact, one question: the quoted span is inside the kept whole line.
    spans = candidate_spans('assert convert("![Alt](<my image (1).png>)") == "![[my image (1).png|Alt]]"',
                            hidden, STATEMENT)
    assert spans[0].startswith("assert convert(") and not any(s == "![Alt](<my image (1).png>)" for s in spans)


def test_whitespace_differences_still_match():
    spans = candidate_spans('assert   convert("![Alt](<my image (1).png>)")\n    == "![[my image (1).png|Alt]]"',
                            HiddenText(PATCH), STATEMENT)
    assert spans


class FakeJev:
    """Answers every candidate with ``p``, or the statuses queued first."""

    def __init__(self, p=0.9, statuses=()):
        self.p, self.statuses, self.requests = p, list(statuses), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append({"headers": dict(request.headers), "body": body})
        if self.statuses:
            return httpx.Response(self.statuses.pop(0), json={"detail": "x"}, headers={"retry-after": "0"})
        return httpx.Response(200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 100},
                                         "answers": {k: {"type": "noul", "noul": self.p} for k in body["questions"]}})


def judge_with(fake, **kwargs):
    return JevLeakJudge("k", client=httpx.Client(transport=httpx.MockTransport(fake)), **kwargs)


def test_judge_asks_one_noul_per_candidate_and_scores_the_maximum(monkeypatch):
    fake = FakeJev(p=0.9)
    verdict = judge_with(fake).judge(reviewer_turn('echo "![Alt](<my image (1).png>)"'), task_statement=STATEMENT,
                                     hidden=HiddenText(PATCH), visible="")
    (request,) = fake.requests
    assert request["headers"]["authorization"] == "Bearer k"
    assert request["body"]["model"] == leak_judge.MODEL and request["body"]["state"]["task_statement"] == STATEMENT
    question = next(iter(request["body"]["questions"].values()))
    assert question["type"] == "noul" and question["instructions"]["candidate"] == "![Alt](<my image (1).png>)"
    assert "my image (1).png" in question["instructions"]["test_context"]
    assert verdict.score == pytest.approx(0.9) and verdict.leak and verdict.model == "jev-1.13.0"


def test_a_turn_without_candidates_scores_zero_without_a_request():
    fake = FakeJev()
    verdict = judge_with(fake).judge(reviewer_turn("pytest -q"), task_statement=STATEMENT,
                                     hidden=HiddenText(PATCH), visible="")
    assert fake.requests == [] and verdict.score == 0.0 and not verdict.leak


def test_many_candidates_are_batched(monkeypatch):
    lines = [f"assert convert('fixture_value_number_{n:03d}') == 'x'" for n in range(BATCH + 5)]
    patch = "\n".join("+" + line for line in lines)
    fake = FakeJev(p=0.2)
    verdict = judge_with(fake).judge("\n".join(lines), task_statement=STATEMENT, hidden=HiddenText(patch), visible="")
    assert len(fake.requests) == 2 and all(len(r["body"]["questions"]) <= BATCH for r in fake.requests)
    assert len(verdict.candidates) == sum(len(r["body"]["questions"]) for r in fake.requests)
    assert not verdict.leak


def test_rate_limits_and_server_errors_are_retried_and_a_bad_request_is_not(monkeypatch):
    monkeypatch.setattr(leak_judge.time, "sleep", lambda s: None)
    fake = FakeJev(p=0.8, statuses=[429, 503])
    turn = reviewer_turn('echo "![Alt](<my image (1).png>)"')
    verdict = judge_with(fake).judge(turn, task_statement=STATEMENT, hidden=HiddenText(PATCH), visible="")
    assert len(fake.requests) == 3 and verdict.score == pytest.approx(0.8)
    with pytest.raises(JevError, match="HTTP 422"):
        judge_with(FakeJev(statuses=[422])).judge(turn, task_statement=STATEMENT, hidden=HiddenText(PATCH), visible="")


# -- the filter over a sprout run ------------------------------------------------------------


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))


def sprout_run(tmp_path, *, command, resolved=True):
    """A one-task sprout run: a parent graded in a baseline root, one resolved round-1 branch."""
    tasks = tmp_path / "tasks"
    write(tasks / "links" / "instruction.md", STATEMENT)
    write(tasks / "links" / "tests" / "test.patch", PATCH)
    write(tasks / "links" / "tests" / "config.json", {"f2p_node_ids": ["tests/test_links.py::test_balanced_parentheses"]})
    parent = tmp_path / "base" / "tasks" / "links" / "baseline" / "attempt-1" / "trials" / "links-parent"
    write(parent / "verifier" / "test-stdout.txt", "FAILED tests/test_links.py::test_balanced_parentheses")
    root = tmp_path / "run"
    trial = root / "tasks" / "links" / "sprout" / "trials" / "links-r1b1-fix"
    write(trial / "result.json", {"verifier_result": {"rewards": {"reward": 1.0 if resolved else 0.0}}})
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": STATEMENT}]
    write(trial / "agent" / "mini-native" / "x.traj.json",
          {"messages": history + [reviewer_turn(command), {"role": "tool", "content": "ok"}]})
    write(root / "tasks" / "links" / "sprout" / "summary.json", {"attempts": [
        {"name": "parent", "round": 0, "journal": str(parent / "agent" / "trajectory.jsonl")},
        {"name": "r1b1-fix", "round": 1, "journal": str(trial / "agent" / "trajectory.jsonl")}]})
    return root, tasks


def test_sprout_branches_read_what_the_reviewer_could_read(tmp_path):
    root, tasks = sprout_run(tmp_path, command='echo "![Alt](<my image (1).png>)"')
    (branch,) = sprout_branches(str(root), str(tasks))
    assert branch.task == "links" and branch.round == 1 and branch.resolved
    assert branch.turn["tool_calls"][0]["id"].startswith("reviewer-r")
    assert branch.hidden.fail_to_pass == ("tests/test_links.py::test_balanced_parentheses",)
    assert "FAILED" in branch.hidden.verifier_output and STATEMENT in branch.visible
    assert list(sprout_branches(str(sprout_run(tmp_path / "u", command="x", resolved=False)[0]),
                                str(tasks))) == []


@pytest.mark.parametrize("p, eligible", [(0.95, False), (0.1, True)])
def test_filter_marks_a_branch_eligible_only_when_both_checks_are_clean(tmp_path, monkeypatch, p, eligible):
    root, tasks = sprout_run(tmp_path, command='echo "![Alt](<my image (1).png>)"')
    fake = FakeJev(p=p)
    monkeypatch.setattr(leak_judge.httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(fake)))
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    out = tmp_path / "out.jsonl"
    main([str(root), "--tasks-dir", str(tasks), "-o", str(out)])
    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["leak_terms"] == [] and row["jev"]["candidate_count"] >= 1
    assert row["eligible"] is eligible and row["jev"]["leak"] is (not eligible)


def test_filter_fails_closed_when_the_judge_errors_or_a_name_leaks(tmp_path, monkeypatch):
    monkeypatch.setattr(leak_judge.time, "sleep", lambda s: None)
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    root, tasks = sprout_run(tmp_path, command='echo "![Alt](<my image (1).png>)"')
    fake = FakeJev(statuses=[400])
    monkeypatch.setattr(leak_judge.httpx, "Client", lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(fake)))
    out = tmp_path / "out.jsonl"
    main([str(root), "--tasks-dir", str(tasks), "-o", str(out)])
    row = json.loads(out.read_text())
    assert "error" in row["jev"] and row["eligible"] is False
    # A hidden test name in the turn: the lexical gate alone makes it ineligible.
    root2, _ = sprout_run(tmp_path / "n", command="pytest tests/test_links.py::test_balanced_parentheses")
    monkeypatch.setattr(leak_judge.httpx, "Client",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(FakeJev(p=0.0))))
    main([str(root2), "--tasks-dir", str(tasks), "-o", str(out)])
    row = json.loads(out.read_text())
    assert row["leak_terms"] and row["eligible"] is False


def test_gate_jev_records_leak_terms_without_gating_on_them(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    root, tasks = sprout_run(tmp_path, command="pytest tests/test_links.py::test_balanced_parentheses")
    monkeypatch.setattr(leak_judge.httpx, "Client",
                        lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(FakeJev(p=0.0))))
    out = tmp_path / "out.jsonl"
    main([str(root), "--tasks-dir", str(tasks), "-o", str(out), "--gate", "jev"])
    row = json.loads(out.read_text())
    assert row["leak_terms"] and row["eligible"] is True


def test_check_turn_flags_on_either_check_and_is_unknown_when_one_could_not_run(monkeypatch):
    monkeypatch.setattr(leak_judge.time, "sleep", lambda s: None)
    hidden, terms = HiddenText(PATCH), frozenset({"test_balanced_parentheses"})
    fixture = reviewer_turn('echo "![Alt](<my image (1).png>)"')
    plain = reviewer_turn("pytest -q")

    def jev(p):
        return judge_with(FakeJev(p=p))

    flagged = check_turn(fixture, terms=terms, visible="", judge=jev(0.9), task_statement=STATEMENT, hidden=hidden)
    assert flagged["leak"] is True and flagged["leak_terms"] == [] and flagged["jev"]["leak"] is True
    named = check_turn(reviewer_turn("pytest -k test_balanced_parentheses"), terms=terms, visible="",
                       judge=jev(0.0), task_statement=STATEMENT, hidden=hidden)
    assert named["leak"] is True and named["leak_terms"] == ["test_balanced_parentheses"]
    clean = check_turn(plain, terms=terms, visible="", judge=jev(0.9), task_statement=STATEMENT, hidden=hidden)
    assert clean["leak"] is False and clean["jev"]["candidate_count"] == 0
    # A check that could not run makes an unflagged turn unknown, never clean.
    assert check_turn(plain, terms=None, visible="")["leak"] is None
    assert check_turn(plain, terms=terms, visible="", judge=jev(0.1), hidden=None)["leak"] is None
    errored = check_turn(fixture, terms=terms, visible="", judge=judge_with(FakeJev(statuses=[400])),
                         task_statement=STATEMENT, hidden=hidden)
    assert errored["leak"] is None and "HTTP 400" in errored["jev"]["error"]
    # Without a judge the lexical check alone decides.
    assert check_turn(plain, terms=terms, visible="")["leak"] is False


def test_a_task_without_a_test_patch_gives_its_tests_files_and_harness_names_are_no_terms(tmp_path):
    from sprout.worker.jobs.review.leak.terms import hidden_terms
    from sprout.worker.jobs.review.leak.judge import patch_files, task_hidden_tests
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("pytest /tests/test_outputs.py\n")
    (tests / "test_outputs.py").write_text("def test_secret_output():\n    assert read() == 'SECRET-42'\n")
    (tests / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    (tests / "config.json").write_text(json.dumps({"FAIL_TO_PASS": json.dumps(["tests/test_outputs.py::test_x"])}))
    patch, fail_to_pass = task_hidden_tests(tmp_path)
    assert "SECRET-42" in added_lines(patch) and "pytest /tests/test_outputs.py" in added_lines(patch)
    assert fail_to_pass == ("tests/test_outputs.py::test_x",), "a JSON-encoded list is decoded, not split"
    terms = hidden_terms(fail_to_pass=list(fail_to_pass), test_patch_paths=patch_files(patch))
    assert "test_outputs.py" in terms and "test.sh" not in terms and "tests/test.sh" not in terms
    # A turn copying the hidden expected value is a candidate; one writing its own test.sh is not.
    assert any("SECRET-42" in span for span in candidate_spans("assert read() == 'SECRET-42'", HiddenText(patch),
                                                               "task"))
    assert candidate_spans("cat > /app/test.sh <<EOF", HiddenText(patch), "task") == []
    assert task_hidden_tests(tmp_path / "missing") == ("", ())


def test_patch_files_names_both_sides_of_a_rename():
    from sprout.worker.jobs.review.leak.judge import patch_files
    patch = "diff --git a/tests/old_test.py b/tests/test_hidden_new.py\nrename from tests/old_test.py\n"
    assert patch_files(patch) == ["tests/old_test.py", "tests/test_hidden_new.py"]
