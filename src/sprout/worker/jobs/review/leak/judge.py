"""Whether a reviewer-written turn copies hidden-test content, judged by Jev.

``worker.jobs.review.leak.terms`` finds the hidden test *names* a turn gives away. A turn can
also copy what the tests check: a fixture string, an input, an expected value,
an exact assertion message. Those are what the audits of SPROUT's DeepSWE runs
found in the turns that leaked, and no list of names catches them.

The judgment is split the way Jev (TypeSafe's System One model,
https://docs.typesafe.ai) is good at it. Code finds the candidates
(:func:`candidate_spans`): the spans of the turn -- quoted strings,
identifiers, whole lines -- that occur in the hidden text (the test patch's
added lines, the FAIL_TO_PASS ids, the verifier output the reviewer read) and
in no public text (the task statement, the history the actor kept at the
point). Jev then answers one narrow question per candidate, with the span's
context in the turn and in the tests: did the reviewer copy specific content
from the hidden tests? A turn's score is its highest candidate's probability;
a turn with no candidate scores 0. Bare numbers are not candidates: Jev reads
numbers poorly, and a value that matters is usually part of a longer span.

The prompt and the threshold were tuned against an agent audit (Sonnet 5.5
judges, prosecutors and refuters, the disagreements re-adjudicated by Opus 5.5
and Sonnet 5.5) of 213 resolved branches of DeepSWE runs; see
``docs/LEAK_JUDGE.md``.

    python -m sprout.worker.jobs.review.leak.judge <run root> [<run root> ...] \\
        --tasks-dir ~/datasets/deep-swe/tasks -o leak-judgements.jsonl

writes one line per resolved branch of a ``eval.evaluate --method sprout``
run: its score, the candidates that scored, the lexical ``leak_terms``, and
whether it may enter RL (``eligible``).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from sprout.worker.jobs.review.leak.terms import hidden_terms, leak_terms, visible_text

URL = "https://api.typesafe.ai/v1/systemone"
#: Pinned, not ``jev-latest``: the threshold is calibrated against this version.
MODEL = "jev-1.13.0"
KEY_ENV = "TYPESAFE_API_KEY"
#: A turn at or above this score is judged to copy hidden-test content.
THRESHOLD = 0.75
#: A judge inside a review, which runs under its own deadline, gives up on a
#: request after this many tries of this long, and the turn stays unchecked.
REVIEW_TIMEOUT_S, REVIEW_ATTEMPTS = 30.0, 3

#: The question asked of every candidate. Its data fields are the candidate,
#: its context in the turn and its context in the hidden tests.
QUESTION = (
    "`candidate` was written by a reviewer (`turn_context`) and also appears in the hidden tests "
    "(`test_context`), which the coding agent never saw. Does `candidate` copy specific content from the "
    "hidden tests: a test input or fixture, an expected output or value, an exact error or assertion "
    "message, or a hidden test's name?"
)
CONTEXT_CHARS = 250
#: Jev 1.13 takes 64k tokens per request and 32k for the state plus the
#: longest question. A request is cut into batches of at most this many
#: candidates, each well inside both.
BATCH = 24
MAX_STATEMENT_CHARS = 40_000

#: One pattern per quote character, each run on its own: a shell command
#: nests them (``python -c 'print(f("x"))'``), and a single alternation would
#: consume the outer quotes and never see the inner string.
_QUOTED = (
    re.compile(r'"((?:[^"\\\n]|\\.){3,200})"'),
    re.compile(r"'((?:[^'\\\n]|\\.){3,200})'"),
    re.compile(r"`([^`\n]{3,200})`"),
)
_ESCAPE = re.compile(r"\\([\"'\\])")
#: A fixture written as one string with escaped line breaks ("A\r\nB\r\n")
#: is several lines in a test file: each piece is a span of its own.
_ESCAPED_BREAK = re.compile(r"(?:\\r)?\\n|\\t")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_\-./:]{5,}")
_NUMERIC = re.compile(r"^-?[\d.e]+$")
_SPACE = re.compile(r"\s+")
_EDGES = " \t'\"`;,"


def _norm(text: str) -> str:
    return _SPACE.sub(" ", text).strip()


def added_lines(test_patch: str) -> str:
    """The lines a patch adds, without their ``+``: what the hidden tests are."""
    return "\n".join(
        line[1:] for line in test_patch.split("\n") if line.startswith("+") and not line.startswith("+++")
    )


@dataclass(frozen=True)
class HiddenText:
    """What the reviewer could read and the actor could not."""

    test_patch: str
    fail_to_pass: tuple[str, ...] = ()
    verifier_output: str = ""

    def text(self) -> str:
        return "\n".join([added_lines(self.test_patch), "\n".join(self.fail_to_pass), self.verifier_output])


def _atoms(text: str) -> set[str]:
    found = set()
    for pattern in _QUOTED:
        for match in pattern.finditer(text):
            inner = match.group(1)
            found.add(inner)
            found.add(_ESCAPE.sub(r"\1", inner))  # "f(\"x\")" inside a shell string is f("x") in the file
            pieces = _ESCAPED_BREAK.split(inner)
            if len(pieces) > 1:
                found.update(piece for piece in pieces if len(piece.strip()) >= 6)
    for match in _WORD.finditer(text):
        found.add(match.group(0).strip("./:-"))
    for line in text.split("\n"):
        line = line.strip()
        if 16 <= len(line) <= 300:
            found.add(line)
    return {atom for atom in found if len(atom.strip(_EDGES)) >= 3 and not _NUMERIC.match(atom.strip(_EDGES))}


def candidate_spans(turn: str, hidden: HiddenText, public: str) -> list[str]:
    """The spans of ``turn`` found in ``hidden`` and nowhere in ``public``, longest first.

    Whitespace is normalised on both sides, so a span re-indented or re-wrapped
    still matches, and a span is public when its core (without surrounding
    quotes and separators) is. A span inside a longer kept span is dropped:
    one fact, one question.
    """
    hidden_n, public_n = _norm(hidden.text()), _norm(public)
    found = sorted(
        (
            atom
            for atom in _atoms(turn)
            if (n := _norm(atom))
            and n in hidden_n
            and n not in public_n
            and _norm(atom.strip(_EDGES)) not in public_n
        ),
        key=len,
        reverse=True,
    )
    kept: list[str] = []
    for atom in found:
        if not any(_norm(atom) in _norm(longer) for longer in kept):
            kept.append(atom)
    return kept


def _window(text: str, needle: str, width: int = CONTEXT_CHARS) -> str:
    at = text.find(needle)
    if at < 0:
        at = text.find(_norm(needle).split(" ")[0])
    if at < 0:
        return ""
    low, high = max(0, at - width), min(len(text), at + len(needle) + width)
    return ("…" if low else "") + text[low:high] + ("…" if high < len(text) else "")


@dataclass(frozen=True)
class LeakJudgement:
    """One turn's judgment: ``score`` is the highest candidate probability (0 with none)."""

    score: float
    candidates: tuple[tuple[str, float], ...]
    model: str | None
    input_tokens: int = 0
    threshold: float = THRESHOLD

    @property
    def leak(self) -> bool:
        return self.score >= self.threshold

    def as_dict(self, top: int = 8) -> dict:
        ranked = sorted(self.candidates, key=lambda c: -c[1])
        return {
            "score": round(self.score, 4),
            "leak": self.leak,
            "threshold": self.threshold,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "candidate_count": len(self.candidates),
            "top_candidates": [{"text": text[:300], "p": round(p, 4)} for text, p in ranked[:top]],
        }


class JevError(RuntimeError):
    pass


def readable_turn(turn: dict) -> str:
    """A turn as Jev reads it: its content, then each bash command decoded.

    Not :func:`worker.jobs.review.leak.terms.turn_text`, which also keeps the raw JSON
    arguments for the lexical check: there a span's first occurrence is
    JSON-escaped, and the context Jev is shown would be too.
    """
    parts = ["## content\n" + (turn.get("content") or "")]
    if turn.get("reasoning_content"):
        parts.append("## reasoning\n" + turn["reasoning_content"])
    for n, call in enumerate(turn.get("tool_calls") or (), 1):
        arguments = (call.get("function") or {}).get("arguments")
        try:
            decoded = json.loads(arguments) if isinstance(arguments, str) else arguments
            command = decoded.get("command") if isinstance(decoded, dict) else None
        except ValueError:
            command = None
        parts.append(f"## command {n}\n" + (command if isinstance(command, str) else str(arguments)))
    return "\n\n".join(parts)


def requests_for(
    turn: dict | str, *, task_statement: str, hidden: HiddenText, visible: str, model: str = MODEL
) -> list[tuple[list[str], dict]]:
    """The Jev requests that judge ``turn``: ``(candidates, payload)`` per batch, none without candidates."""
    text = turn if isinstance(turn, str) else readable_turn(turn)
    spans = candidate_spans(text, hidden, task_statement + "\n" + visible)
    tests = added_lines(hidden.test_patch)
    state = {"task_statement": task_statement[:MAX_STATEMENT_CHARS]}
    batches = []
    for start in range(0, len(spans), BATCH):
        batch = spans[start : start + BATCH]
        questions = {
            f"c{n}": {
                "type": "noul",
                "instructions": {
                    "candidate": span,
                    "turn_context": _window(text, span),
                    "test_context": _window(tests, span) or "(in the hidden tests' grading output)",
                    "question": QUESTION,
                },
            }
            for n, span in enumerate(batch)
        }
        batches.append((batch, {"state": state, "model": model, "questions": questions}))
    return batches


def combine(
    batches: list[tuple[list[str], dict]], bodies: list[dict], threshold: float = THRESHOLD
) -> LeakJudgement:
    """One judgment from the answers to :func:`requests_for`'s batches."""
    scored: list[tuple[str, float]] = []
    model, tokens = None, 0
    for (spans, _), body in zip(batches, bodies, strict=True):
        model, tokens = body.get("model"), tokens + int((body.get("usage") or {}).get("input_tokens") or 0)
        scored += [(span, float(body["answers"][f"c{n}"]["noul"])) for n, span in enumerate(spans)]
    return LeakJudgement(max((p for _, p in scored), default=0.0), tuple(scored), model, tokens, threshold)


@dataclass
class JevLeakJudge:
    """Asks Jev about a turn's candidates. ``client`` is injectable for tests."""

    key: str
    model: str = MODEL
    threshold: float = THRESHOLD
    url: str = URL
    timeout_s: float = 120.0
    max_attempts: int = 6
    client: httpx.Client | None = None
    _owned: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            self.client, self._owned = httpx.Client(timeout=self.timeout_s), True

    def close(self) -> None:
        if self._owned and self.client is not None:
            self.client.close()

    def judge(
        self, turn: dict | str, *, task_statement: str, hidden: HiddenText, visible: str
    ) -> LeakJudgement:
        """Judge one turn (a message dict, or its text) against ``hidden``, ``visible`` being h_s."""
        batches = requests_for(
            turn, task_statement=task_statement, hidden=hidden, visible=visible, model=self.model
        )
        return combine(batches, [self._post(payload) for _, payload in batches], self.threshold)

    def _post(self, payload: dict) -> dict:
        """One request, retried on 429, 5xx and dropped connections; any other status is an error."""
        reason = ""
        for attempt in range(self.max_attempts):
            try:
                response = self.client.post(
                    self.url, json=payload, headers={"Authorization": f"Bearer {self.key}"}
                )
            except httpx.TransportError as error:
                reason = f"{type(error).__name__}: {error}"
                time.sleep(min(30, 2**attempt))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                reason = f"HTTP {response.status_code}"
                time.sleep(_retry_after(response, attempt))
                continue
            if response.status_code != 200:
                raise JevError(f"HTTP {response.status_code}: {response.text[:500]}")
            return response.json()
        raise JevError(f"gave up after {self.max_attempts} attempts: {reason}")


def check_turn(
    turn: dict,
    *,
    terms: frozenset[str] | None,
    visible: str,
    judge: JevLeakJudge | None = None,
    task_statement: str = "",
    hidden: HiddenText | None = None,
) -> dict:
    """Both leak checks of one turn, as ``branching.review``'s ``turn_check`` reports them.

    ``leak_terms`` is ``worker.jobs.review.leak.terms``'s (None without a vocabulary), ``jev``
    this module's judgment (None without a judge or hidden text; an ``error``
    when the call failed). ``leak`` is True when either check flags the turn,
    None when neither does but one could not run, else False: a caller that
    credits only ``leak is False`` fails closed.
    """
    found = None if terms is None else leak_terms(turn, terms, visible)
    jev = None
    if judge is not None and hidden is not None:
        try:
            jev = judge.judge(turn, task_statement=task_statement, hidden=hidden, visible=visible).as_dict()
        except (
            Exception
        ) as error:  # an unreachable judge or a reply of the wrong shape: unchecked, never fatal
            jev = {"error": f"{type(error).__name__}: {error}"[:500]}
    if found or (jev is not None and jev.get("leak") is True):
        leak = True
    elif found is None or (judge is not None and (jev is None or "error" in jev)):
        leak = None
    else:
        leak = False
    return {"leak": leak, "leak_terms": found, "jev": jev}


def _retry_after(response: httpx.Response, attempt: int) -> float:
    try:
        return min(60.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return float(min(30, 2**attempt))


def read_key_env(variable: str = KEY_ENV) -> str:
    """The TypeSafe key in ``$variable``."""
    key = os.environ.get(variable, "").strip()
    if not key:
        raise ValueError(f"no TypeSafe key in ${variable}")
    return key


def read_key(key_file: str | None = None) -> str:
    """The TypeSafe key: ``key_file`` when given, else ``$TYPESAFE_API_KEY``."""
    if key_file:
        return Path(key_file).expanduser().read_text().strip()
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise SystemExit(f"no TypeSafe key: set {KEY_ENV} or pass --key-file")
    return key


# -- a eval.evaluate sprout run, branch by branch -------------------------------------------

_ROUND = re.compile(r"-r(\d+)b\d+-")
REVIEWER_CALL = "reviewer-r"


def _load(path: str):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def _resolved(trial: str) -> bool:
    result = _load(f"{trial}/result.json") or {}
    return bool(((result.get("verifier_result") or {}).get("rewards") or {}).get("reward"))


def _verifier_output(trial: str) -> str:
    path = f"{trial}/verifier/test-stdout.txt"
    return open(path, errors="replace").read()[-60_000:] if os.path.exists(path) else ""


def _trial_of(journal: str) -> str:
    """The trial directory of an attempt's ``agent/trajectory.jsonl``."""
    return str(Path(journal).parent.parent)


@dataclass(frozen=True)
class Branch:
    run: str
    task: str
    trial: str
    round: int
    resolved: bool
    turn: dict
    visible: str
    task_statement: str
    hidden: HiddenText


def sprout_branches(root: str, tasks_dir: str, *, resolved_only: bool = True) -> Iterable[Branch]:
    """Every branch of a sprout run whose history holds a reviewer-inserted turn.

    The hidden text of a branch of round r is the task's test patch and
    FAIL_TO_PASS ids and the verifier output of every attempt graded before
    round r (the parent's and earlier rounds'), which is what the reviewer read.
    """
    for summary_path in sorted(glob.glob(f"{root}/tasks/*/sprout/summary.json")):
        task = summary_path.split("/tasks/")[1].split("/")[0]
        summary = _load(summary_path) or {}
        task_dir = Path(tasks_dir) / task
        statement = (task_dir / "instruction.md").read_text(errors="replace")
        test_patch, fail_to_pass = task_hidden_tests(task_dir)
        graded = [
            (a.get("round") or 0, _trial_of(a["journal"]))
            for a in summary.get("attempts", [])
            if a.get("journal")
        ]
        for trial in sorted(glob.glob(f"{root}/tasks/{task}/sprout/trials/*")):
            match = _ROUND.search(os.path.basename(trial))
            resolved = _resolved(trial)
            if not match or (resolved_only and not resolved):
                continue
            natives = glob.glob(f"{trial}/agent/mini-native/*.traj.json")
            if not natives:
                continue
            messages = (_load(natives[0]) or {}).get("messages") or []
            at = next(
                (
                    i
                    for i, m in enumerate(messages)
                    if m.get("role") == "assistant"
                    and any(str(c.get("id", "")).startswith(REVIEWER_CALL) for c in m.get("tool_calls") or ())
                ),
                None,
            )
            if at is None:
                continue
            round_no = int(match.group(1))
            verdicts = "\n".join(_verifier_output(t) for r, t in graded if r < round_no)
            yield Branch(
                os.path.basename(root.rstrip("/")),
                task,
                trial,
                round_no,
                resolved,
                messages[at],
                visible_text(messages[:at]),
                statement,
                HiddenText(test_patch, fail_to_pass, verdicts),
            )


def patch_files(test_patch: str) -> list[str]:
    """Every file a ``diff --git`` patch names, old and new side (a rename names both)."""
    return sorted(
        {name for pair in re.findall(r"^diff --git a/(\S+) b/(\S+)", test_patch, re.M) for name in pair}
    )


#: Larger files under a task's tests/ are fixtures or binaries, not test code to quote.
MAX_TEST_FILE_BYTES = 1 << 20
#: Harbor's own files under a task's tests/: every task has them, so their
#: names say nothing about its tests (an agent writes a test.sh of its own).
#: Their content is still hidden text; their names are not leak terms.
HARNESS_FILES = frozenset(
    {"test.sh", "run-tests.sh", "run_tests.sh", "verify.sh", "config.json", "Dockerfile"}
)


def _fail_to_pass(value) -> tuple[str, ...]:
    """FAIL_TO_PASS as a tuple of ids; SWE-bench metadata keeps it as a JSON-encoded list."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = [value]
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value)


def task_hidden_tests(task_dir: Path) -> tuple[str, tuple[str, ...]]:
    """A Harbor task's hidden tests as a patch, and its FAIL_TO_PASS ids.

    ``tests/test.patch`` when the task has one (DeepSWE). Otherwise every text
    file under ``tests/`` written as an added file, so the same candidate search
    reads it (TerminalBench keeps its tests as files); a ``HARNESS_FILES`` one
    without the ``diff --git`` header ``patch_files`` reads, so its name is no
    leak term. FAIL_TO_PASS comes from ``tests/config.json`` (``f2p_node_ids`` or
    ``FAIL_TO_PASS``) when present. A task with neither gives ``("", ())``.
    """
    tests = Path(task_dir) / "tests"
    config = _load(str(tests / "config.json")) or {}
    fail_to_pass = _fail_to_pass(config.get("f2p_node_ids") or config.get("FAIL_TO_PASS") or ())
    if (tests / "test.patch").is_file():
        return (tests / "test.patch").read_text(errors="replace"), fail_to_pass
    parts = []
    for path in sorted(p for p in tests.rglob("*") if p.is_file()) if tests.is_dir() else ():
        if path.stat().st_size > MAX_TEST_FILE_BYTES:
            continue
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            continue
        name = path.relative_to(tests.parent).as_posix()
        header = (
            f"+++ b/{name}\n"
            if path.name in HARNESS_FILES
            else f"diff --git a/{name} b/{name}\n+++ b/{name}\n"
        )
        parts.append(header + "\n".join("+" + line for line in text.split("\n")))
    return "\n".join(parts), fail_to_pass


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Judge the reviewer-inserted turns of sprout runs for copied "
        "hidden-test content, and mark which branches may enter RL."
    )
    parser.add_argument("roots", nargs="+", help="eval.evaluate / examples.qwen.run sprout run roots")
    parser.add_argument("--tasks-dir", required=True, help="the dataset's task directories")
    parser.add_argument("-o", "--output", required=True, help="JSONL, one line per branch")
    parser.add_argument("--key-file", help=f"TypeSafe key file (default: ${KEY_ENV})")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--all-branches", action="store_true", help="judge unresolved branches too")
    parser.add_argument(
        "--gate",
        choices=["both", "jev"],
        default="both",
        help="both (default): eligible only when neither Jev nor leak_terms flags the turn, the "
        "RL driver's rule for names; jev: leak_terms is recorded but does not gate",
    )
    args = parser.parse_args(argv)
    judge = JevLeakJudge(read_key(args.key_file), model=args.model, threshold=args.threshold)
    counts = {"branches": 0, "leak": 0, "leak_terms": 0, "eligible": 0, "errors": 0}
    try:
        with open(args.output, "w") as out:
            for root in args.roots:
                for branch in sprout_branches(root, args.tasks_dir, resolved_only=not args.all_branches):
                    terms = leak_terms(
                        branch.turn,
                        hidden_terms(
                            fail_to_pass=list(branch.hidden.fail_to_pass),
                            test_patch_paths=patch_files(branch.hidden.test_patch),
                        ),
                        branch.visible,
                    )
                    row = {
                        "run": branch.run,
                        "task": branch.task,
                        "trial": branch.trial,
                        "round": branch.round,
                        "resolved": branch.resolved,
                        "leak_terms": terms,
                    }
                    try:
                        verdict = judge.judge(
                            branch.turn,
                            task_statement=branch.task_statement,
                            hidden=branch.hidden,
                            visible=branch.visible,
                        )
                        row["jev"] = verdict.as_dict()
                        # Fail closed: eligible only when the checks ran and found nothing.
                        row["eligible"] = (
                            branch.resolved and not verdict.leak and (args.gate == "jev" or not terms)
                        )
                        counts["leak"] += verdict.leak
                    except JevError as error:
                        row["jev"], row["eligible"] = {"error": str(error)}, False
                        counts["errors"] += 1
                    counts["branches"] += 1
                    counts["leak_terms"] += bool(terms)
                    counts["eligible"] += row["eligible"]
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
    finally:
        judge.close()
    print(json.dumps(counts), file=sys.stderr)


if __name__ == "__main__":
    main()
