# Leak judge: does a reviewer turn copy hidden-test content?

`src/sprout/methods/leak_judge.py` decides, for each reviewer-inserted turn of a
SPROUT run, whether it carries content the actor could only have had from the
hidden tests, and writes which branches may enter RL.

`methods.leak` (`leak_terms`) already catches hidden test **names**. The audits
of the DeepSWE runs showed that the turns that leak mostly copy test **content**
instead: a fixture string (`![Alt](<my image (1).png>)`), an input
(`margin: not-a-valid-value`), an expected message
(`/*! [NoSuchType.Foo]: type "NoSuchType" not found in current package */`), a
VTIMEZONE block at `+0300`. No list of names catches those.

## Method

Jev (TypeSafe's System One model, `jev-1.13.0`, https://docs.typesafe.ai) returns
calibrated yes/no probabilities, reads literally, does poorly with large
irrelevant state, numbers and multi-hop questions. So the work is split:

1. **Candidates, in code** (`candidate_spans`). Spans of the turn -- quoted
   strings (each quote character matched on its own, so the inner string of
   `python -c '...("x")'` is seen; escaped quotes undone; a string with `\r\n`
   inside cut into its lines), identifiers, whole lines -- that occur in the
   hidden text (the test patch's added lines, FAIL_TO_PASS, the verifier output
   the reviewer read) and in no public text (task statement, h_s), after
   whitespace normalisation. Bare numbers are dropped. A span inside a longer
   kept span is dropped.
2. **One Noul per candidate, in Jev.** State: the task statement. Question: the
   candidate, 250 characters of its context in the turn and in the tests, and
   "Does `candidate` copy specific content from the hidden tests: a test input or
   fixture, an expected output or value, an exact error or assertion message, or
   a hidden test's name?" Batches of 24 candidates per request.
3. **Branch score** = the highest candidate probability (0 with no candidate);
   `leak` when the score is at least `THRESHOLD` (0.75).

The RL filter (`python -m sprout.methods.leak_judge`) marks a resolved branch
`eligible` only when Jev ran and did not flag it and, with `--gate both` (the
default, the RL driver's rule), `leak_terms` is empty too. A Jev error makes a
branch ineligible.

## Calibration

Labels: the 213 resolved branches of five DeepSWE sprout runs (Qwen3.6-35B-A3B
actor; Opus 5.5 and Qwen reviewers; 4+3 and 8x1 schedules), each audited by a
Sonnet 5.5 judge with a prosecutor and two refuters, some checked by hand. The
39 branches where an early Jev prompt and the audit disagreed most were
re-adjudicated by Opus 5.5 and Sonnet 5.5 independently, each grepping the
public sources for every flagged span: 27 audit labels held, 3 flipped (two to
leak, one to not leak), 6 split (left out). Final: 17 leak, 190 not leak.

| Prompt / extraction | AUC |
|---|---|
| v1: one "is this a test-specific detail" Noul | 0.92 (dev, raw labels) |
| v2: + "would an engineer write this without the tests" Noul | 0.89 -- that question alone scores 0.60 |
| v2 leak question only (bare numbers dropped) | 0.93 (dev), 0.87 (all, raw labels) |
| same, adjudicated labels | 0.90 |
| + nested quotes, escaped line breaks (this module) | **0.955** |

At `THRESHOLD = 0.75` on the adjudicated labels: 16 of 17 leaks flagged, 20 of
190 non-leaks (11%) flagged. At 0.80: 14/17 and 12/190. The missed leak's only
evidence is a `### B` heading. The threshold was chosen on the same 207
branches (the held-out split has three leaks), so treat these as in-sample.
The non-leaks Jev flags are mostly turns that write their own example values
which happen to match the tests (`myArray[1:3]`, cookie strings); both
adjudicators called these not leaks.

On the same runs, `leak_terms` flags 5 of the 17 leaks (all also flagged by Jev)
and 26 of 190 non-leaks; with `--gate both`, 155 of 213 branches are eligible.

Cost: 0.59 M input tokens for all 213 branches (1111 candidates), $0.025.

## Usage

```bash
python -m sprout.methods.leak_judge runs/<sprout run> [runs/<another> ...] \
    --tasks-dir ~/datasets/deep-swe/tasks --key-file ~/.typesafe_key \
    -o leak-judgements.jsonl          # --gate jev to not gate on leak_terms
```

One JSON line per resolved branch: `run`, `task`, `trial`, `round`, `leak_terms`,
`jev` (`score`, `leak`, `threshold`, `model`, `candidate_count`,
`top_candidates`), `eligible`. In code: `JevLeakJudge(key).judge(turn,
task_statement=..., hidden=HiddenText(test_patch, fail_to_pass, verifier_output),
visible=h_s)`.

## In the RL driver's review

With `review.leak_judge` in the adapter config (`{"threshold", "model",
"resamples", "api_key_env"}`, all optional; the key lives in the review
worker's environment, `TYPESAFE_API_KEY` by default), the review job checks
every repair turn the reviewer writes **before** it is taken
(`runstore/review.py` `_candidate`, through `branching.review`'s `turn_check`):

1. `check_turn` runs both checks against the point's parent: `leak_terms`
   (hidden names, h_s) and Jev (the test patch the grade applied -- the grade
   now names it as `test_patch`, a path like `model_patch` -- its FAIL_TO_PASS,
   the verdict as the analyst read it, the task statement, h_s). `leak` is True
   when either flags the turn, None when neither does but one could not run
   (no facts, Jev unreachable), else False.
2. A flagged turn goes back to the reviewer, `resamples` times at most (default
   2), with `TURN_REJECTED`: which branch, and that it states specifics the
   actor could not know -- never the flagged spans, which come from the hidden
   tests the reviewer has not read. A rejection does not count against
   `max_attempts`. Every rejected turn and its check is in the plan
   (`review_attempts[].turn_checks`, `turn_rejections`).
3. Once the resamples are spent the turn is taken anyway and keeps its check;
   so is the last flagged turn when the rewrite after it yields no plan (its
   request fails, or the reviewer's attempts run out) -- `taken_flagged` in the
   plan. A judge that cannot answer (30 s per request, 3 tries) or answers
   in a shape it cannot read leaves the turn unchecked (`leak` None), never
   fails the review; a worker without the key fails it before any model call.
   The candidate carries the check as `leak_judge`, and each repair's
   `metadata.search.leak_judge` lists the check of every turn its chain
   inserted, in round order. The trainer decides: a repair is clean only when
   every entry's `leak` is False.

Without `leak_judge` nothing is checked and the review's and the driver's
output are what they were (no `leak_judge` key anywhere).

## In a Harbor sprout evaluation

`methods.evaluate --method sprout --leak-judge` (assistant-turn guidance; the
key in `$TYPESAFE_API_KEY`, or `--leak-judge-key-env`) does the same in every
review round: each turn the reviewer writes is checked before the round's plan
is taken, a flagged one goes back up to `--leak-judge-resamples` times (default
2), and the checks are in `plan-roundN.json` (`selected_branches[].turn_check`,
`review_attempts[].turn_checks`, `turn_rejections`, `taken_flagged`). The hidden
text is the task's `tests/test.patch` (or, for a task without one, every text
file under `tests/` as added lines) and FAIL_TO_PASS from `tests/config.json`,
with the verifier output of every attempt graded so far -- what the reviewer
read. `examples.qwen.run --method sprout --leak-judge [--leak-judge-resamples N]
[--leak-judge-threshold T]` passes it to every sprout stage and records it in the
manifest (`review.leak_judge`). Both modules live in `methods` (`methods.leak`,
`methods.leak_judge`), so the Harbor path and the Run Store review share them.

## Limits

- Only verbatim (whitespace-normalised) overlap becomes a candidate: a
  paraphrased expectation ("the third item should be dropped") is not seen.
- The model is pinned (`jev-1.13.0`); re-calibrate before moving the threshold
  to a newer version.
