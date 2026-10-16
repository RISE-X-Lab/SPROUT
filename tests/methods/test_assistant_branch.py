"""Assistant-turn and point-only branches: the reviewer's turn reaches the actor exactly,
and the prompts it is authored under come from one validated YAML file."""

import json
from types import SimpleNamespace

import pytest
import yaml

from sprout.harness_agents.mini_swe.history import available_branch_points
from tests.harness.test_assistant_turn import assistant_turn
from tests.harness.test_mini_swe import pytestmark as mini_only
from tests.runstore.test_mini_native import parent_run
from sprout.methods import branching
from sprout.methods.branching import reviewer_context
from sprout.methods.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH, load_prompts

HEADING = "## Probe first, then fix, then verify"
FENCE_RULE = "Return exactly one fenced"


def render():
    return DEFAULT_PROMPTS.reviewer_assistant_turn.format(problem="task", reports="[]", count_rule="At most 1.")


def probe_first_section(template=DEFAULT_PROMPTS.reviewer_assistant_turn):
    """The probe-first section: from its heading to the fence rule that follows it."""
    start = template.index(HEADING)
    return template[start:template.index(FENCE_RULE, start)]


def write_prompts(path, mutate):
    """The packaged file with ``mutate`` applied to its parsed mapping."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    mutate(data)
    path.write_text(yaml.safe_dump(data))
    return path


def test_assistant_prompt_describes_real_execution_of_the_authored_turn():
    prompt = render()
    assert "ordinary content" in prompt and "record the REAL observations" in prompt


def test_the_assistant_prompt_asks_for_probe_first_turns_and_renders_cleanly():
    assert HEADING in DEFAULT_PROMPTS.reviewer_assistant_turn
    rendered = render()                       # no KeyError: {problem} {reports} {count_rule} and nothing else
    assert "\n\n\n" not in rendered
    assert rendered.index("supported there.") < rendered.index(HEADING) < rendered.index(FENCE_RULE)
    assert "supported there.\n\n" + HEADING in rendered, "one blank line before the section"
    section = probe_first_section()
    assert section.endswith("committing is allowed.\n\n"), "one blank line after it"
    lines = section.rstrip("\n").split("\n")
    assert len(lines) <= 22 and max(map(len, lines)) <= 80
    for word in ("reviewer", "checkpoint", "restore", "replay", "branch", "grader", "hidden"):
        assert word not in section.lower(), word
    assert "{" not in section and "}" not in section
    for template in (DEFAULT_PROMPTS.reviewer_point, DEFAULT_PROMPTS.reviewer_user_hint):
        assert HEADING not in template


def test_an_invalid_prompts_file_is_refused_naming_the_key(tmp_path):
    def drop(data):
        del data["reviewer"]["point"]
    with pytest.raises(ValueError, match=r"missing prompt reviewer\.point"):
        load_prompts(write_prompts(tmp_path / "missing.yaml", drop))

    def wrong(data):
        data["analyst"]["feedback"] = "Your previous reply was rejected: {reason}"
    with pytest.raises(ValueError, match=r"analyst\.feedback has the wrong placeholders: "
                                         r"missing \['error'\], unexpected \['reason'\]"):
        load_prompts(write_prompts(tmp_path / "wrong.yaml", wrong))

    def dropped_field(data):
        data["reviewer"]["user_hint"] = data["reviewer"]["user_hint"].replace("{reports}", "(reports)")
    with pytest.raises(ValueError, match=r"reviewer\.user_hint .* missing \['reports'\], unexpected \[\]"):
        load_prompts(write_prompts(tmp_path / "dropped.yaml", dropped_field))

    # A template that does not parse is refused by key too, not with Formatter's bare
    # message: a literal brace left single is the likeliest slip in this file.
    def unbalanced(data):
        data["analyst"]["feedback"] += " {unbalanced"
    with pytest.raises(ValueError, match=r"analyst\.feedback is not a valid str\.format template: "
                                         r"expected '\}' before end of string") as refused:
        load_prompts(write_prompts(tmp_path / "unbalanced.yaml", unbalanced))
    assert str(refused.value).startswith(str(tmp_path / "unbalanced.yaml") + ": ")

    def single_brace(data):
        data["reviewer"]["point"] = data["reviewer"]["point"].replace('{{"synthesis"', '{"synthesis"')
    with pytest.raises(ValueError, match=r"reviewer\.point is not a valid str\.format template: Single '\}'"):
        load_prompts(write_prompts(tmp_path / "single.yaml", single_brace))

    def typo(data):
        data["reviewer"]["user_hints"] = data["reviewer"]["user_hint"]
    with pytest.raises(ValueError, match=r"unknown prompt keys \['reviewer\.user_hints'\]"):
        load_prompts(write_prompts(tmp_path / "typo.yaml", typo))

    def not_text(data):
        data["shepherd"]["system"] = ["a", "list"]
    with pytest.raises(ValueError, match=r"shepherd\.system must be a non-empty string, not list"):
        load_prompts(write_prompts(tmp_path / "list.yaml", not_text))

    (tmp_path / "broken.yaml").write_text("analyst: [unclosed\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_prompts(tmp_path / "broken.yaml")
    (tmp_path / "scalar.yaml").write_text("just text\n")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_prompts(tmp_path / "scalar.yaml")
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "absent.yaml")


@mini_only
def test_review_context_and_selection_validate_the_exact_prefix(tmp_path, monkeypatch):
    _, parent, native_points = parent_run(tmp_path, monkeypatch)
    points = available_branch_points(parent.journal_path)
    context = reviewer_context(parent.journal_path, points)
    assert set(context["prefix_message_counts"]) == {"2", "3", "4"}
    early_count = context["prefix_message_counts"]["2"]
    early = context["messages"][:early_count]
    assert early[-1]["role"] == "tool" and "echo late" not in json.dumps(early)
    turn = assistant_turn()
    plan = {"branches": [{"base": "parent", "branch_step": 2, "assistant_turn": turn}]}
    kwargs = dict(limit=1, round_no=1, attempts={"parent": SimpleNamespace(outcome=parent)},
                  checkpoints={"parent": points}, guidance_mode="assistant-turn")
    choices = branching.prepare_branches(plan, **kwargs)
    assert choices[0].assistant_turn == turn and choices[0].cut == native_points[0].native["cut"]
    # A reviewer-authored turn never ends the episode: that decision is the actor's.
    submit = assistant_turn()
    submit["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"command": "git add -A && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})
    submit_plan = {"branches": [{"base": "parent", "branch_step": 2, "assistant_turn": submit}]}
    with pytest.raises(ValueError, match=r"^branch 1 \(parent@2\): a reviewer-authored turn never submits; "
                                         r"command 1 contains COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT$"):
        branching.prepare_branches(submit_plan, **kwargs)
    plan["branches"][0]["hint"] = "must not leak"
    with pytest.raises(ValueError, match="must not include a user hint"):
        branching.prepare_branches(plan, **kwargs)
    del plan["branches"][0]["hint"]
    turn["tool_calls"][0]["id"] = next(m["tool_calls"][0]["id"] for m in early if m.get("tool_calls"))
    with pytest.raises(ValueError, match="unique across history"):
        branching.prepare_branches(plan, **kwargs)
    point = {"base": "parent", "branch_step": 2, "why": "Only a restart point."}
    kwargs["guidance_mode"] = "none"
    selected = branching.prepare_branches({"branches": [point]}, **kwargs)[0]
    assert selected.hint == "" and selected.assistant_turn is None
    assert selected.cut == native_points[0].native["cut"]
    for forbidden in ["hint", "assistant_turn", "command", "content", "reasoning"]:
        with pytest.raises(ValueError, match="only point-selection fields"):
            branching.prepare_branches({"branches": [{**point, forbidden: ""}]}, **kwargs)
    # The hint mode carries the reviewer's hint and resolves the same cut.
    kwargs["guidance_mode"] = "user-hint"
    hinted = branching.prepare_branches(
        {"branches": [{"base": "parent", "branch_step": 2, "hint": "inspect", "why": "test"}]}, **kwargs)[0]
    assert hinted.hint == "inspect" and hinted.cut == native_points[0].native["cut"]


def test_nonmini_parent_is_rejected_even_with_mini_actor(tmp_path):
    journal = tmp_path / "claude.jsonl"
    journal.write_text(json.dumps({"type": "run.started", "slot": "claude-code"}) + "\n")
    with pytest.raises(ValueError, match="only mini-swe-agent"):
        branching.require_mini_parent(journal)
