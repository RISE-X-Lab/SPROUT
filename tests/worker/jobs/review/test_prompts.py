"""``load_prompts`` reads a prompts file only when every template can render, and names what is wrong."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS_PATH, load_prompts


def edited_copy(path: Path, edit: Callable[[dict[str, Any]], object]) -> Path:
    """The packaged prompts file at ``path``, with ``edit`` applied to its mapping."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    edit(data)
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize(
    "edit, message",
    [
        pytest.param(
            lambda d: d["reviewer"].pop("point"), r"missing prompt reviewer\.point$", id="missing-key"
        ),
        pytest.param(
            lambda d: d["analyst"].update(feedback="Rejected: {reason}"),
            r"prompt analyst\.feedback has the wrong placeholders: "
            r"missing \['error'\], unexpected \['reason'\]",
            id="wrong-placeholders",
        ),
        pytest.param(
            lambda d: d["reviewer"].update(point=d["reviewer"]["point"].replace("{reports}", "(reports)")),
            r"prompt reviewer\.point has the wrong placeholders: missing \['reports'\], unexpected \[\]",
            id="a-dropped-placeholder",
        ),
        pytest.param(
            lambda d: d["analyst"].update(feedback="Rejected: {error} {unbalanced"),
            r"prompt analyst\.feedback is not a valid str\.format template: "
            r"expected '\}' before end of string",
            id="unclosed-brace",
        ),
        pytest.param(
            lambda d: d["reviewer"].update(
                point=d["reviewer"]["point"].replace('{{"synthesis"', '{"synthesis"')
            ),
            r"prompt reviewer\.point is not a valid str\.format template: Single '\}' encountered",
            id="single-literal-brace",
        ),
        pytest.param(
            lambda d: d["reviewer"].update(user_hints=d["reviewer"]["user_hint"]),
            r"unknown prompt keys \['reviewer\.user_hints'\]",
            id="misspelled-key",
        ),
        pytest.param(
            lambda d: d["shepherd"].update(system=["a", "list"]),
            r"prompt shepherd\.system must be a non-empty string, not list",
            id="not-text",
        ),
        pytest.param(
            lambda d: d["transport"].update(json_directive="  "),
            r"prompt transport\.json_directive must be a non-empty string, not str",
            id="blank-text",
        ),
    ],
)
def test_a_prompts_file_that_cannot_render_is_refused_with_the_file_and_the_key_named(
    tmp_path, edit, message
):
    path = edited_copy(tmp_path / "prompts.yaml", edit)

    with pytest.raises(ValueError, match=message) as refused:
        load_prompts(path)

    assert str(refused.value).startswith(f"{path}: ")


@pytest.mark.parametrize(
    "text, message",
    [
        ("analyst: [unclosed\n", "not valid YAML"),
        ("just text\n", "the prompts file must be a mapping, not str"),
    ],
)
def test_a_prompts_file_that_is_not_a_yaml_mapping_is_refused(tmp_path, text, message):
    path = tmp_path / "prompts.yaml"
    path.write_text(text)

    with pytest.raises(ValueError, match=message):
        load_prompts(path)


def test_a_missing_prompts_file_raises_instead_of_falling_back_to_the_packaged_one(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "absent.yaml")


def test_a_verbatim_prompt_keeps_its_single_braces(tmp_path):
    """``shepherd.system`` is sent as it is, never through ``str.format``."""
    path = edited_copy(
        tmp_path / "prompts.yaml", lambda d: d["shepherd"].update(system='Answer {"step": 1}.')
    )

    prompts = load_prompts(path)

    assert prompts.shepherd_system == 'Answer {"step": 1}.'


def test_the_prompts_record_their_path_and_the_sha256_of_the_file_bytes(tmp_path):
    """A run records the digest, and ``sha256sum`` of the file must give the same value."""
    path = edited_copy(tmp_path / "prompts.yaml", lambda d: None)

    prompts = load_prompts(path)

    assert (prompts.path, prompts.sha256) == (path, hashlib.sha256(path.read_bytes()).hexdigest())
