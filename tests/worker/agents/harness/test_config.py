"""The harness options of ``RunSpec.extra["harness"]``."""

from __future__ import annotations

import pytest

from sprout.worker.agents.harness.config import HarnessConfig


def test_options_default_to_evaluation_mode_with_the_papers_tools() -> None:
    config = HarnessConfig.from_extra({})

    assert (config.toolset, config.on_invalid, config.require_submit) == ("default", "feedback", None)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"require_submit": "yes"}, "harness require_submit must be a boolean"),
        ({"toolset": "arctic"}, "harness toolset must be one of default, full"),
        ({"on_invalid": "retry"}, "harness on_invalid must be feedback or terminate"),
        ({"submit": True}, "Unknown harness options: submit"),
    ],
)
def test_a_bad_option_is_refused_with_its_name(options: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HarnessConfig.from_extra({"harness": options})
