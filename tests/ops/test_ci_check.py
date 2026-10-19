"""``ops/contracts/ci_check.py``: a pinned upstream that lost a name the adapters use fails the build."""

from __future__ import annotations

from importlib.metadata import version

import pydantic
import pytest

from ops.contracts import ci_check
from ops.contracts.ci_check import Report, check_classes, check_normalizer_alignment, check_python_api


class Settings(pydantic.BaseModel):
    """A model whose fields a contract can list."""

    timeout_s: int


def checked(check, contract):
    """The report of one check of ``contract``: (checks, failures, skips)."""
    report = Report()
    check(contract, report)
    return report.checks, report.failures, report.skips


def test_a_method_that_the_class_lacks_fails_its_contract():
    contract = {
        "name": "x",
        "classes": [{"module": "json", "cls": "JSONDecoder", "methods": ["decode", "decode_all"]}],
    }

    assert checked(check_classes, contract) == (3, [("json.JSONDecoder.decode_all", "")], [])


def test_a_property_counts_as_a_method_of_the_class():
    contract = {
        "name": "x",
        "classes": [{"module": "pathlib", "cls": "PurePath", "methods": ["name", "with_name"]}],
    }

    assert checked(check_classes, contract) == (3, [], [])


@pytest.mark.parametrize(
    "module,cls,fields,missing",
    [
        (__name__, "Settings", ["timeout_s", "retries"], "tests.ops.test_ci_check.Settings field retries"),
        (
            "sprout.worker.checkpoint.rollback",
            "Checkpoint",
            ["step", "label"],
            "sprout.worker.checkpoint.rollback.Checkpoint field label",
        ),
    ],
    ids=["pydantic-model", "dataclass"],
)
def test_a_field_that_the_model_lacks_fails_its_contract(module, cls, fields, missing):
    contract = {"name": "x", "classes": [{"module": module, "cls": cls, "fields": fields}]}

    assert checked(check_classes, contract) == (3, [(missing, "")], [])


def test_a_class_that_the_module_lacks_fails_with_the_import_error():
    contract = {"name": "x", "classes": [{"module": "json", "cls": "JSONParser"}]}

    assert checked(check_classes, contract) == (
        1,
        [("json.JSONParser", "module 'json' has no attribute 'JSONParser'")],
        [],
    )


def test_an_event_type_that_the_normalizer_does_not_name_fails():
    contract = {
        "slot": "s",
        "normalizer_module": "json.decoder",
        "event_types": ["JSONDecodeError", "mini.turn"],
    }

    assert checked(check_normalizer_alignment, contract) == (2, [("s handles event mini.turn", "")], [])


def test_an_installed_version_other_than_the_pinned_one_fails():
    contract = {
        "name": "x",
        "upstream": {"package": "pytest", "exact_version": True, "verified_versions": {"sdk": "0.0.1"}},
        "python_api": {"module": "json"},
    }

    assert checked(check_python_api, contract) == (
        1,
        [("pytest == 0.0.1", "installed: %s" % version("pytest"))],
        [],
    )


def test_a_python_api_that_is_not_installed_is_skipped_not_failed():
    contract = {"name": "x", "python_api": {"module": "no_such_sdk", "exports": ["Agent"]}}

    assert checked(check_python_api, contract) == (
        0,
        [],
        [("x python api", "no_such_sdk not installed (No module named 'no_such_sdk')")],
    )


def test_a_client_init_keyword_or_method_that_the_sdk_lacks_fails():
    client = {
        "module": "json",
        "cls": "JSONDecoder",
        "init_kwargs": ["object_hook", "hook"],
        "methods": ["decode", "parse"],
    }
    contract = {"name": "x", "python_api": {"module": "json", "exports": ["loads"], "client": client}}

    assert checked(check_python_api, contract) == (
        5,
        [("JSONDecoder.__init__ accepts hook", ""), ("JSONDecoder.parse", "")],
        [],
    )


@pytest.mark.parametrize(
    "methods,code,last",
    [
        (["decode"], 0, "2 checks, 0 failures, 0 skipped"),
        (["parse"], 1, "Adapter drift detected. Update the adapter or the contract."),
    ],
    ids=["holds", "violated"],
)
def test_main_exits_1_on_a_violated_contract_and_0_otherwise(
    tmp_path, capsys, monkeypatch, methods, code, last
):
    (tmp_path / "decoder.yaml").write_text(
        "name: decoder\nclasses:\n  - module: json\n    cls: JSONDecoder\n    methods: %s\n" % methods
    )
    monkeypatch.setattr(ci_check, "CONTRACTS_DIR", tmp_path)

    result = ci_check.main([])

    assert (result, capsys.readouterr().out.splitlines()[-1]) == (code, last)


def test_main_exits_2_when_no_contract_has_the_requested_slot(capsys):
    result = ci_check.main(["--slot", "no-such-slot"])

    assert (result, capsys.readouterr().err) == (2, "no contracts matched\n")
