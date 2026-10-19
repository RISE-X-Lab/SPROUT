"""``ops/gc_layers.py``: it deletes a layer of an AgentENV store only if no record names it and it is old."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from ops import gc_layers

REFERENCED = "a" * 64
ORPHAN = "b" * 64
FRESH_ORPHAN = "c" * 64
#: Three hours ago: older than the default --min-age-hours of 2.
OLD = time.time() - 3 * 3600


def posixfs_store(tmp_path):
    """A store whose one record names the REFERENCED layer, and whose layers are four files."""
    repository = tmp_path / "store" / "repository"
    records = repository / "catalog" / "records"
    records.mkdir(parents=True)
    record = {
        "id": "snap-1",
        "state": "deleted",
        "manifest": {"layers": [{"digest": "sha256:" + REFERENCED}]},
    }
    (records / "snap-1.json").write_text(json.dumps(record))
    layers = repository / "managed-layers"
    layers.mkdir()
    for name, mtime in (
        ("sha256_" + REFERENCED + ".tar", OLD),
        ("sha256_" + ORPHAN, OLD),
        ("sha256_" + FRESH_ORPHAN, None),
        ("blob-1", OLD),
    ):
        (layers / name).write_bytes(b"x" * 1024)
        if mtime is not None:
            os.utime(layers / name, (mtime, mtime))
    return tmp_path / "store"


def gc(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["gc_layers.py", *map(str, args)])
    return gc_layers.main()


def layer_names(store):
    return sorted(path.name for path in (store / "repository" / "managed-layers").iterdir())


def test_a_dry_run_reports_the_orphaned_layers_and_deletes_nothing(tmp_path, monkeypatch, capsys):
    store = posixfs_store(tmp_path)

    code = gc(monkeypatch, "--store", store)

    assert (code, capsys.readouterr().out.splitlines(), layer_names(store)) == (
        0,
        [
            "records scanned: 1, distinct digests referenced: 1",
            "layers: 2 referenced/kept, 1 too young to judge, 1 orphaned",
            "reclaimable: 0.0 GiB",
            "",
            "DRY RUN -- nothing deleted. Pass --yes to reclaim.",
        ],
        ["blob-1", "sha256_" + REFERENCED + ".tar", "sha256_" + ORPHAN, "sha256_" + FRESH_ORPHAN],
    )


def test_yes_deletes_only_an_unreferenced_layer_older_than_the_minimum_age(tmp_path, monkeypatch, capsys):
    """A layer of an unknown name, a referenced one and a young one, which may await its record, stay."""
    store = posixfs_store(tmp_path)

    code = gc(monkeypatch, "--store", store, "--yes")

    assert (code, capsys.readouterr().out.splitlines()[-1], layer_names(store)) == (
        0,
        "deleted 1 layers, freed 0.0 GiB",
        ["blob-1", "sha256_" + REFERENCED + ".tar", "sha256_" + FRESH_ORPHAN],
    )


@pytest.mark.parametrize(
    "present,missing",
    [("managed-layers", "catalog/records"), ("catalog/records", "managed-layers")],
    ids=["no-records", "no-layers"],
)
def test_a_directory_that_is_not_a_posixfs_store_exits_2(tmp_path, monkeypatch, capsys, present, missing):
    repository = tmp_path / "store" / "repository"
    (repository / present).mkdir(parents=True)

    code = gc(monkeypatch, "--store", tmp_path / "store", "--yes")

    assert (code, capsys.readouterr().err) == (
        2,
        "not a posixfs snapshot store: missing %s\n" % (repository / missing),
    )
