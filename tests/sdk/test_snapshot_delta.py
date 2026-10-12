import pytest

from sprout_sandbox.pool import Snapshot


@pytest.mark.parametrize("value,expected", [(True, True), (False, False), (None, None),
                                           ("false", None), (0, None)])
def test_snapshot_delta_empty_preserves_boolean_and_unknown(value, expected):
    snapshot = Snapshot.from_api({"snapshotID": "s", "deltaEmpty": value})
    assert snapshot.delta_empty is expected


def test_older_snapshot_without_delta_tag_remains_unknown():
    assert Snapshot.from_api({"snapshotID": "s"}).delta_empty is None
