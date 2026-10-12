"""The attempt child inherits the podman and store the worker resolved."""

from sprout.runstore.worker import CHILD_ENV_PASSTHROUGH


def test_the_attempt_child_inherits_the_podman_and_store_variables():
    """The podman and store checks export SPROUT_PODMAN_BIN / SPROUT_PODMAN_STORE into the
    worker's environment; the child that builds the sandboxes must see them too, or it would
    run them on `podman` from PATH while the worker asks another podman whether they are gone."""
    assert {"SPROUT_PODMAN_BIN", "SPROUT_PODMAN_STORE", "PATH", "HOME"} <= set(CHILD_ENV_PASSTHROUGH)
