"""``TemplateBuilder``: the microVM template of an image, reused when it exists and built when not.

The builder talks to a scripted AgentENV API on a local port (``agentenv_stub``). The names below
are the template names of ``IMAGE`` for the runtime binary ``runtime v1``. ``test_template_recipe.py``
pins how a name derives from its inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sprout.worker.sandbox.templates as templates
from sprout.worker.sandbox.template_recipe import TemplateError
from sprout.worker.sandbox.templates import TemplateBuilder
from tests.worker.sandbox.agentenv import Request, scripted_build

IMAGE = "swebench/sweb.eval.x86_64.django__django-11848:latest"
#: The name of the template of ``IMAGE`` on port 3000, without a shape.
NAME = "sprout-swebench-09174ad6d0cef19bbb31a625"
#: ``chmod`` of the staged binaries: an upload does not carry the mode bits.
CHMOD = "chmod +x /usr/local/bin/sprout-runtime; chmod +x /usr/local/bin/rg 2>/dev/null || true"


@pytest.fixture
def runtime_bin(tmp_path: Path) -> Path:
    path = tmp_path / "sprout-runtime"
    path.write_bytes(b"runtime v1")
    return path


@pytest.fixture(autouse=True)
def no_poll_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll a build's status without the 3 s pause between two polls."""
    monkeypatch.setattr(templates, "BUILD_POLL_SECONDS", 0.0)


def builder(stub, runtime_bin: Path, **options) -> TemplateBuilder:
    return TemplateBuilder(server_url=stub.url, api_key="key", runtime_bin=runtime_bin, **options)


def test_a_missing_runtime_binary_is_refused_when_the_builder_is_made(tmp_path):
    with pytest.raises(TemplateError, match="^runtime binary not found at .*/no-such-file; build it "):
        TemplateBuilder(server_url="http://127.0.0.1:1", api_key="key", runtime_bin=tmp_path / "no-such-file")


def test_a_build_stages_the_runtime_in_a_cold_started_sandbox_and_builds_the_template_from_its_snapshot(
    agentenv_stub, runtime_bin
):
    """The staged snapshot has no name, so a retry after a failed build cannot collide with it."""
    scripted_build(agentenv_stub)

    name = builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    upload = {"name": "file", "filename": "sprout-runtime", "content": b"runtime v1"}
    build = {
        "fromTemplate": "staged-snap",
        "steps": [{"type": "RUN", "args": [CHMOD]}],
        "startCmd": "/usr/local/bin/sprout-runtime --port 3000",
        "readyCmd": "timeout 1 bash -c '</dev/tcp/127.0.0.1/3000'",
    }
    assert name == NAME
    assert agentenv_stub.requests == [
        Request("GET", "/templates/aliases/" + NAME),
        Request("GET", "/templates/aliases/" + NAME),
        Request("POST", "/sandboxes-cold", body={"image": IMAGE, "timeout": 3600, "autoPause": False}),
        Request("POST", "/files", query={"path": "/usr/local/bin/sprout-runtime"}, body=upload),
        Request("POST", "/sandboxes/staging-sb/snapshots", body={"diskOnly": True}),
        Request("DELETE", "/sandboxes/staging-sb"),
        Request("POST", "/v3/templates", body={"name": NAME}),
        Request("POST", "/v2/templates/tid/builds/bid", body=build),
        Request("GET", "/templates/tid/builds/bid/status"),
    ]


def test_each_request_carries_the_api_key_and_an_upload_goes_to_the_file_service(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)

    builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    [upload] = agentenv_stub.requests_to("/files")
    assert agentenv_stub.header_values("x-api-key") == {"key"}
    assert (upload.headers["x-agentenv-sandbox-id"], upload.headers["x-agentenv-target-port"]) == (
        "staging-sb",
        "49983",
    )


@pytest.mark.parametrize(
    "resources, shape, name",
    [
        (
            {"cpu": 2, "memory_mb": 2048},
            {"cpuCount": 2, "memoryMB": 2048},
            "sprout-swebench-b41f2d4701a6991b0869b06d",
        ),
        (
            {"cpu": 2, "memory_mb": 2048, "disk_size_mb": 65536},
            {"cpuCount": 2, "memoryMB": 2048, "diskSizeMB": 65536},
            "sprout-swebench-d12d50be1561fe66fd8586d8",
        ),
    ],
    ids=["cpu-and-memory", "with-disk"],
)
def test_the_task_shape_reaches_the_cold_start_and_the_template(
    agentenv_stub, runtime_bin, resources, shape, name
):
    """A snapshot-based build cannot change the shape, so the template declares the staged one."""
    scripted_build(agentenv_stub)

    built = builder(agentenv_stub, runtime_bin).template_for(IMAGE, resources)

    assert built == name
    assert agentenv_stub.bodies("/sandboxes-cold", "/v3/templates") == [
        {"image": IMAGE, "timeout": 3600, "autoPause": False, **shape},
        {"name": name, **shape},
    ]


def test_a_template_that_exists_is_used_without_a_build_or_a_snapshot_lookup(agentenv_stub, runtime_bin):
    """An image reference cannot be a snapshot id, and the catalog answers such a name with an error."""
    agentenv_stub.answer("GET", "/templates/aliases/" + NAME, (200, {"templateID": "tid"}))
    agentenv_stub.answer("GET", "/templates/tid/builds/tid/status", (200, {"status": "ready"}))

    name = builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    assert (name, agentenv_stub.paths()) == (
        NAME,
        ["GET /templates/aliases/" + NAME, "GET /templates/tid/builds/tid/status"],
    )


def test_a_resolved_name_is_reused_without_asking_the_server_again(agentenv_stub, runtime_bin):
    """A batch builds each distinct image once."""
    scripted_build(agentenv_stub)
    resolver = builder(agentenv_stub, runtime_bin)
    first = resolver.template_for(IMAGE)
    asked = len(agentenv_stub.requests)

    second = resolver.template_for(IMAGE)

    assert (second, len(agentenv_stub.requests)) == (first, asked)


def test_a_name_that_the_catalog_knows_as_a_snapshot_is_used_as_it_is(agentenv_stub, runtime_bin):
    """A replay starts from a checkpoint's snapshot id. A build would cold-start it as an image."""
    agentenv_stub.answer("GET", "/snapshots/01a0-checkpoint", (200, {"snapshotID": "01a0-checkpoint"}))

    name = builder(agentenv_stub, runtime_bin).template_for("01a0-checkpoint")

    assert (name, agentenv_stub.paths()) == ("01a0-checkpoint", ["GET /snapshots/01a0-checkpoint"])


def test_a_name_that_the_backend_knows_as_a_template_is_used_as_it_is(agentenv_stub, runtime_bin):
    """``/snapshots/{name}`` answers only for snapshots of sandboxes, and 404 for a built template."""
    agentenv_stub.answer("GET", "/templates/aliases/" + NAME, (200, {"templateID": "tid"}))
    agentenv_stub.answer("GET", "/templates/tid/builds/tid/status", (200, {"status": "ready"}))

    name = builder(agentenv_stub, runtime_bin).template_for(NAME)

    assert (name, agentenv_stub.bodies("/v3/templates")) == (NAME, [])


def test_a_bare_name_that_the_backend_does_not_know_gets_a_template_built(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)

    name = builder(agentenv_stub, runtime_bin).template_for("ubuntu")

    assert name == "sprout-swebench-cd8c10016b6a1f83f09a5862"
    assert agentenv_stub.paths()[:2] == ["GET /snapshots/ubuntu", "GET /templates/aliases/ubuntu"]
    assert agentenv_stub.bodies("/sandboxes-cold") == [
        {"image": "ubuntu", "timeout": 3600, "autoPause": False}
    ]


def test_a_lookup_that_the_server_cannot_answer_raises_instead_of_reading_as_not_found(
    agentenv_stub, runtime_bin
):
    """A name that reads as not found gets a second build."""
    agentenv_stub.answer("GET", "/templates/aliases/" + NAME, (500, "database is locked"))

    with pytest.raises(TemplateError, match="^could not check %s: HTTP 500 database is locked$" % NAME):
        builder(agentenv_stub, runtime_bin).template_for(IMAGE)


def test_the_staging_sandbox_is_deleted_when_its_snapshot_fails(agentenv_stub, runtime_bin):
    """Left running, it would hold a VM for the rest of the run."""
    scripted_build(agentenv_stub)
    agentenv_stub.answer("POST", "/sandboxes/staging-sb/snapshots", (500, "disk full"))

    with pytest.raises(
        TemplateError, match="^could not snapshot the staged runtime for %s: HTTP 500 disk full$" % IMAGE
    ):
        builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    assert agentenv_stub.paths()[-1] == "DELETE /sandboxes/staging-sb"


def test_a_build_that_is_still_running_is_waited_for(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)
    agentenv_stub.answer(
        "GET",
        "/templates/tid/builds/bid/status",
        (200, {"status": "building"}),
        (200, {"status": "building"}),
        (200, {"status": "ready"}),
    )

    name = builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    assert (name, agentenv_stub.paths()[-3:]) == (NAME, ["GET /templates/tid/builds/bid/status"] * 3)


def test_a_build_that_fails_raises(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)
    agentenv_stub.answer("GET", "/templates/tid/builds/bid/status", (200, {"status": "error"}))

    with pytest.raises(TemplateError, match=r"^template build failed for %s \(%s\): " % (NAME, IMAGE)):
        builder(agentenv_stub, runtime_bin).template_for(IMAGE)


def test_a_build_that_does_not_finish_in_its_time_raises_instead_of_hanging_the_run(
    agentenv_stub, runtime_bin
):
    scripted_build(agentenv_stub)
    agentenv_stub.answer("GET", "/templates/tid/builds/bid/status", (200, {"status": "building"}))

    with pytest.raises(TemplateError, match=r"did not finish in 0s \(last status: building\)$"):
        builder(agentenv_stub, runtime_bin, build_timeout=0.2).template_for(IMAGE)


@pytest.mark.parametrize(
    "reply",
    [(409, {"message": "conflict"}), (400, "alias 'x' already points to 'y', cannot rebind")],
    ids=["conflict", "alias-already-points"],
)
def test_a_template_that_another_builder_created_first_is_used(agentenv_stub, runtime_bin, reply):
    """Two workers can build the same image at once. The second one uses the first one's build."""
    scripted_build(agentenv_stub)
    agentenv_stub.answer("POST", "/v3/templates", reply)

    name = builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    assert (name, agentenv_stub.paths()[-1]) == (NAME, "POST /v3/templates")


def test_a_template_that_cannot_be_created_raises(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)
    agentenv_stub.answer("POST", "/v3/templates", (400, "invalid name"))

    with pytest.raises(
        TemplateError, match="^could not create template %s for %s: HTTP 400 invalid name$" % (NAME, IMAGE)
    ):
        builder(agentenv_stub, runtime_bin).template_for(IMAGE)


@pytest.mark.parametrize(
    "alias, status, name",
    [
        ((200, {"templateID": "t0"}), (200, {"status": "error"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "failed"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "failure"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "cancelled"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "canceled"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "Error"}), NAME + "-r1"),
        ((200, {"templateID": "t0"}), (200, {"status": "ready"}), NAME),
        ((200, {"templateID": "t0"}), (200, {"status": "a-state-added-later"}), NAME),
        ((200, {"templateID": "t0"}), (200, {}), NAME),
        ((200, {"templateID": "t0"}), (404, {}), NAME),
        ((200, {}), (200, {"status": "error"}), NAME),
    ],
    ids=[
        "error",
        "failed",
        "failure",
        "cancelled",
        "canceled",
        "upper-case",
        "ready",
        "unknown-state",
        "no-status",
        "no-status-endpoint",
        "alias-without-template-id",
    ],
)
def test_only_a_build_that_is_known_to_have_failed_makes_its_name_unusable(
    agentenv_stub, runtime_bin, alias, status, name
):
    """An alias cannot be rebound, so a failed build owns its name for good, and the next name gets a
    build. Any other answer leaves the template usable. A bad one fails at spawn, and names do not
    multiply."""
    scripted_build(agentenv_stub)
    agentenv_stub.answer("GET", "/templates/aliases/" + NAME, alias)
    agentenv_stub.answer("GET", "/templates/t0/builds/t0/status", status)

    built = builder(agentenv_stub, runtime_bin).template_for(IMAGE)

    assert built == name


def failed_builds(stub, *names: str) -> None:
    """Script each of ``names`` as the alias of a template whose build failed."""
    stub.answer("GET", "/templates/t0/builds/t0/status", (200, {"status": "error"}))
    for name in names:
        stub.answer("GET", "/templates/aliases/" + name, (200, {"templateID": "t0"}))


def test_the_search_for_a_usable_name_gives_up_after_six_names(agentenv_stub, runtime_bin):
    """A need for more than a couple means that a build fails repeatably."""
    failed_builds(agentenv_stub, NAME, *("%s-r%d" % (NAME, n) for n in range(1, 6)))

    with pytest.raises(
        TemplateError,
        match="^no usable template name for %s: 6 variants of %s are taken by failed builds$" % (IMAGE, NAME),
    ):
        builder(agentenv_stub, runtime_bin).template_for(IMAGE)


def test_another_runtime_port_changes_the_name_and_the_start_and_ready_commands(agentenv_stub, runtime_bin):
    scripted_build(agentenv_stub)

    name = builder(agentenv_stub, runtime_bin, runtime_port=34122).template_for(IMAGE)

    [build] = agentenv_stub.bodies("/v2/templates/tid/builds/bid")
    assert (name, build["startCmd"], build["readyCmd"]) == (
        "sprout-swebench-f49d1b1dbac4ed95b25b6fd9",
        "/usr/local/bin/sprout-runtime --port 34122",
        "timeout 1 bash -c '</dev/tcp/127.0.0.1/34122'",
    )


def test_ripgrep_is_staged_beside_the_runtime_and_changes_the_name(agentenv_stub, runtime_bin, tmp_path):
    """A template without ``rg`` must not be reused once ``rg`` is available."""
    scripted_build(agentenv_stub)
    ripgrep = tmp_path / "rg"
    ripgrep.write_bytes(b"rg v14")

    name = builder(agentenv_stub, runtime_bin, ripgrep_bin=ripgrep).template_for(IMAGE)

    assert name == "sprout-swebench-556def47ccdd2a3801376107"
    assert agentenv_stub.requests_to("/files") == [
        Request(
            "POST",
            "/files",
            query={"path": "/usr/local/bin/sprout-runtime"},
            body={"name": "file", "filename": "sprout-runtime", "content": b"runtime v1"},
        ),
        Request(
            "POST",
            "/files",
            query={"path": "/usr/local/bin/rg"},
            body={"name": "file", "filename": "rg", "content": b"rg v14"},
        ),
    ]


def test_a_ripgrep_binary_that_is_gone_is_left_out_instead_of_failing_the_build(
    agentenv_stub, runtime_bin, tmp_path
):
    """The cache entry of ``ensure_ripgrep`` can be cleaned away. A template without ``rg`` still works."""
    scripted_build(agentenv_stub)

    name = builder(agentenv_stub, runtime_bin, ripgrep_bin=tmp_path / "absent-rg").template_for(IMAGE)

    assert (name, len(agentenv_stub.bodies("/files"))) == (NAME, 1)


#: A regctl that logs its arguments and prints an image config whose ENV has a value with a space.
STUB_REGCTL = """#!/bin/sh
echo "$@" >> "$REGCTL_LOG"
echo '{"Env":["PATH=/opt/venv/bin:/usr/bin","LANG=C.UTF-8","MSG=has space"]}'
"""


def test_with_image_env_the_runtime_starts_under_the_images_env_in_a_template_of_its_own(
    agentenv_stub, runtime_bin, tmp_path, monkeypatch
):
    """The guest agent rebuilds ``PATH``. A template built without ``image_env`` keeps its name."""
    scripted_build(agentenv_stub)
    regctl = tmp_path / "regctl"
    regctl.write_text(STUB_REGCTL)
    regctl.chmod(0o755)
    monkeypatch.setenv("REGCTL_LOG", str(tmp_path / "regctl.log"))

    name = builder(agentenv_stub, runtime_bin, image_env=True, regctl_bin=regctl).template_for(IMAGE)

    [build] = agentenv_stub.bodies("/v2/templates/tid/builds/bid")
    assert (name, build["startCmd"]) == (
        "sprout-swebench-616723e5c9dfd16d728382c4",
        "env PATH=/opt/venv/bin:/usr/bin LANG=C.UTF-8 'MSG=has space' "
        "/usr/local/bin/sprout-runtime --port 3000",
    )
    assert (tmp_path / "regctl.log").read_text() == "image config %s --format {{json .Config}}\n" % IMAGE
