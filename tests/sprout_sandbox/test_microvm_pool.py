"""``MicroVMPool``, the client of an AgentENV server, and the ``Snapshot`` that every pool returns.

Firecracker needs ``/dev/kvm`` and an AgentENV server, so the medium tests run the pool against the
fake server of ``fake_agentenv.py``. They check the client's side of the contract: the routes, the
field names, the routing headers, and what the pool holds after each call.
"""

from __future__ import annotations

from collections.abc import Coroutine, Iterator
from typing import Any

import httpx
import pytest

from sprout_sandbox import MicroVMPool, Sandbox, Snapshot, ToolResult
from tests.sprout_sandbox.fake_agentenv import FakeAgentEnv, serving_agentenv

pytestmark = pytest.mark.anyio


def present(mapping: dict, *keys: str) -> dict:
    """The entries of ``mapping`` for those of ``keys`` that it has."""
    return {key: mapping[key] for key in keys if key in mapping}


def create_through(pool: MicroVMPool, route: str) -> Coroutine[Any, Any, Sandbox]:
    """A create of a sandbox through ``route``: from a template, or a cold start from an image."""
    if route == "/sandboxes-cold":
        return pool.spawn_from_image("registry.example/task:tag")
    return pool.spawn(image="swe-base")


@pytest.fixture
def agentenv() -> Iterator[FakeAgentEnv]:
    with serving_agentenv() as fake:
        yield fake


@pytest.mark.parametrize(
    "delta_empty, expected",
    [
        pytest.param(True, True, id="true"),
        pytest.param(False, False, id="false"),
        pytest.param(None, None, id="null"),
        pytest.param("false", None, id="text"),
        pytest.param(0, None, id="zero"),
    ],
)
def test_a_snapshot_reports_an_empty_delta_only_when_the_server_says_so_with_a_boolean(delta_empty, expected):
    assert Snapshot.from_api({"snapshotID": "snap-7", "deltaEmpty": delta_empty}).delta_empty is expected


def test_a_snapshot_reads_its_id_and_layer_counts_from_the_api():
    body = {"snapshotID": "snap-7", "rootfsLayerCount": 4, "memoryLayerCount": 2}

    snapshot = Snapshot.from_api(body, disk_only=True)

    assert snapshot == Snapshot(
        id="snap-7", rootfs_layers=4, memory_layers=2, disk_only=True, delta_empty=None
    )


def test_a_snapshot_from_a_server_that_reports_no_layers_has_no_layer_counts():
    assert Snapshot.from_api({"snapshotID": "snap-7"}) == Snapshot(id="snap-7")


def test_the_pool_offers_snapshots_cold_starts_uploads_and_downloads():
    pool = MicroVMPool("http://agentenv.invalid")

    capabilities = (
        pool.supports_snapshot(),
        pool.supports_cold_start(),
        pool.supports_upload(),
        pool.supports_download(),
    )

    assert capabilities == (True, True, True, True)


@pytest.mark.parametrize(
    "env, expected_body",
    [
        pytest.param(
            None,
            {"templateID": "swe-base", "timeout": 900, "autoPause": True, "autoResume": {"enabled": True}},
            id="no-env",
        ),
        pytest.param(
            {"A": "1"},
            {
                "templateID": "swe-base",
                "timeout": 900,
                "autoPause": True,
                "autoResume": {"enabled": True},
                "envVars": {"A": "1"},
            },
            id="env",
        ),
    ],
)
async def test_a_spawn_asks_for_the_template_with_the_pools_ttl_and_a_pause_that_resumes(
    agentenv, env, expected_body
):
    """AgentENV pauses a sandbox after 15 s by default. A sandbox kept for a grade must resume on its
    next call."""
    pool = MicroVMPool(agentenv.url, sandbox_ttl=900)

    sandbox = await pool.spawn(image="swe-base", env=env)

    assert agentenv.posted() == [("/sandboxes", expected_body)]
    assert (sandbox.sandbox_id, pool.list()) == ("vm-1", [sandbox])


@pytest.mark.parametrize(
    "api_key, expected",
    [
        pytest.param("secret-key", {"x-api-key": "secret-key"}, id="key"),
        pytest.param("", {}, id="no-key"),
    ],
)
async def test_the_api_key_travels_as_x_api_key_and_never_as_a_bearer_token(agentenv, api_key, expected):
    """AgentENV checks ``X-API-KEY``. It lets a request with any ``Authorization`` header pass."""
    pool = MicroVMPool(agentenv.url, api_key=api_key)

    await pool.spawn(image="swe-base")

    [request] = agentenv.requests
    assert present(request.headers, "x-api-key", "authorization") == expected


async def test_a_spawn_without_a_template_is_refused_before_any_request(agentenv):
    pool = MicroVMPool(agentenv.url)

    with pytest.raises(ValueError, match="^spawn needs a template or snapshot id$"):
        await pool.spawn(image="")

    assert agentenv.requests == []


@pytest.mark.parametrize(
    "resources, expected_shape",
    [
        pytest.param(None, {}, id="no-shape"),
        pytest.param(
            {"cpu": 2, "memory_mb": 8192, "disk_size_mb": 65536},
            {"cpuCount": 2, "memoryMB": 8192, "diskSizeMB": 65536},
            id="full-shape",
        ),
        pytest.param({"memory_mb": 512.9}, {"memoryMB": 512}, id="memory-only"),
    ],
)
async def test_a_cold_start_asks_for_the_image_with_the_pools_ttl_and_the_shape(
    agentenv, resources, expected_shape
):
    pool = MicroVMPool(agentenv.url, sandbox_ttl=11400)

    await pool.spawn_from_image("registry.example/task:tag", resources=resources)

    [request] = agentenv.requests
    assert (request.path, request.json) == (
        "/sandboxes-cold",
        {
            "image": "registry.example/task:tag",
            "timeout": 11400,
            "autoPause": True,
            "autoResume": {"enabled": True},
            **expected_shape,
        },
    )


# The full product of the two parameters.
@pytest.mark.parametrize(
    "allow, route, expected",
    [
        pytest.param(None, "/sandboxes", {}, id="default-template"),
        pytest.param(None, "/sandboxes-cold", {}, id="default-cold"),
        pytest.param(
            True,
            "/sandboxes",
            {"allowInternetAccess": True, "allow_internet_access": True},
            id="allow-template",
        ),
        pytest.param(True, "/sandboxes-cold", {"allowInternetAccess": True}, id="allow-cold"),
        pytest.param(
            False,
            "/sandboxes",
            {"allowInternetAccess": False, "allow_internet_access": False},
            id="deny-template",
        ),
        pytest.param(False, "/sandboxes-cold", {"allowInternetAccess": False}, id="deny-cold"),
    ],
)
async def test_the_network_policy_reaches_both_create_routes_and_is_absent_by_default(
    agentenv, allow, route, expected
):
    """The deployed POST /sandboxes reads only the snake_case key, and /sandboxes-cold only the camelCase one.
    Without a policy, the server's own default applies."""
    pool = MicroVMPool(agentenv.url, allow_internet=allow)

    await create_through(pool, route)

    [request] = agentenv.requests
    assert (request.path, present(request.json, "allowInternetAccess", "allow_internet_access")) == (
        route,
        expected,
    )


@pytest.mark.parametrize(
    "body", [pytest.param({}, id="empty"), pytest.param({"sandbox_id": "vm-1"}, id="snake-case")]
)
async def test_a_create_reply_without_a_sandbox_id_is_an_error(agentenv, body):
    agentenv.create_body = body
    pool = MicroVMPool(agentenv.url)

    with pytest.raises(RuntimeError, match="^no sandbox id in response: "):
        await pool.spawn(image="swe-base")

    assert pool.list() == []


async def test_a_call_through_a_spawned_sandbox_goes_to_the_proxy_with_the_routing_headers(agentenv):
    pool = MicroVMPool(agentenv.url, runtime_port=3000)
    sandbox = await pool.spawn(image="swe-base", agent_id="reviewer")

    result = await sandbox.call("shell", command="true")

    proxied = agentenv.requests[-1]
    assert result == ToolResult(output="ok", is_error=False)
    assert (
        proxied.path,
        proxied.headers["x-agentenv-sandbox-id"],
        proxied.headers["x-agentenv-target-port"],
    ) == (
        "/",
        "vm-1",
        "3000",
    )
    assert proxied.json["params"] == {
        "name": "shell",
        "arguments": {"command": "true"},
        "agent_id": "reviewer",
    }


async def test_a_destroy_deletes_only_the_named_sandbox_and_drops_its_handle(agentenv):
    pool = MicroVMPool(agentenv.url)
    first, second = await pool.spawn(image="swe-base"), await pool.spawn(image="swe-base")

    await pool.destroy(first)

    assert agentenv.paths("DELETE") == ["/sandboxes/vm-1"]
    assert (pool.list(), first.sandbox_id) == ([second], None)


async def test_a_sandbox_that_the_server_no_longer_has_is_destroyed_without_an_error(agentenv):
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")
    agentenv.delete_status = 404

    await pool.destroy(sandbox)

    assert (pool.list(), sandbox.sandbox_id) == ([], None)


async def test_a_failed_delete_raises_and_keeps_the_sandbox_in_the_pool(agentenv):
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")
    agentenv.delete_status = 500

    with pytest.raises(httpx.HTTPStatusError, match="500 Internal Server Error"):
        await pool.destroy(sandbox)

    assert (pool.list(), sandbox.sandbox_id) == ([sandbox], "vm-1")


async def test_a_destroy_skips_a_handle_that_the_pool_does_not_hold(agentenv):
    pool = MicroVMPool(agentenv.url)
    unregistered = pool.handle("vm-9")

    await pool.destroy(unregistered)

    assert (agentenv.requests, unregistered.sandbox_id) == ([], "vm-9")


async def test_an_attached_sandbox_is_destroyed_when_the_pool_closes(agentenv):
    """A process that created a sandbox can lend it to another process, which attaches to it."""
    pool = MicroVMPool(agentenv.url)
    pool.attach("vm-lent")

    await pool.close()

    assert agentenv.paths("DELETE") == ["/sandboxes/vm-lent"]


async def test_a_forgotten_sandbox_keeps_running_when_the_pool_closes(agentenv):
    """A sandbox whose final snapshot failed holds the only copy of the rollout."""
    pool = MicroVMPool(agentenv.url)
    kept, other = await pool.spawn(image="swe-base"), await pool.spawn(image="swe-base")

    await pool.forget(kept)
    await pool.close()

    assert agentenv.paths("DELETE") == ["/sandboxes/vm-2"]
    assert (kept.sandbox_id, other.sandbox_id, pool.list()) == (None, None, [])


@pytest.mark.parametrize(
    "name, disk_only, expected_body",
    [
        pytest.param(None, False, {}, id="plain"),
        pytest.param("step-3", False, {"name": "step-3"}, id="named"),
        pytest.param(None, True, {"diskOnly": True}, id="disk-only"),
        pytest.param("step-3", True, {"name": "step-3", "diskOnly": True}, id="named-disk-only"),
    ],
)
async def test_a_snapshot_asks_for_the_name_and_the_disk_only_capture(
    agentenv, name, disk_only, expected_body
):
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")

    snapshot = await pool.snapshot(sandbox, name=name, disk_only=disk_only)

    assert (agentenv.requests[-1].path, agentenv.requests[-1].json) == (
        "/sandboxes/vm-1/snapshots",
        expected_body,
    )
    assert snapshot == Snapshot(id="snap-7", rootfs_layers=4, memory_layers=2, disk_only=disk_only)


@pytest.mark.parametrize(
    "source, expected_disk_only",
    [
        pytest.param(Snapshot(id="snap-7", disk_only=True), True, id="disk-only-snapshot"),
        pytest.param(Snapshot(id="snap-7", disk_only=False), False, id="full-snapshot"),
        pytest.param("snap-7", False, id="bare-id"),
    ],
)
async def test_a_squash_of_a_snapshot_keeps_whether_it_is_disk_only(agentenv, source, expected_disk_only):
    pool = MicroVMPool(agentenv.url)

    squashed = await pool.squash(source, name="flat")

    [request] = agentenv.requests
    assert (request.path, request.json) == ("/snapshots/snap-7/squash", {"name": "flat"})
    assert squashed == Snapshot(
        id="snap-flat", rootfs_layers=1, memory_layers=1, disk_only=expected_disk_only
    )


async def test_a_snapshot_of_a_destroyed_sandbox_is_refused(agentenv):
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")
    await pool.destroy(sandbox)

    with pytest.raises(RuntimeError, match=r"^sandbox has no id \(already destroyed\?\)$"):
        await pool.snapshot(sandbox)


async def test_an_upload_posts_the_host_file_to_the_file_service_of_the_sandbox(agentenv, tmp_path):
    """The file service needs no runtime, and a fixture can be binary and several megabytes."""
    source = tmp_path / "fixture.bin"
    source.write_bytes(bytes(range(256)) * 64)
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")

    await pool.upload_file(sandbox, source, "/testbed/fixture.bin")

    upload = agentenv.requests[-1]
    assert (upload.method, upload.path, upload.params) == ("POST", "/files", {"path": "/testbed/fixture.bin"})
    assert (upload.headers["x-agentenv-sandbox-id"], upload.headers["x-agentenv-target-port"]) == (
        "vm-1",
        "49983",
    )
    assert upload.files == {"file": ("fixture.bin", bytes(range(256)) * 64)}


async def test_a_download_writes_the_guest_file_to_the_host_path(agentenv, tmp_path):
    agentenv.file = bytes(range(256)) * 4096
    destination = tmp_path / "logs.tar.gz"
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")

    await pool.download_file(sandbox, "/tmp/logs.tar.gz", destination)

    download = agentenv.requests[-1]
    assert (download.method, download.path, download.params) == (
        "GET",
        "/files",
        {"path": "/tmp/logs.tar.gz"},
    )
    assert (download.headers["x-agentenv-sandbox-id"], download.headers["x-agentenv-target-port"]) == (
        "vm-1",
        "49983",
    )
    assert destination.read_bytes() == bytes(range(256)) * 4096


@pytest.mark.parametrize(
    "status, cut_at",
    [pytest.param(404, None, id="error-status"), pytest.param(200, 1000, id="connection-lost")],
)
async def test_a_failed_download_raises_and_leaves_the_host_path_as_it_was(
    agentenv, tmp_path, status, cut_at
):
    agentenv.file, agentenv.file_status, agentenv.file_cut_at = bytes(range(256)) * 4096, status, cut_at
    destination = tmp_path / "logs.tar.gz"
    destination.write_bytes(b"previous archive")
    pool = MicroVMPool(agentenv.url)
    sandbox = await pool.spawn(image="swe-base")

    with pytest.raises(httpx.HTTPError):
        await pool.download_file(sandbox, "/tmp/logs.tar.gz", destination)

    assert destination.read_bytes() == b"previous archive"
    assert sorted(tmp_path.iterdir()) == [destination], "no partial file is left beside it"
