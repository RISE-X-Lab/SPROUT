"""``PodmanPool`` against the fake ``podman``, ``btrfs`` and ``findmnt``, with a fake runtime in each sandbox.

The fakes record each command, so a test can check the exact command line that the pool sends, and
the order of the commands across the tools. ``test_podman_live.py`` runs the pool against the real
tools. ``conftest.py`` gives ``pool``: an online pool on ``podman_store`` whose runtimes answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout_sandbox import HttpBackend, PodmanPool, Snapshot
from sprout_sandbox.btrfs_store import config_path
from sprout_sandbox.podman_cli import PodmanError
from sprout_sandbox.podman_containers import PodmanContainers
from tests.sprout_sandbox.fake_runtime import SocketWatcher
from tests.support.tools import recorded_sequence

pytestmark = pytest.mark.anyio

IMAGE = "docker.io/library/python:3.11-slim"
#: The OCI config subset that the fake podman reports for every image whose User is ``appuser``.
APPUSER_CONFIG = {
    "Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
    "WorkingDir": "/testbed",
    "User": "appuser",
    "Entrypoint": None,
    "Cmd": ["python3"],
}
#: A sandbox id that no pool minted, in the form that a pool mints.
SANDBOX_ID = "sprout-0123456789ab"


def the_call(calls: list[list[str]], command: str) -> list[str]:
    """The one call among ``calls`` whose first word is ``command``."""
    [call] = [call for call in calls if call[0] == command]
    return call


def starts(calls: list[list[str]], sandbox_id: str) -> list[list[str]]:
    """The calls that create, connect or start the container ``sandbox_id``, after its network exists.

    The readiness probe can ask podman for the container's state. Those calls are left out."""
    created = [
        index
        for index, call in enumerate(calls)
        if call[:2] == ["network", "create"] and call[-1] == sandbox_id
    ]
    return [call for call in calls[created[0] + 1 :] if call[0] != "inspect"]


def shape_flags(run: list[str]) -> list[str]:
    """The arguments of a ``run`` call between the image's working directory and ``--rootfs``."""
    return run[run.index("--workdir") + 2 : run.index("--rootfs")]


def cwds_of(pairs: list[tuple[str, str]], commands: set[str]) -> set[str]:
    """The working directories of the calls among ``pairs`` whose first word is in ``commands``."""
    return {cwd for command, cwd in pairs if command in commands}


def cwds_except(pairs: list[tuple[str, str]], commands: set[str]) -> set[str]:
    """The working directories of the calls among ``pairs`` whose first word is not in ``commands``."""
    return {cwd for command, cwd in pairs if command not in commands}


def names_in(directory: Path) -> list[str]:
    """The names of the entries of ``directory``, sorted."""
    return sorted(path.name for path in directory.iterdir())


def image_commands(calls: list[list[str]]) -> list[list[str]]:
    """The calls among ``calls`` that inspect, pull or export an image."""
    return [call for call in calls if call[0] in ("image", "pull", "export")]


def run_sandbox_ids(calls: list[list[str]]) -> list[str]:
    """The name of each container that a ``run`` or ``create --name`` call among ``calls`` started."""
    return [
        call[call.index("--name") + 1] for call in calls if call[0] in ("run", "create") and "--name" in call
    ]


def rootfs_left(sandbox_id: str, *, store: Path) -> str:
    """The error of a destroy of ``sandbox_id`` whose rootfs the fake btrfs failed to delete."""
    return (
        "container %s is removed but its rootfs %s/sandboxes/%s is not (btrfs subvolume delete failed "
        "(exit 1): ERROR: simulated delete failure); `harness reap --store %s` reclaims it"
        % (sandbox_id, store, sandbox_id, store)
    )


def touch_oom_marker(directory: Path) -> None:
    """Do what conmon does in its working directory when the kernel OOM-kills a process of the container."""
    (directory / "oom").touch()


@pytest.fixture
def oom_kills(podman_store: Path) -> Iterator[None]:
    """A conmon stand-in that marks each sandbox of ``podman_store`` as OOM-killed, as soon as it appears."""
    conmon = SocketWatcher(podman_store / "sockets", touch_oom_marker)
    yield
    conmon.close()


def test_a_pool_refuses_a_runtime_binary_that_does_not_exist(fake_podman, podman_store, tmp_path):
    with pytest.raises(ValueError) as raised:
        PodmanPool(podman_store, tmp_path / "absent")

    assert str(raised.value) == (
        "runtime_bin does not exist: %s (cd runtime && go generate ./... && go build -o sprout-runtime .)"
        % (tmp_path / "absent")
    )


def test_a_pool_refuses_a_store_that_does_not_exist(fake_podman, podman_store, fake_runtime_bin):
    with pytest.raises(ValueError) as raised:
        PodmanPool(podman_store / "absent", fake_runtime_bin)

    assert str(raised.value) == (
        "store does not exist: %s (create one with sudo ops/btrfs_store.sh <mount point> <image file> <size>)"
        % (podman_store / "absent")
    )


def test_the_pool_offers_snapshots_cold_starts_uploads_and_downloads(pool):
    capabilities = (
        pool.supports_snapshot(),
        pool.supports_cold_start(),
        pool.supports_upload(),
        pool.supports_download(),
    )

    assert capabilities == (True, True, True, True)


async def test_a_spawn_runs_the_runtime_on_the_rootfs_with_the_images_env_and_workdir_and_the_callers(
    pool, fake_podman, podman_store, fake_runtime_bin
):
    """The image's User is not passed: the runtime stays the container's root, so that the root steps
    of a Harbor trial work on a non-root image. The harness drops to that user for each command."""
    fake_podman.image_user("appuser")

    sandbox = await pool.spawn(IMAGE, env={"A": "1", "B": "x y"}, resources={"cpu": 2, "memory_mb": 4096})

    sid = sandbox.sandbox_id
    assert the_call(fake_podman.calls(), "run") == [
        "run", "-d", "--name", sid, "--network", sid,
        "--label", "sprout.managed=1", "--label", "sprout.store=%s" % podman_store,
        "-v", "%s:/run/sprout" % (podman_store / "sockets" / sid),
        "-v", "%s:/usr/local/bin/sprout-runtime:ro" % fake_runtime_bin,
        "--entrypoint", "/usr/local/bin/sprout-runtime",
        "-e", "PATH=/usr/local/bin:/usr/bin", "-e", "LANG=C.UTF-8", "-e", "FOO=a=b c",
        "-e", "A=1", "-e", "B=x y",
        "--workdir", "/testbed",
        "--cpus", "2", "--memory", "4096m",
        "--rootfs", str(podman_store / "sandboxes" / sid), "--listen", "unix:/run/sprout/runtime.sock",
    ]  # fmt: skip


async def test_each_sandbox_gets_a_network_of_its_own_before_its_container_starts(
    pool, fake_podman, podman_store
):
    """No other sandbox joins the network, so none can move its gateway. The network has no podman DNS
    server and no route to the network of another sandbox."""
    sandbox = await pool.spawn(IMAGE)

    calls = fake_podman.calls()
    network = [
        "network", "create", "--disable-dns", "--opt", "isolate=true",
        "--label", "sprout.managed=1", "--label", "sprout.store=%s" % podman_store, sandbox.sandbox_id,
    ]  # fmt: skip
    assert calls.index(network) < calls.index(the_call(calls, "run"))


@pytest.mark.parametrize(
    "resources, expected_flags",
    [
        pytest.param(None, [], id="no-shape"),
        pytest.param({"cpu": 1.5}, ["--cpus", "1.5"], id="cpu"),
        pytest.param({"memory_mb": 512.9}, ["--memory", "512m"], id="memory"),
        pytest.param({"disk_size_mb": 65536}, [], id="disk-has-no-podman-flag"),
    ],
)
async def test_the_sandbox_shape_becomes_podmans_cpu_and_memory_limits(
    pool, fake_podman, resources, expected_flags
):
    await pool.spawn_from_image(IMAGE, resources=resources)

    assert shape_flags(the_call(fake_podman.calls(), "run")) == expected_flags


async def test_a_spawn_returns_once_the_runtime_has_answered_its_tool_list(pool, runtimes):
    """``podman run -d`` returns before the runtime listens. An earlier first call would fail to connect."""
    sandbox = await pool.spawn(IMAGE)

    assert runtimes.of(sandbox.sandbox_id).bodies() == [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]


async def test_a_call_through_a_sandbox_reaches_its_runtime_over_its_socket_as_its_agent(pool, runtimes):
    sandbox = await pool.spawn(IMAGE, agent_id="main")

    await sandbox.call("shell", command="ls")

    assert runtimes.of(sandbox.sandbox_id).bodies()[-1]["params"] == {
        "name": "shell",
        "arguments": {"command": "ls"},
        "agent_id": "main",
    }


async def test_an_offline_sandbox_is_created_and_disconnected_before_it_starts(
    fake_podman, podman_store, fake_runtime_bin, runtimes
):
    """``run`` would give the container its network for an instant. Its first command must run offline."""
    pool = PodmanPool(podman_store, fake_runtime_bin, allow_internet=False)

    sandbox = await pool.spawn(IMAGE)

    sid = sandbox.sandbox_id
    [create, disconnect, start] = starts(fake_podman.calls(), sid)
    assert (create[:3], "-d" in create) == (["create", "--name", sid], False)
    assert (disconnect, start) == (["network", "disconnect", sid, sid], ["start", sid])


@pytest.mark.parametrize(
    "allow_internet, container_commands",
    [pytest.param(None, {"run"}, id="online"), pytest.param(False, {"create", "start"}, id="offline")],
)
async def test_podman_runs_from_the_store_and_a_container_starts_from_its_socket_directory(
    fake_podman,
    podman_store,
    fake_runtime_bin,
    runtimes,
    tmp_path,
    monkeypatch,
    allow_internet,
    container_commands,
):
    """conmon writes its ``oom`` marker into the directory that podman runs from. No podman command runs
    from the caller's directory, which can be a source checkout."""
    monkeypatch.chdir(tmp_path)
    pool = PodmanPool(podman_store, fake_runtime_bin, allow_internet=allow_internet)
    await pool.prepare(IMAGE)
    before = len(fake_podman.cwds())

    sandbox = await pool.spawn(IMAGE)

    spawned = fake_podman.cwds()[before:]
    assert cwds_of(spawned, container_commands) == {str(podman_store / "sockets" / sandbox.sandbox_id)}
    assert cwds_except(spawned, container_commands) == {str(podman_store)}


async def test_a_spawn_without_an_image_or_a_snapshot_is_refused(pool, fake_podman):
    with pytest.raises(ValueError, match="^spawn needs an image reference or a snapshot id$"):
        await pool.spawn("")

    assert fake_podman.calls() == []


async def test_a_runtime_that_exits_before_answering_fails_the_spawn_at_once_with_its_state_and_logs(
    fake_podman, podman_store, fake_runtime_bin
):
    """A runtime that cannot start, such as one linked to glibc in a musl image, fails the spawn at once."""
    fake_podman.container_state("exited 127")
    pool = PodmanPool(podman_store, fake_runtime_bin)

    with pytest.raises(PodmanError) as raised:
        await pool.spawn(IMAGE)

    [sid] = run_sandbox_ids(fake_podman.calls())
    assert str(raised.value) == (
        "runtime in %s exited before answering; container exited (exit 127); logs: fake logs" % sid
    )


async def test_a_runtime_killed_for_memory_fails_the_spawn_and_the_error_says_so(
    fake_podman, podman_store, fake_runtime_bin, oom_kills
):
    fake_podman.container_state("exited 137")
    pool = PodmanPool(podman_store, fake_runtime_bin)

    with pytest.raises(PodmanError) as raised:
        await pool.spawn(IMAGE)

    [sid] = run_sandbox_ids(fake_podman.calls())
    assert str(raised.value) == (
        "runtime in %s exited before answering; container exited (exit 137); "
        "a process in it was OOM-killed (conmon's oom marker); logs: fake logs" % sid
    )


async def test_a_runtime_that_never_answers_fails_the_spawn_after_the_deadline(
    fake_podman, podman_store, fake_runtime_bin, monkeypatch
):
    monkeypatch.setattr(PodmanContainers, "READY_TIMEOUT_S", 0.3)
    pool = PodmanPool(podman_store, fake_runtime_bin)

    with pytest.raises(
        PodmanError, match=r"did not answer within 0 s \(.+\); container running \(exit 0\); logs: fake logs$"
    ):
        await pool.spawn(IMAGE)


async def test_a_failed_spawn_leaves_no_container_network_rootfs_or_socket_directory(
    fake_podman, podman_store, fake_runtime_bin
):
    """The caller gets only the exception, and cannot reach the sandbox to destroy it."""
    fake_podman.container_state("exited 127")
    pool = PodmanPool(podman_store, fake_runtime_bin)

    with pytest.raises(PodmanError):
        await pool.spawn(IMAGE)

    [sid] = run_sandbox_ids(fake_podman.calls())
    assert ["rm", "-f", sid] in fake_podman.calls()
    assert fake_podman.networks() == {}
    assert names_in(podman_store / "sandboxes") == names_in(podman_store / "sockets") == []
    assert pool.list() == []


def test_the_runtime_socket_of_a_sandbox_is_in_the_sockets_directory_of_the_store(
    fake_podman, podman_store, fake_runtime_bin
):
    pool = PodmanPool(podman_store, fake_runtime_bin)

    sandbox = pool.attach(SANDBOX_ID)

    assert isinstance(sandbox.backend, HttpBackend)
    assert sandbox.backend.uds == str(podman_store / "sockets" / SANDBOX_ID / "runtime.sock")


def test_a_pool_given_a_socket_directory_puts_the_runtime_socket_of_each_sandbox_there(
    fake_podman, podman_store, fake_runtime_bin, tmp_path
):
    pool = PodmanPool(podman_store, fake_runtime_bin, socket_dir=tmp_path / "sockets")

    sandbox = pool.attach(SANDBOX_ID)

    assert isinstance(sandbox.backend, HttpBackend)
    assert sandbox.backend.uds == str(tmp_path / "sockets" / SANDBOX_ID / "runtime.sock")


async def test_snapshots_of_a_sandbox_are_disk_only_and_numbered_in_order(pool):
    """A snapshot stacks no layers, so it reports no layer count, and the checkpointer never re-boards."""
    sandbox = await pool.spawn(IMAGE)

    snapshots = [await pool.snapshot(sandbox), await pool.snapshot(sandbox), await pool.snapshot(sandbox)]

    sid = sandbox.sandbox_id
    assert snapshots == [
        Snapshot(id=sid + "-0001", disk_only=True),
        Snapshot(id=sid + "-0002", disk_only=True),
        Snapshot(id=sid + "-0003", disk_only=True),
    ]


@pytest.mark.parametrize(
    "name, suffix",
    [
        pytest.param("final", "-final", id="safe"),
        pytest.param("Run_01:step 7/final", "-Run_01-step-7-final", id="unsafe-characters"),
        pytest.param("-.trimmed.-", "-trimmed", id="leading-and-trailing"),
        pytest.param("../..", "-snapshot", id="nothing-safe"),
        pytest.param("x" * 49, "-" + "x" * 48, id="over-48"),
    ],
)
async def test_a_snapshot_name_becomes_a_path_safe_suffix_of_its_id(pool, name, suffix):
    sandbox = await pool.spawn(IMAGE)

    snapshot = await pool.snapshot(sandbox, name=name)

    assert snapshot.id == sandbox.sandbox_id + "-0001" + suffix


async def test_a_snapshot_is_a_read_only_btrfs_snapshot_of_the_rootfs_with_the_image_config_beside_it(
    pool, fake_btrfs, podman_store
):
    sandbox = await pool.spawn(IMAGE)
    rootfs = podman_store / "sandboxes" / sandbox.sandbox_id

    snapshot = await pool.snapshot(sandbox)

    target = podman_store / "snapshots" / snapshot.id
    assert fake_btrfs.calls()[-1] == ["subvolume", "snapshot", "-r", str(rootfs), str(target)]
    assert config_path(target).read_text() == config_path(rootfs).read_text()


async def test_a_snapshot_with_memory_is_refused_and_makes_nothing(pool, podman_store):
    """A btrfs snapshot holds files, not processes. The session records the step as not captured."""
    sandbox = await pool.spawn(IMAGE)

    with pytest.raises(NotImplementedError, match="^podman snapshots are disk-only"):
        await pool.snapshot(sandbox, disk_only=False)

    assert names_in(podman_store / "snapshots") == []


async def test_a_rootfs_that_holds_subvolumes_is_not_snapshotted(pool, podman_store):
    """A btrfs snapshot shows a nested subvolume as an empty directory, so a branch would lack those files."""
    sandbox = await pool.spawn(IMAGE)
    rootfs = podman_store / "sandboxes" / sandbox.sandbox_id
    await pool.store.create(rootfs / "adv")
    await pool.store.create(rootfs / "adv" / "deeper")

    with pytest.raises(PodmanError) as raised:
        await pool.snapshot(sandbox)

    assert str(raised.value) == (
        "rootfs of %s holds nested subvolumes (/adv/deeper, /adv); a snapshot would omit their contents"
        % sandbox.sandbox_id
    )
    assert (names_in(podman_store / "snapshots"), pool.list()) == ([], [sandbox])


async def test_a_spawn_from_an_image_starts_on_a_writable_snapshot_of_the_images_base(
    pool, fake_btrfs, podman_store
):
    base = await pool.prepare(IMAGE)

    sandbox = await pool.spawn(IMAGE)

    rootfs = podman_store / "sandboxes" / sandbox.sandbox_id
    assert fake_btrfs.calls()[-1] == ["subvolume", "snapshot", str(base), str(rootfs)]


async def test_a_spawn_from_a_snapshot_id_starts_on_a_writable_snapshot_of_it_without_an_image(
    pool, fake_podman, fake_btrfs, podman_store
):
    parent = await pool.spawn(IMAGE)
    snapshot = await pool.snapshot(parent)
    before = len(fake_podman.calls())

    branch = await pool.spawn(snapshot.id)

    assert fake_btrfs.calls()[-1] == [
        "subvolume",
        "snapshot",
        str(podman_store / "snapshots" / snapshot.id),
        str(podman_store / "sandboxes" / branch.sandbox_id),
    ]
    assert image_commands(fake_podman.calls()[before:]) == []


@pytest.mark.parametrize(
    "name", [pytest.param(None, id="plain"), pytest.param("Run_01:step 7/final", id="named")]
)
async def test_the_id_of_a_deleted_snapshot_is_a_missing_snapshot_not_an_image_to_pull(
    pool, fake_podman, podman_store, name
):
    """After a sweep, or on another store. Given to podman, the id would fail with a registry error."""
    sandbox = await pool.spawn(IMAGE)
    snapshot = await pool.snapshot(sandbox, name=name)
    await pool.delete_snapshot(snapshot.id)
    before = len(fake_podman.calls())

    with pytest.raises(PodmanError) as raised:
        await pool.spawn(snapshot.id)

    missing = "no snapshot %r in %s/snapshots (deleted, swept, or taken on another store)"
    assert str(raised.value) == missing % (snapshot.id, podman_store)
    assert fake_podman.calls()[before:] == []


async def test_a_bare_name_that_is_not_a_snapshot_of_the_store_is_a_docker_image_reference(pool, fake_podman):
    await pool.spawn("alpine")

    assert fake_podman.calls()[0] == ["image", "inspect", "--format", "json", "docker.io/library/alpine"]


async def test_a_branch_and_an_attached_sandbox_report_the_image_config_of_their_line(
    pool, fake_podman, podman_store, fake_runtime_bin
):
    """The commands of a branch and of a kept sandbox run as the image's User, with its Env and WorkingDir."""
    fake_podman.image_user("appuser")
    parent = await pool.spawn(IMAGE)
    branch = await pool.spawn((await pool.snapshot(parent)).id)
    kept = PodmanPool(podman_store, fake_runtime_bin).attach(branch.sandbox_id)
    before = len(fake_podman.calls())

    configs = [pool.image_config(parent), pool.image_config(branch), pool.image_config(kept)]

    assert configs == [APPUSER_CONFIG, APPUSER_CONFIG, APPUSER_CONFIG]
    assert fake_podman.calls()[before:] == [], "the config comes from the store"


async def test_a_deleted_snapshot_is_gone_and_a_sandbox_spawned_from_it_keeps_its_files(pool, podman_store):
    parent = await pool.spawn(IMAGE)
    (podman_store / "sandboxes" / parent.sandbox_id / "answer.txt").write_text("ready\n")
    snapshot = await pool.snapshot(parent)
    branch = await pool.spawn(snapshot.id)

    await pool.delete_snapshot(snapshot.id)

    assert names_in(podman_store / "snapshots") == []
    assert (podman_store / "sandboxes" / branch.sandbox_id / "answer.txt").read_text() == "ready\n"


@pytest.mark.parametrize(
    "allow_internet, allow",
    [pytest.param(None, True, id="online"), pytest.param(False, False, id="offline")],
)
async def test_setting_the_network_that_a_sandbox_already_has_runs_no_command(
    fake_podman, podman_store, fake_runtime_bin, runtimes, allow_internet, allow
):
    pool = PodmanPool(podman_store, fake_runtime_bin, allow_internet=allow_internet)
    sandbox = await pool.spawn(IMAGE)
    before = len(fake_podman.calls())

    await pool.set_network(sandbox, allow)

    assert fake_podman.calls()[before:] == []


async def test_setting_the_network_disconnects_and_reconnects_the_sandboxs_own_network(pool, fake_podman):
    sandbox = await pool.spawn(IMAGE)
    before = len(fake_podman.calls())

    await pool.set_network(sandbox, False)
    await pool.set_network(sandbox, True)

    sid = sandbox.sandbox_id
    assert fake_podman.calls()[before:] == [
        ["network", "disconnect", sid, sid],
        ["network", "connect", sid, sid],
    ]


@pytest.mark.parametrize(
    "allow_internet, allow",
    [pytest.param(None, False, id="is-not-connected"), pytest.param(False, True, id="already-connected")],
)
async def test_a_refusal_of_podman_because_the_network_is_already_so_counts_as_done(
    fake_podman, podman_store, fake_runtime_bin, runtimes, allow_internet, allow
):
    pool = PodmanPool(podman_store, fake_runtime_bin, allow_internet=allow_internet)
    sandbox = await pool.spawn(IMAGE)
    fake_podman.refuse_network_changes()
    await pool.set_network(sandbox, allow)
    before = len(fake_podman.calls())

    await pool.set_network(sandbox, allow)

    assert fake_podman.calls()[before:] == []


async def test_a_network_change_that_fails_for_another_reason_raises_and_changes_nothing(pool, fake_podman):
    sandbox = await pool.spawn(IMAGE)
    sid = sandbox.sandbox_id
    fake_podman.fail("network")

    with pytest.raises(
        PodmanError, match=r"^podman network disconnect failed \(exit 125\): Error: simulated"
    ):
        await pool.set_network(sandbox, False)

    fake_podman.fail()
    before = len(fake_podman.calls())
    await pool.set_network(sandbox, False)
    assert fake_podman.calls()[before:] == [["network", "disconnect", sid, sid]], (
        "the sandbox is still online"
    )


async def test_a_destroy_removes_the_container_first_then_its_network_and_its_rootfs(
    pool, podman_store, tmp_path
):
    """A mounted subvolume cannot be deleted, and podman removes no network that a container is on.
    ``network rm -f`` would remove the containers on the network."""
    sandbox = await pool.spawn(IMAGE)
    sid, rootfs = sandbox.sandbox_id, str(podman_store / "sandboxes" / sandbox.sandbox_id)
    before = len(recorded_sequence(tmp_path))

    await pool.destroy(sandbox)

    assert recorded_sequence(tmp_path)[before:] == [
        ["podman", "rm", "-f", sid],
        ["podman", "network", "rm", sid],
        ["podman", "unshare", "find", rootfs, "-mindepth", "1", "-inum", "256"],
        ["btrfs", "property", "set", rootfs, "ro", "false"],
        ["btrfs", "subvolume", "delete", rootfs],
    ]


async def test_a_destroyed_sandbox_leaves_no_rootfs_socket_directory_or_network_and_its_handle_names_none(
    pool, fake_podman, podman_store
):
    sandbox = await pool.spawn(IMAGE)

    await pool.destroy(sandbox)

    assert names_in(podman_store / "sandboxes") == names_in(podman_store / "sockets") == []
    assert fake_podman.networks() == {}
    assert (pool.list(), sandbox.sandbox_id) == ([], None)


async def test_a_destroy_deletes_the_subvolumes_inside_a_rootfs_deepest_first_and_then_the_rootfs(
    pool, podman_store, tmp_path
):
    """A subvolume that holds another cannot be deleted. The container's subvolumes belong to ids of
    its user namespace, so they are deleted from inside it."""
    sandbox = await pool.spawn(IMAGE)
    sid, rootfs = sandbox.sandbox_id, podman_store / "sandboxes" / sandbox.sandbox_id
    await pool.store.create(rootfs / "adv")
    await pool.store.create(rootfs / "adv" / "deeper")
    before = len(recorded_sequence(tmp_path))

    await pool.destroy(sandbox)

    deeper, adv = str(rootfs / "adv" / "deeper"), str(rootfs / "adv")
    assert recorded_sequence(tmp_path)[before:] == [
        ["podman", "rm", "-f", sid],
        ["podman", "network", "rm", sid],
        ["podman", "unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"],
        ["podman", "unshare", "btrfs", "property", "set", deeper, "ro", "false"],
        ["podman", "unshare", "btrfs", "subvolume", "delete", deeper],
        ["podman", "unshare", "btrfs", "property", "set", adv, "ro", "false"],
        ["podman", "unshare", "btrfs", "subvolume", "delete", adv],
        ["btrfs", "property", "set", str(rootfs), "ro", "false"],
        ["btrfs", "subvolume", "delete", str(rootfs)],
    ]
    assert not rootfs.exists()


async def test_a_destroy_reclaims_the_rootfs_of_a_container_that_is_already_gone(
    pool, fake_podman, podman_store
):
    sandbox = await pool.spawn(IMAGE)
    fake_podman.lose_containers()

    await pool.destroy(sandbox)

    assert (names_in(podman_store / "sandboxes"), pool.list()) == ([], [])


@pytest.mark.parametrize(
    "podman_fails, btrfs_fails, expected, network_left, rootfs_left",
    [
        pytest.param(
            ("network",),
            (),
            "container {sid} is removed but its network {sid} is not (podman network rm failed (exit 125): "
            "Error: simulated network failure); `harness reap --store {store}` reclaims it",
            True,
            False,
            id="network",
        ),
        pytest.param(
            (),
            ("delete",),
            "container {sid} is removed but its rootfs {rootfs} is not (btrfs subvolume delete failed "
            "(exit 1): ERROR: simulated delete failure); `harness reap --store {store}` reclaims it",
            False,
            True,
            id="rootfs",
        ),
        pytest.param(
            ("network",),
            ("delete",),
            "container {sid} is removed but its network {sid} is not (podman network rm failed (exit 125): "
            "Error: simulated network failure) and its rootfs {rootfs} is not (btrfs subvolume delete failed "
            "(exit 1): ERROR: simulated delete failure); `harness reap --store {store}` reclaims them",
            True,
            True,
            id="both",
        ),
    ],
)
async def test_a_teardown_that_fails_half_way_drops_the_handle_and_names_what_is_left(
    pool,
    fake_podman,
    fake_btrfs,
    podman_store,
    podman_fails,
    btrfs_fails,
    expected,
    network_left,
    rootfs_left,
):
    """A retry cannot finish the teardown, and a pool that kept the sandbox would fail each close again."""
    sandbox = await pool.spawn(IMAGE)
    sid, rootfs = sandbox.sandbox_id, podman_store / "sandboxes" / sandbox.sandbox_id
    fake_podman.fail(*podman_fails)
    fake_btrfs.fail(*btrfs_fails)

    with pytest.raises(PodmanError) as raised:
        await pool.destroy(sandbox)

    assert str(raised.value) == expected.format(sid=sid, rootfs=rootfs, store=podman_store)
    assert (sid in fake_podman.networks(), rootfs.is_dir()) == (network_left, rootfs_left)
    assert (pool.list(), sandbox.sandbox_id, names_in(podman_store / "sockets")) == ([], None, [])


async def test_closing_a_pool_tries_every_sandbox_and_names_each_leftover(pool, fake_btrfs, podman_store):
    first, second = await pool.spawn(IMAGE), await pool.spawn(IMAGE)
    first_id, second_id = first.sandbox_id, second.sandbox_id
    fake_btrfs.fail("delete")

    with pytest.raises(PodmanError) as raised:
        await pool.close()

    assert str(raised.value) == "%s; %s" % (
        rootfs_left(first_id, store=podman_store),
        rootfs_left(second_id, store=podman_store),
    )
    assert pool.list() == []


async def test_a_forgotten_sandbox_keeps_running_when_the_pool_closes(pool, fake_podman, podman_store):
    """A sandbox whose final snapshot failed holds the only copy of the rollout."""
    kept, other = await pool.spawn(IMAGE), await pool.spawn(IMAGE)
    kept_id, other_id = kept.sandbox_id, other.sandbox_id
    before = len(fake_podman.calls())

    await pool.forget(kept)
    await pool.close()

    assert fake_podman.calls()[before:][0] == ["rm", "-f", other_id]
    assert ["rm", "-f", kept_id] not in fake_podman.calls()
    assert (names_in(podman_store / "sandboxes"), names_in(podman_store / "sockets")) == (
        [kept_id, kept_id + ".json"],
        [kept_id],
    )
    assert (kept.sandbox_id, pool.list()) == (None, [])


async def test_a_destroy_skips_a_sandbox_that_the_pool_does_not_hold(
    pool, fake_podman, podman_store, fake_runtime_bin
):
    foreign = PodmanPool(podman_store, fake_runtime_bin).attach(SANDBOX_ID)

    await pool.destroy(foreign)

    assert (fake_podman.calls(), foreign.sandbox_id) == ([], SANDBOX_ID)


async def test_a_sandbox_attached_by_id_is_destroyed_like_the_pools_own(
    pool, fake_podman, podman_store, fake_runtime_bin
):
    """The grade of a kept sandbox attaches to it in another process, and destroys it after the verifier."""
    sandbox = await pool.spawn(IMAGE)
    grader = PodmanPool(podman_store, fake_runtime_bin)
    kept = grader.attach(sandbox.sandbox_id)

    await grader.destroy(kept)

    assert ["rm", "-f", sandbox.sandbox_id] in fake_podman.calls()
    assert (names_in(podman_store / "sandboxes"), grader.list(), kept.sandbox_id) == ([], [], None)


async def test_a_call_through_an_attached_sandbox_reaches_its_runtime_as_the_attaching_agent(
    pool, runtimes, podman_store, fake_runtime_bin
):
    sandbox = await pool.spawn(IMAGE)
    kept = PodmanPool(podman_store, fake_runtime_bin).attach(sandbox.sandbox_id, agent_id="grader")

    await kept.call("shell", command="bash /tests/test.sh")

    assert runtimes.of(sandbox.sandbox_id).bodies()[-1]["params"] == {
        "name": "shell",
        "arguments": {"command": "bash /tests/test.sh"},
        "agent_id": "grader",
    }


async def test_destroying_one_sandbox_keeps_the_network_of_another(pool, fake_podman):
    first, second = await pool.spawn(IMAGE), await pool.spawn(IMAGE)
    second_id = second.sandbox_id

    await pool.destroy(first)

    assert list(fake_podman.networks()) == [second_id]


async def test_host_files_go_in_and_out_with_podman_cp_and_not_through_the_runtime(
    pool, fake_podman, tmp_path
):
    """The tool surface cannot carry a binary fixture of several megabytes."""
    sandbox = await pool.spawn(IMAGE)
    sid = sandbox.sandbox_id

    await pool.upload_file(sandbox, tmp_path / "fixture.bin", "/testbed/fixture.bin")
    await pool.download_file(sandbox, "/testbed/model.patch", tmp_path / "model.patch")

    assert fake_podman.calls()[-2:] == [
        ["cp", str(tmp_path / "fixture.bin"), "%s:/testbed/fixture.bin" % sid],
        ["cp", "%s:/testbed/model.patch" % sid, str(tmp_path / "model.patch")],
    ]
