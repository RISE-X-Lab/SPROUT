# ruff: noqa: E402
"""Large writes remain bounded, atomic, and on the owned ToolAccess path."""

import asyncio
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("opencollab")
from opencollab.adapters.env import ExecResult

from sprout.worker.agents.opencollab import environment
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.tools.access import ToolAccess

from .test_opencollab import Model, Sandbox, call, response, run


@pytest.fixture
def port(tmp_path):
    sandbox = Sandbox(tmp_path)
    with JournalWriter(tmp_path / "calls.jsonl") as journal:
        control = RunControl()
        tools = ToolAccess(sandbox, journal, control=control)
        env = environment.SproutEnvironment(
            tools,
            str(tmp_path),
            control,
            time.monotonic() + 120,
        )
        env.turn_id = "large-write"
        yield env


@pytest.mark.parametrize("size", [0, 32769, 300000, 4 * 1024 * 1024])
def test_large_utf8_roundtrip_bounded_owned_calls(tmp_path, port, size):
    target = tmp_path / "quoted ' 中文" / "file.txt"
    content = "中" * (size // 3) + "x" * (size % 3)
    asyncio.run(port.write_file(str(target), content))
    assert target.read_bytes() == content.encode()
    calls = [row for row in read_journal(tmp_path / "calls.jsonl") if row["type"] == "tool.started"]
    assert calls
    assert max(len(row["args"]["command"].encode()) for row in calls) < 65536
    assert len(calls) == 2 + (size + 32767) // 32768
    assert not list(target.parent.glob(".sprout-write-*"))


def test_existing_executable_permissions_are_preserved(tmp_path, port):
    target = tmp_path / "script"
    target.write_text("original")
    target.chmod(0o751)
    asyncio.run(port.write_file(str(target), "new" * 100000))
    assert target.stat().st_mode & 0o777 == 0o751


@pytest.mark.parametrize("failure", ["chunk", "checksum", "cancel"])
def test_failure_keeps_original_and_cleans_owned_stage(tmp_path, port, failure):
    target = tmp_path / "target"
    target.write_text("original")
    execute = port.exec_cmd
    chunks = 0

    async def inject(command, timeout=120):
        nonlocal chunks
        if "base64 -d >>" in command:
            chunks += 1
            if chunks == 2:
                if failure == "chunk":
                    return ExecResult(1, "", "injected append failure")
                if failure == "cancel":
                    raise asyncio.CancelledError()
                command = command.replace("base64 -d >>", "head -c 1 >>")
        return await execute(command, timeout)

    port.exec_cmd = inject
    error = asyncio.CancelledError if failure == "cancel" else OSError
    with pytest.raises(error):
        asyncio.run(port.write_file(str(target), "new" * 100000))
    assert target.read_text() == "original"
    assert not list(tmp_path.glob(".sprout-write-*"))


def test_collision_does_not_overwrite_or_delete_foreign_stage(tmp_path, port, monkeypatch):
    monkeypatch.setattr(environment, "uuid4", lambda: SimpleNamespace(hex="collision"))
    stage = tmp_path / ".sprout-write-collision"
    stage.write_text("not ours")
    with pytest.raises(OSError, match="create stage"):
        asyncio.run(port.write_file(str(tmp_path / "target"), "new"))
    assert stage.read_text() == "not ours"
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize(
    "path,content",
    [
        ("relative", "value"),
        ("/", "value"),
        ("/target", "x" * (4 * 1024 * 1024 + 1)),
    ],
)
def test_invalid_write_has_no_sandbox_effect(tmp_path, port, path, content):
    with pytest.raises(ValueError):
        asyncio.run(port.write_file(path, content))
    assert not port.call_ids


def test_native_small_str_replace_in_large_existing_file(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "large.py"
    original = "# padding\n" * 30000 + "old_unique_value = 1\n"
    target.write_text(original)
    result, _, journal = run(
        root,
        tmp_path / "native",
        Model(
            [
                response(
                    call(
                        "edit",
                        "file_write",
                        path="large.py",
                        mode="str_replace",
                        old_str="old_unique_value = 1",
                        new_str="new_unique_value = 2",
                    )
                ),
                response(content="Done."),
            ]
        ),
    )
    assert result.status == "completed", result.error
    assert target.read_text() == original.replace("old_unique_value = 1", "new_unique_value = 2")
    calls = [row for row in read_journal(journal) if row["type"] == "tool.started"]
    assert len(calls) > 5
    assert max(len(row["args"]["command"].encode()) for row in calls) < 65536


def test_same_runtime_binary_accepts_large_write(tmp_path, port):
    """No Podman/global cleanup fixtures: only a private local runtime process."""
    import httpx

    from sprout.worker.tools.result import CommandOutcome, ToolResult
    from tests.support.paths import RUNTIME_BIN

    binary = RUNTIME_BIN
    if not binary.is_file():
        pytest.skip("prebuilt runtime unavailable")
    # Unix socket path length is limited, independent of shell argument size.
    import tempfile

    with tempfile.TemporaryDirectory(prefix="sprout-write-") as temporary:
        socket = Path(temporary) / "r.sock"
        process = subprocess.Popen(
            [str(binary), "--listen", f"unix:{socket}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        try:
            for _ in range(100):
                if socket.exists():
                    break
                assert process.poll() is None
                time.sleep(0.02)
            with httpx.Client(
                transport=httpx.HTTPTransport(uds=str(socket)), base_url="http://runtime", timeout=10
            ) as client:

                def call_runtime(name, args, timeout):
                    import json

                    reply = client.post(
                        "/",
                        json={
                            "jsonrpc": "2.0",
                            "id": 1,
                            "method": "tools/call",
                            "params": {"name": name, "arguments": args},
                        },
                    )
                    reply.raise_for_status()
                    result = reply.json()["result"]
                    output = result["content"][0]["text"]
                    if result.get("isError"):
                        value = json.loads(output)
                        outcome = CommandOutcome(
                            exit_code=value["exit_code"], stdout=value["stdout"], stderr=value["stderr"]
                        )
                    else:
                        outcome = CommandOutcome(exit_code=0, stdout=output)
                    return ToolResult(not result.get("isError"), output, outcome=outcome)

                port.tools.session.call = call_runtime
                target = tmp_path / "runtime-large.txt"
                content = "runtime 中文\n" * 30000
                asyncio.run(port.write_file(str(target), content))
                assert target.read_bytes() == content.encode()
        finally:
            process.terminate()
            process.wait(timeout=8)
