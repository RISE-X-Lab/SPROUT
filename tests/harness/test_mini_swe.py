"""Real upstream mini loop over a real HTTP model boundary and the run's ToolAccess.

The model replies are controlled and the VM filesystem is a temporary directory.
Shell commands really execute; snapshot contents are copied at actual boundaries.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import threading
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest

from sprout.harness.core.journal import read_journal
from sprout.harness.core.result import CommandOutcome, ToolResult
from sprout.harness.orchestrator.run import Orchestrator, OwnedSandbox, RunSpec
from sprout.harness_checkpoint.rollback import branch_checkpoints
from sprout.harness.agents.mini_swe.history import read_entries, training_messages
from sprout.runstore.messages import clean_messages

pytestmark = pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("minisweagent") is None,
    reason='install the mini extra: pip install -e ".[mini]"')


class FilesystemSession:
    sandbox_id = "fixture"

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshots = {}
        self.destroyed = False

    def supports_snapshot(self):
        return True

    def execute(self, name, args, timeout=30):
        assert name == "shell"
        directory = args.get("working_dir", "/testbed")
        if directory != "/testbed":
            return ToolResult(False, "", f"failed to start shell: chdir {directory}: no such file or directory")
        result = subprocess.run(
            ["sh", "-c", args["command"]], cwd=self.root,
            text=True, capture_output=True, timeout=timeout,
        )
        output = (result.stdout + result.stderr).replace(str(self.root), "/testbed")
        return ToolResult(result.returncode == 0, output)

    def snapshot(self, **kwargs):
        sid = "snapshot-" + uuid4().hex
        self.snapshots[sid] = {str(p.relative_to(self.root)): p.read_bytes()
                               for p in self.root.rglob("*") if p.is_file()}
        return SimpleNamespace(id=sid, rootfs_layers=None, memory_layers=None)

    def call(self, name, args, timeout=300):
        assert name == "shell"
        assert args["working_dir"] == "/testbed"
        result = subprocess.run(["bash", "-c", args["command"]], cwd=self.root,
                                text=True, capture_output=True, timeout=args["timeout"])
        if result.returncode == 0 and not result.stderr:
            # The real runtime reports this case as bare stdout with no
            # outcome; always returning one hid an adapter defect.
            return ToolResult(True, result.stdout)
        return ToolResult(result.returncode == 0, result.stdout, outcome=CommandOutcome(
            exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr))

    def destroy(self):
        self.destroyed = True


def owned_filesystem(session):
    return OwnedSandbox(session=session, sandbox_id="fixture")


@contextmanager
def model_server(replies):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests = []
    iterator = iter(replies)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            assert self.path == "/v1/chat/completions"
            reply = next(iterator)
            reply = reply(body) if callable(reply) else reply
            status, reply = reply if isinstance(reply, tuple) else (200, reply)
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def reply(*commands, content="Thinking.", reasoning="Inspect the actual file.", ids=None):
    return {"id": "chat-" + uuid4().hex, "object": "chat.completion", "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "tool_calls",
                         "message": {"role": "assistant", "content": content,
                                     "reasoning_content": reasoning,
                                     "tool_calls": [
                                         {"id": ids[i] if ids else uuid4().hex, "type": "function",
                                          "function": {"name": "bash", "arguments": json.dumps({"command": command})}}
                                         for i, command in enumerate(commands)]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 3},
                      "completion_tokens_details": {"reasoning_tokens": 5}}}


def spec(tmp_path, url, **changes):
    values = {
        "prompt": "Write the answer.", "slot": "mini-swe-agent", "model": "fixture",
        "timeout_s": 30,
        "journal_path": tmp_path / "trajectory.jsonl",
        "extra": {"native_home": str(tmp_path / "native-home"), "rollout_contract": {
            "message_export": True, "model_endpoint": url, "model": "fixture",
            "deadline_at": time.time() + 30, "max_turns": 5,
            "sampling_params": {"temperature": .8, "top_k": 20, "max_new_tokens": 300},
        }},
    }
    values.update(changes)
    return RunSpec(**values)


def test_upstream_loop_executes_then_captures_and_records_submission(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("printf first > answer", "cat answer"),
                       reply("printf second > answer"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    assert memory.destroyed and (memory.root / "answer").read_text() == "second"
    # One snapshot per closed turn, taken when the next turn starts; the last
    # turn's is the rollout's final disk snapshot.
    captures = [r for r in read_journal(outcome.journal_path) if r["type"] == "checkpoint.captured"]
    assert [(r["step"], r.get("final", False)) for r in captures] == [(2, False), (3, False), (4, True)]
    assert [r["disk_only"] for r in captures] == [True, True, True]
    points = branch_checkpoints(outcome.journal_path)
    assert set(points) == {2, 3, 4} and points[4].final
    assert outcome.final_snapshot_id == points[4].snapshot_id
    assert memory.snapshots[points[2].snapshot_id]["answer"] == b"first"
    assert memory.snapshots[points[3].snapshot_id]["answer"] == b"second"
    entries = read_entries((tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl").read_bytes())
    messages = clean_messages(training_messages(entries))
    assert len([m for m in messages if m["role"] == "tool"]) == 4
    assert "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in messages[-1]["content"]
    assert all(m["reasoning_content"] == "Inspect the actual file."
               for m in messages if m["role"] == "assistant")
    assert all(request["tools"][0]["function"]["name"] == "bash" for request in requests)
    system = requests[0]["messages"][0]
    assert "/testbed" in system["content"]
    assert all(request["messages"][0] == system for request in requests)
    workspace = next(entry["workspace"] for entry in entries if entry["type"] == "mini.session")
    assert workspace == {"cwd": "/testbed", "repository_dir": None}
    assert all(request["top_k"] == 20 and request["max_tokens"] == 300 for request in requests)
    assert outcome.usage["input_tokens"] == 33 and outcome.usage["reasoning_output_tokens"] == 15


#: sglang's refusal of a prompt plus completion budget past the context window.
CONTEXT_REFUSAL = (400, {"object": "error", "type": "BadRequestError", "code": 400, "message":
                         "Requested token count exceeds the model's maximum context length of "
                         "262144 tokens. You requested a total of 262203 tokens."})


# Two ways a rollout runs out of model: the turn budget refuses the next request
# at the gateway; the model refuses a conversation past its context window.
@pytest.mark.parametrize("max_turns,replies,stop_reason,failed_requests", [
    pytest.param(1, [reply("echo fix > answer")], "max_turns_reached", [], id="turn-budget"),
    pytest.param(5, [reply("echo fix > answer"), CONTEXT_REFUSAL], "context_exhausted",
                 [(400, "context_length_exceeded")], id="context-window"),
])
def test_budget_or_context_exhaustion_ends_the_rollout_as_gradable(
        tmp_path, monkeypatch, max_turns, replies, stop_reason, failed_requests):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server(replies) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["rollout_contract"]["max_turns"] = max_turns
        outcome = Orchestrator(out_dir=tmp_path).run(run_spec)
    # One request per reply: no extra request past the budget, a refusal never retried.
    assert outcome.stop_reason == stop_reason and len(requests) == len(replies)
    assert outcome.status == "timeout", "A budget- or context-limited workspace must remain gradable"
    assert (memory.root / "answer").read_text().strip() == "fix"
    entries = read_entries((tmp_path / "native-home" / f"{outcome.native_session_id}.jsonl").read_bytes())
    messages = clean_messages(training_messages(entries))
    assert messages[-1]["role"] == "tool"
    # The one closed turn's branch point is the final snapshot.
    points = branch_checkpoints(outcome.journal_path)
    assert list(points) == [1] and outcome.final_snapshot_id == points[1].snapshot_id
    failed = [r for r in read_journal(outcome.journal_path) if r["type"] == "model.request.failed"]
    assert [(r["status_code"], r["error_type"]) for r in failed] == failed_requests


def test_any_other_bad_request_is_still_an_error(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    refusal = (400, {"error": {"message": "budget exhausted for this run", "type": "invalid_request_error"}})
    with model_server([reply("echo fix > answer"), refusal]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "error" and outcome.stop_reason is None


def test_invalid_configured_directory_fails_before_model_or_agent_tool(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["mini"] = {"environment": {"cwd": "/missing-workspace"}}
        outcome = Orchestrator().run(run_spec)
    assert "chdir /missing-workspace" in outcome.error
    assert requests == []
    assert not any(event["type"] == "tool.started" for event in read_journal(outcome.journal_path))


def test_without_turn_checkpoints_the_final_snapshot_is_still_taken(tmp_path, monkeypatch):
    memory = FilesystemSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo fixed > answer"), reply("echo more >> answer")]) as (url, requests):
        run_spec = spec(tmp_path, url)
        run_spec.extra["rollout_contract"]["max_turns"] = 2
        run_spec.checkpoint_enabled = False
        outcome = Orchestrator(out_dir=tmp_path).run(run_spec)
    assert outcome.status == "timeout", outcome.error
    captures = [r for r in read_journal(outcome.journal_path) if r["type"] == "checkpoint.captured"]
    assert len(captures) == 1 and captures[0]["final"] and captures[0]["disk_only"]
    assert memory.snapshots[outcome.final_snapshot_id]["answer"] == b"fixed\nmore\n"
    assert memory.destroyed


class UncertainSession(FilesystemSession):
    """A sandbox whose shell call never settles in the way ``kind`` names."""

    def __init__(self, root, kind, running=False, exit_code=None):
        super().__init__(root)
        self.kind, self.running, self.exit_code = kind, running, exit_code
        self.calls = 0

    def call(self, name, args, timeout=300):
        self.calls += 1
        if self.kind == "transport_exception":
            raise TimeoutError("synthetic uncertain transport")
        return ToolResult(False, "not settled", outcome=CommandOutcome(
            timed_out=self.kind == "runtime_timeout", running=self.running, exit_code=self.exit_code))


@pytest.mark.parametrize("failure_kind,running,exit_code", [
    ("transport_exception", False, None),
    ("runtime_timeout", False, None),
    ("runtime_timeout", True, 137),
    ("runtime_still_running", True, None),
])
def test_uncertain_execution_stops_mini_before_another_model_request(
        tmp_path, monkeypatch, failure_kind, running, exit_code):
    memory = UncertainSession(tmp_path / "sandbox", failure_kind, running, exit_code)
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply("echo unsettled"), reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "error" and "execution uncertain" in outcome.error
    # The run stops at the uncertain call: no second model request, no second call.
    assert len(requests) == 1 and memory.calls == 1
    events = read_journal(outcome.journal_path)
    assert len([e for e in events if e.get("type") == "tool.started"]) == 1
    failure = next(e for e in events if e.get("type") == "execution.uncertain")
    assert failure["kind"] == failure_kind and failure["step"] == 1
    # The final snapshot is still taken (the VM is never destroyed without it),
    # but a state reached through an uncertain call is never a branch point.
    final = next(e for e in events if e.get("type") == "checkpoint.captured")
    assert final["final"] and final["uncertain"]
    assert branch_checkpoints(outcome.journal_path) == {}


class SettledTimeoutSession(FilesystemSession):
    def call(self, name, args, timeout=300):
        if args["command"].endswith("partial-write"):
            (self.root / "answer").write_text("partial")
            return ToolResult(False, "", outcome=CommandOutcome(
                exit_code=137, stdout="partial output", stderr="partial stderr", timed_out=True))
        return super().call(name, args, timeout)


# A command that settled without succeeding -- a real exit code, a timeout the
# runtime reported -- is an observation, not an infrastructure failure: the
# model sees it in its next request and the turn is still a branch point.
@pytest.mark.parametrize("command,returncode,output,exception_info,files", [
    pytest.param("printf diagnostic; exit 42", 42, ["diagnostic"], None, {}, id="exit-code"),
    pytest.param("partial-write", 137, ["partial output", "partial stderr"], "Command timed out",
                 {"answer": b"partial"}, id="settled-timeout"),
])
def test_settled_failure_reaches_the_next_query_and_leaves_exact_gradable_state(
        tmp_path, monkeypatch, command, returncode, output, exception_info, files):
    memory = SettledTimeoutSession(tmp_path / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    with model_server([reply(command),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        outcome = Orchestrator(out_dir=tmp_path).run(spec(tmp_path, url))
    assert outcome.status == "completed", outcome.error
    observed = json.loads(requests[1]["messages"][-1]["content"])
    assert observed["returncode"] == returncode and observed.get("exception_info") == exception_info
    assert all(fragment in observed["output"] for fragment in output)
    points = branch_checkpoints(outcome.journal_path)
    assert memory.snapshots[points[1].snapshot_id] == files
