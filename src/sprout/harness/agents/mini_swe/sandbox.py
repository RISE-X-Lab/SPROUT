"""Resolve the owned sandbox workspace before any model-generated action."""

from copy import deepcopy

from sprout.harness.agents.mini_swe.history import load_prefix


def _stdout(result, purpose):
    outcome = getattr(result, "outcome", None)
    if not result.success or (outcome is not None and (
        outcome.exit_code != 0 or outcome.running or outcome.timed_out or outcome.truncated
    )):
        detail = result.error or (getattr(outcome, "stderr", "") if outcome else "") or result.output
        raise ValueError(f"mini {purpose} failed: {detail or 'shell returned no diagnostic'}")
    text = outcome.stdout if outcome is not None else result.output
    value = text.strip()
    if not value.startswith("/") or "\n" in value or "\r" in value:
        raise ValueError(f"mini {purpose} returned an invalid absolute path: {value!r}")
    return value


def resolve_workspace(session, extra, journal):
    configured = extra.get("mini", {}).get("environment", {}).get("cwd")
    inherited = None
    if extra.get("native_prefix"):
        entries = load_prefix(extra["native_prefix"])
        metadata = next(entry for entry in entries if entry.get("type") == "mini.session")
        inherited = metadata.get("workspace")
    if inherited and configured not in (None, inherited["cwd"]):
        raise ValueError("mini branch working directory differs from its inherited workspace")
    requested = inherited["cwd"] if inherited else configured
    if requested is not None and (not isinstance(requested, str) or not requested.startswith("/")):
        raise ValueError("mini working directory must be an absolute path or omitted for discovery")
    arguments = {"command": "pwd -P", "timeout": 10}
    if requested is not None:
        arguments["working_dir"] = requested
    cwd = _stdout(session.execute("shell", arguments, timeout=20), "working-directory probe")
    repository = session.execute("shell", {
        "command": "git rev-parse --show-toplevel", "working_dir": cwd, "timeout": 10,
    }, timeout=20)
    repository_dir = _stdout(repository, "repository probe") if repository.success else None
    if requested is None and repository_dir is not None and cwd != repository_dir:
        cwd = _stdout(session.execute("shell", {
            "command": "pwd -P", "working_dir": repository_dir, "timeout": 10,
        }, timeout=20), "repository working-directory probe")
    workspace = {"cwd": cwd, "repository_dir": repository_dir}
    if inherited is not None and workspace != inherited:
        raise ValueError("mini restored workspace differs from the parent's recorded workspace")
    resolved = deepcopy(extra)
    resolved.setdefault("mini", {}).setdefault("environment", {})["cwd"] = cwd
    resolved["mini_workspace"] = workspace
    journal.emit("mini.workspace", **workspace, inherited=inherited is not None)
    return resolved
