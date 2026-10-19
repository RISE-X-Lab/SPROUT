"""The ``grade`` job, and the SWE-rebench-V2 grader that works as the official harness does.

``grade`` hands a ``harbor`` grade to ``sprout.worker.jobs.harbor.grade``. A
``swe-rebench-v2`` grade does what the upstream ``scripts/eval.py`` does, in two
sandboxes. The agent's final snapshot only gives its diff against the base commit.
A pristine sandbox from the task image gets that diff and the hidden ``test_patch``,
runs ``install_config.test_cmd``, and parses the log with the parser that
``install_config.log_parser`` names. ``parser_path`` and ``grader_revision`` pin the
parser file by checksum. The task is resolved when every FAIL_TO_PASS and
PASS_TO_PASS test passed. Nothing that the agent changed outside its diff can reach
the verdict.

An unresolved task is a completed grade, not a retry. A failure that the agent's work
can cause is a completed grade of 0 with a ``reason``: ``model_patch_unavailable``,
``model_patch_apply_failed``, ``test_patch_apply_failed``, ``tests_timed_out`` or
``test_runner_killed``. A grade that cannot be made raises, and the job runs again while
attempts remain: a sandbox that does not start, a file that does not move, or a dataset
or parser whose checksum differs. A worker claims only the grades whose checksums its
files had at its start. The result fields and files: ``docs/RUNSTORE.md``, "Official grading".
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shlex
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sprout.runstore.specs import GradeSpec
from sprout.worker.jobs.harbor.grade import grade_harbor
from sprout.worker.sandbox.backends import with_egress

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession


def dataset_sha256(data: bytes) -> str:
    """The ``dataset_sha256`` of a dataset file whose content is ``data``: its SHA-256 in hex."""
    return hashlib.sha256(data).hexdigest()


def parser_revision(data: bytes) -> str:
    """The ``grader_revision`` of a parser file whose content is ``data``: ``sha256:<hex>``."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def dataset_rows(spec: GradeSpec) -> list[dict[str, Any]]:
    """The task rows of the grade's frozen dataset, a JSON list or a JSONL file.

    Raises ValueError if the file's SHA256 differs from ``dataset_sha256``, or if a
    JSON file does not hold a list.
    """
    if spec.dataset_path is None:
        raise ValueError("Grading requires the frozen dataset_path")
    data = Path(spec.dataset_path).read_bytes()
    if dataset_sha256(data) != spec.dataset_sha256:
        raise ValueError("Frozen dataset checksum changed")
    if spec.dataset_path.endswith(".jsonl"):
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("Frozen dataset must contain a list of tasks")
    return rows


def grade(
    body: dict[str, Any],
    directory: Path,
    *,
    session_factory: Callable[..., SandboxSession],
    kept_sandbox: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The result of the grade job ``body``, the fields of a ``GradeSpec``.

    A ``harbor`` grade with an ``in_place_job`` runs in ``kept_sandbox``, None when it is gone.
    Raises ValueError for an unsupported benchmark, or a dataset without one row of ``instance_id``.
    """
    spec = GradeSpec(**body)
    if spec.benchmark == "harbor":
        return grade_harbor(spec, directory, session_factory=session_factory, kept_sandbox=kept_sandbox)
    rows = dataset_rows(spec)
    selected = [row for row in rows if row.get("instance_id") == spec.instance_id]
    if len(selected) != 1:
        raise ValueError("Frozen dataset does not identify exactly one grading task")
    # Egress is a setting of the backend that runs the grading sandboxes. podman (the
    # default) disconnects them before they start, and AgentENV creates them offline.
    backend = with_egress(spec.backend, allow=spec.verifier_network == "allow")
    if spec.benchmark == "swe-rebench-v2":
        return grade_swerebench(spec, selected[0], backend, directory, session_factory=session_factory)
    raise ValueError(f"Unsupported official grader {spec.benchmark!r}")


#: The PATH of the test script: the toolchain directories of the SWE-rebench images.
PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/opt/conda/envs/testbed/bin:/opt/conda/bin:/usr/local/cargo/bin:"
    "/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def load_parsers(spec: GradeSpec) -> ModuleType:
    """The parser module at ``parser_path``. Raises ValueError if its checksum is not ``grader_revision``."""
    path = Path(spec.parser_path or "")
    if not path.is_file() or parser_revision(path.read_bytes()) != spec.grader_revision:
        raise ValueError("SWE-rebench parser revision differs from grader_revision")
    module_spec = importlib.util.spec_from_file_location("sprout_swerebench_parsers", path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"SWE-rebench parser {path} is not a Python module")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _normalize(name: str) -> str:
    """``name`` without the duration that a test runner appends, such as `` [12 ms]``."""
    for pattern in (
        r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$",
        r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b",
        r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$",
    ):
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name.strip()


def _passed(status_map: Mapping[str, str], names: Iterable[str]) -> list[bool]:
    actual = {_normalize(name): value for name, value in status_map.items()}
    return [actual.get(_normalize(name)) == "PASSED" for name in names]


def _expected(task: Mapping[str, Any]) -> list[str]:
    return list(task.get("FAIL_TO_PASS") or []) + list(task.get("PASS_TO_PASS") or [])


def resolved(status_map: Mapping[str, str], task: Mapping[str, Any]) -> bool:
    """True when every FAIL_TO_PASS and PASS_TO_PASS test of ``task`` passed. False without tests."""
    outcomes = _passed(status_map, _expected(task))
    return bool(outcomes) and all(outcomes)


def progress(status_map: Mapping[str, str], task: Mapping[str, Any]) -> float:
    """The share of the task's FAIL_TO_PASS tests that passed: 1.0 is resolved.

    PASS_TO_PASS tests pass before any change, so the share leaves them out. One that
    fails makes it 0.0. A task without FAIL_TO_PASS tests scores its verdict.
    """
    targets = _passed(status_map, task.get("FAIL_TO_PASS") or [])
    if not targets:  # nothing to fix: the verdict is all there is
        return float(resolved(status_map, task))
    if not all(_passed(status_map, task.get("PASS_TO_PASS") or [])):
        return 0.0
    return sum(targets) / len(targets)


def expected_pass_share(status_map: Mapping[str, str], task: Mapping[str, Any]) -> float:
    """The share of all expected tests of ``task`` that passed, PASS_TO_PASS included. 0.0 without tests."""
    outcomes = _passed(status_map, _expected(task))
    return sum(outcomes) / len(outcomes) if outcomes else 0.0


#: The facts about the task's hidden tests that each completed grade adds (:func:`patch_facts`).
#: No analyst or reviewer prompt may show them.
GRADE_FACT_KEYS = ("test_patch_paths", "touched_test_paths", "fail_to_pass")


def patch_facts(model_patch: str, row: dict[str, Any]) -> dict[str, list[str]]:
    """The :data:`GRADE_FACT_KEYS` of a grade of ``model_patch`` on the task ``row``.

    They are the paths of the test patch, those of them that ``model_patch`` names too,
    and the FAIL_TO_PASS tests.
    """
    test_patch_paths = patch_paths(row["test_patch"])
    return {
        "test_patch_paths": test_patch_paths,
        "touched_test_paths": sorted(set(patch_paths(model_patch)) & set(test_patch_paths)),
        "fail_to_pass": list(row.get("FAIL_TO_PASS") or []),
    }


def patch_paths(text: str) -> list[str]:
    """Every path that a patch names, old and new, sorted and unique, without /dev/null.

    The patch is read as ``git apply`` reads it, with quoted names, renames, copies, mode
    changes and binary files. A text without a ``diff --git`` line is a plain unified diff.
    """
    # Split on "\n" only: ``splitlines`` also splits on characters that a hunk line can hold.
    lines = text.split("\n")
    names: list[str | None] = []
    if not any(line.startswith("diff --git ") for line in lines):
        # A plain unified diff names its files in each ``---``, ``+++``, ``@@`` triple.
        for index, line in enumerate(lines[:-2]):
            if (
                line.startswith("--- ")
                and lines[index + 1].startswith("+++ ")
                and lines[index + 2].startswith("@@")
            ):
                names += [_side(line), _side(lines[index + 1])]
        return sorted({name for name in names if name})
    # A ``diff --git`` line at column 0 opens a file's header, which ends at the first hunk or
    # binary payload. Inside a hunk, a ``---`` or ``+++`` line is a removed or added line.
    in_header = False
    for line in lines:
        if line.startswith("diff --git "):
            in_header = True
            # The header line also names a file without ---/+++ lines, such as a mode change or a binary file.
            names += _header_paths(line[len("diff --git ") :])
        elif not in_header:
            continue
        elif line.startswith(("@@", "GIT binary patch", "Binary files ")):
            in_header = False
        elif line.startswith(("--- ", "+++ ")):
            names.append(_side(line))
        else:
            for marker in _NAME_LINES:
                if line.startswith(marker):  # no a/ or b/ on these
                    names.append(_quoted(line[len(marker) :]))
    return sorted({name for name in names if name})


#: Extended header lines that name a file without a prefix.
_NAME_LINES = ("rename from ", "rename to ", "copy from ", "copy to ")
#: The C escapes git writes in a quoted name, beside ``\ooo`` for a byte.
_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
_OCTAL = re.compile(r"[0-3][0-7]{2}")


def _unquote(text: str, start: int = 0) -> tuple[str | None, int]:
    """The C-quoted name that opens at ``text[start]``, and the index after its closing quote.

    The escapes are bytes, and git escapes each non-ASCII byte (``core.quotepath``). So the
    name is decoded as UTF-8 after all escapes are in place. None for a quote that never closes.
    """
    raw = bytearray()
    index = start + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            return raw.decode("utf-8", errors="replace"), index + 1
        if char == "\\":
            octal = _OCTAL.match(text, index + 1)
            if octal:
                raw.append(int(octal.group(), 8))
                index = octal.end()
                continue
            if text[index + 1 : index + 2] in _C_ESCAPES:
                raw.append(_C_ESCAPES[text[index + 1]])
                index += 2
                continue
        raw += char.encode("utf-8", errors="replace")
        index += 1
    return None, index


def _quoted(name: str) -> str | None:
    """``name`` as written in a patch, C-quoted or not."""
    return _unquote(name)[0] if name.startswith('"') else name


def _unprefixed(name: str | None) -> str | None:
    """A patch's ``a/X`` or ``b/X`` as ``X``. None for /dev/null."""
    if name is None or name == "/dev/null":
        return None
    return name[2:] if name[:2] in ("a/", "b/") else name


def _side(line: str) -> str | None:
    """The path of a ``--- `` or ``+++ `` line, without what follows a TAB.

    git writes a TAB after a name that holds a space, and diff writes one before a date.
    """
    return _unprefixed(_quoted(line[4:].split("\t", 1)[0]))


def _header_paths(names: str) -> list[str | None]:
    """The paths of a ``diff --git`` header when its names can be told apart, else an empty list."""
    if names.startswith('"'):
        first, end = _unquote(names)
        if first is None or names[end : end + 1] != " ":
            return []
        pair = [first, _quoted(names[end + 1 :])]
    elif names.endswith('"') and ' "' in names:  # an unquoted name holds no quote
        split = names.index(' "')
        pair = [names[:split], _unquote(names, split + 1)[0]]
    else:
        # Unquoted names are told apart only when they are symmetric (``a/X b/X``),
        # because a name can hold a space followed by ``b/``.
        middle = (len(names) - 1) // 2
        old, new = names[:middle], names[middle + 1 :]
        if (
            names[middle : middle + 1] != " "
            or old[:2] not in ("a/", "b/")
            or new[:2] not in ("a/", "b/")
            or old[2:] != new[2:]
        ):
            return []  # asymmetric: a rename or copy, whose own lines name both
        pair = [old, new]
    return [_unprefixed(name) for name in pair]


#: Each git command of the grader trusts the repository, whoever owns it. Else git refuses a repository
#: of another uid, and the refusal would score the agent 0 (``model_patch_unavailable``), not fail the grade.
GIT = "git -c safe.directory='*'"

#: Applies the patch file ``%s`` as the upstream ``scripts/eval.py`` does. A patch that git cannot apply,
#: a merge conflict included, is not applied: the upstream harness has no other way to apply it.
APPLY = GIT + " apply -v --3way --recount --ignore-space-change --whitespace=nowarn %s"


#: Writes the agent's diff against the base commit to a file, so that the exit status is git's own. The
#: options override what the agent's git config can set: external diff, textconv, colour, fsmonitor, prefixes.
DIFF = (
    GIT + " -c core.fsmonitor=false add -A && " + GIT + " -c core.quotepath=off -c core.fsmonitor=false diff"
    " --cached --binary --no-ext-diff --no-textconv --no-color --src-prefix=a/ --dst-prefix=b/ %s > %s"
)


def guest_directory() -> str:
    """A new path under ``/tmp`` for the grade's files in one grading sandbox."""
    return "/tmp/sprout-grade-" + uuid4().hex


def _guest_directory(session: SandboxSession) -> str:
    from sprout.worker.sandbox.commands import checked

    guest = guest_directory()
    # ``mkdir -m 700`` refuses a path that exists. A file or symlink that the agent left at a
    # fixed name in its snapshot then cannot stand in for a file that the grader writes or reads.
    checked(session, "mkdir -m 700 " + shlex.quote(guest), 120)
    return guest


def _create(session: SandboxSession, image: str, resources: dict[str, Any], failure: str) -> None:
    if not session.create(image, resources):
        reason = session.create_error
        raise RuntimeError(failure + (": " + reason if reason else ""))


def _upload(session: SandboxSession, source: Path, destination: str) -> None:
    """Copy ``source`` into the sandbox through the backend's file channel.

    A file never travels inside a command. The runtime runs a command as one ``sh -c``
    argument, which Linux caps at 128 KiB (``MAX_ARG_STRLEN``), and reports a longer one as exit 1.
    """
    if not session.upload_file(source, destination):
        raise RuntimeError(
            f"Could not upload {source.name} to {destination} in the grading sandbox"
            + _transfer_error(session)
        )


def _download(session: SandboxSession, source: str, destination: Path) -> None:
    if not session.download_file(source, destination):
        raise RuntimeError(f"Could not download {source} from the grading sandbox" + _transfer_error(session))


def _transfer_error(session: SandboxSession) -> str:
    reason = session.transfer_error
    return ": " + reason if reason else ""


def _apply(path: str) -> str:
    return APPLY % shlex.quote(path)


def _zero(reason: str, error_output: str, common: dict[str, Any], **extra: str) -> dict[str, Any]:
    """A completed grade of 0 that no test verdict decided."""
    return {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": reason,
        "error_output": error_output,
        **extra,
        **common,
    }


def model_patch(
    spec: GradeSpec,
    row: dict[str, Any],
    backend: dict[str, Any],
    destination: Path,
    *,
    session_factory: Callable[..., SandboxSession],
) -> tuple[bytes | None, str]:
    """The agent's diff against the base commit, from its final snapshot, saved at ``destination``.

    ``(None, git's error)`` when the snapshot's repository cannot give one, for example
    after the agent removed ``.git`` or the base commit. The grade then scores 0.
    """
    from sprout.worker.sandbox.commands import shell

    if spec.snapshot_id is None:
        raise ValueError("SWE-rebench grading needs the snapshot_id of the agent's final snapshot")
    workdir = "/" + row["repo"].split("/", 1)[1]
    session = session_factory(quiet=True, backend=backend)
    try:
        _create(session, spec.snapshot_id, spec.resources, "Could not restore the SWE-rebench final snapshot")
        guest_patch = _guest_directory(session) + "/model.patch"
        _, stderr, code = shell(
            session,
            f"cd {shlex.quote(workdir)} && "
            + DIFF % (shlex.quote(row["base_commit"]), shlex.quote(guest_patch)),
            int(spec.timeout_s),
        )
        if code:
            return None, stderr[-4000:] or f"exit code {code}"
        _download(session, guest_patch, destination)
        return destination.read_bytes(), ""
    finally:
        session.destroy()


def grade_swerebench(
    spec: GradeSpec,
    row: dict[str, Any],
    backend: dict[str, Any],
    directory: Path,
    *,
    session_factory: Callable[..., SandboxSession],
) -> dict[str, Any]:
    """The grade of the agent's final snapshot on the SWE-rebench task ``row`` (module docstring).

    ``directory`` keeps ``model.patch``, ``test.patch``, ``eval.sh`` and ``swerebench-tests.log``.
    """
    parsers = load_parsers(spec)
    config = row["install_config"]
    if isinstance(config, str):
        config = json.loads(config)
    parser = parsers.NAME_TO_PARSER.get(config["log_parser"]) or getattr(parsers, config["log_parser"], None)
    if parser is None:
        raise ValueError("Unknown SWE-rebench log parser")
    script = _eval_script(config["test_cmd"])
    if not row["test_patch"].strip() or not row["base_commit"]:
        raise ValueError("SWE-rebench needs test_patch and base_commit")
    if not spec.image:
        raise ValueError("SWE-rebench grading needs the task image")
    workdir = "/" + row["repo"].split("/", 1)[1]
    patch_file, test_file, script_file = (
        directory / "model.patch",
        directory / "test.patch",
        directory / "eval.sh",
    )
    log = directory / "swerebench-tests.log"
    test_file.write_bytes(row["test_patch"].encode())
    script_file.write_bytes(script.encode())
    patch, error_output = model_patch(spec, row, backend, patch_file, session_factory=session_factory)
    if patch is None:
        patch_file.write_bytes(b"")  # every result names this path, also when nothing was graded
    # ``test_patch`` is the path of the hidden tests as applied. Only the review's leak judge
    # (``sprout.worker.jobs.review.leak.judge``) reads it. No analyst prompt shows it.
    common = {
        "model_patch": str(patch_file),
        "test_patch": str(test_file),
        "grader_revision": spec.grader_revision,
        **patch_facts((patch or b"").decode("utf-8", errors="replace"), row),
    }
    if patch is None:
        return _zero("model_patch_unavailable", error_output, common)
    session = session_factory(quiet=True, backend=backend)
    try:
        return _grade_in_pristine(
            session,
            image=spec.image,
            resources=spec.resources,
            timeout_s=spec.timeout_s,
            workdir=workdir,
            patch=patch,
            patch_file=patch_file,
            test_file=test_file,
            script_file=script_file,
            log=log,
            parser=parser,
            row=row,
            common=common,
        )
    finally:
        session.destroy()


def _eval_script(test_cmd: Any) -> str:
    """The ``eval.sh`` that runs each command of ``test_cmd``, and fails when one of them fails.

    Raises ValueError unless ``test_cmd`` is a command or a list of commands, none of them blank."""
    commands = test_cmd
    if isinstance(commands, str):
        commands = [commands]
    if not commands or any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError("SWE-rebench test_cmd must contain commands")
    return (
        "set -uo pipefail\nexport PATH=" + shlex.quote(PATH) + "\n"
        "export CI=1 PAGER=cat MANPAGER=cat PIP_PROGRESS_BAR=off TQDM_DISABLE=1\n"
        "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\nFAIL=0\n"
        + "\n".join(f"{command} || FAIL=1" for command in commands)
        + '\nexit "$FAIL"\n'
    )


def _grade_in_pristine(
    session: SandboxSession,
    *,
    image: str,
    resources: dict[str, Any],
    timeout_s: float,
    workdir: str,
    patch: bytes,
    patch_file: Path,
    test_file: Path,
    script_file: Path,
    log: Path,
    parser: Callable[[str], dict[str, str] | None],
    row: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, Any]:
    """The grade of ``patch`` with the hidden tests, in ``session`` started from the task ``image``.

    The caller destroys ``session``."""
    from sprout.worker.sandbox.commands import checked, shell, shell_outcome

    _create(session, image, resources, "Could not start a pristine SWE-rebench sandbox from the task image")
    prefix = f"cd {shlex.quote(workdir)} && "
    checked(session, prefix + GIT + " reset -q --hard HEAD", 120)
    guest = _guest_directory(session)
    if patch.strip():
        _upload(session, patch_file, guest + "/model.patch")
        _, stderr, code = shell(session, prefix + _apply(guest + "/model.patch"), 300)
        if code:
            return _zero("model_patch_apply_failed", stderr[-4000:], common)
    _upload(session, test_file, guest + "/test.patch")
    _, stderr, code = shell(session, prefix + _apply(guest + "/test.patch"), 300)
    if code:
        return _zero("test_patch_apply_failed", stderr[-4000:], common)
    _upload(session, script_file, guest + "/eval.sh")
    guest_log = guest + "/tests.log"
    _, stderr, test_code, timed_out = shell_outcome(
        session,
        prefix + f"bash {shlex.quote(guest + '/eval.sh')} > {shlex.quote(guest_log)} 2>&1",
        int(timeout_s),
    )
    if timed_out:
        # The log so far shows a reviewer where the tests hung. The verdict does not need
        # it, so a failed download loses only the log.
        saved = session.download_file(guest_log, log)
        return _zero(
            "tests_timed_out",
            f"the tests did not finish within {int(timeout_s)} s",
            common,
            **({"log": str(log)} if saved else {}),
        )
    _download(session, guest_log, log)
    if test_code not in (0, 1):
        # The tests' own output is in the log. stderr holds only the outer shell's output,
        # such as a failed ``cd``, and is usually empty.
        killed = f"exit code {test_code}" + (f"\n{stderr[-4000:]}" if stderr.strip() else "")
        return _zero("test_runner_killed", killed, common, log=str(log))
    status_map = parser(log.read_text(encoding="utf-8", errors="replace")) or {}
    return {
        "status": "completed",
        "resolved": resolved(status_map, row),
        "progress": progress(status_map, row),
        "expected_pass_share": expected_pass_share(status_map, row),
        "test_statuses": status_map,
        "log": str(log),
        **common,
    }
