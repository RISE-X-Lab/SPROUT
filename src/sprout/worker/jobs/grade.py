"""The queued SWE-rebench grader, the way the official harness grades.

SWE-rebench-V2's ``scripts/eval.py`` starts a fresh container from the task
image, applies the model's diff and the hidden ``test_patch`` with ``git apply
--3way``, runs ``install_config.test_cmd``, parses the output with the parser
``install_config.log_parser`` names in ``lib/agent/log_parsers.py`` (pinned here
by checksum as ``parser_path`` / ``grader_revision``), and resolves when every
FAIL_TO_PASS and PASS_TO_PASS test passed. This does the same in two sandboxes:
the agent's final snapshot only supplies the diff (``git diff`` against the
base commit, untracked files included); the tests run on a pristine image, so
nothing the agent changed outside its diff can reach the verdict.

Files cross into and out of a grading sandbox through the backend's file
channel (``upload_file`` / ``download_file``: ``podman cp``, AgentENV's
``/files``), in a private directory made for the grade, never inside a command.
The runtime runs each command as one ``sh -c`` argument, which Linux caps at
``MAX_ARG_STRLEN`` (128 KiB), and it reports the ``E2BIG`` of a longer one as
a bare exit 1: staging the patches as base64 in the command failed every
grade of a task whose diff or test patch passed about 96 KiB.

Unresolved is a completed result, not a retry, and so is a failure the
agent's work can cause, which scores 0: a diff its snapshot cannot produce
(``model_patch_unavailable``), a patch that does not apply, tests that run out
of time (``tests_timed_out``), a test runner that exits with neither 0 nor 1
(``test_runner_killed``, an OOM kill among them). A grade that could not be
made raises -- a sandbox that would not start, a file that would not move, a
parser or dataset whose checksum differs -- and the job fails.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shlex
from pathlib import Path
from uuid import uuid4

from sprout.contract.backend import backend_name
from sprout.runstore.specs import GradeSpec


def dataset_rows(spec: GradeSpec) -> list[dict]:
    data = Path(spec.dataset_path).read_bytes()
    if hashlib.sha256(data).hexdigest() != spec.dataset_sha256:
        raise ValueError("Frozen dataset checksum changed")
    if spec.dataset_path.endswith(".jsonl"):
        return [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = json.loads(data)
    if not isinstance(rows, list):
        raise ValueError("Frozen dataset must contain a list of tasks")
    return rows


def grade(body: dict, directory: Path, *, session_factory) -> dict:
    spec = GradeSpec(**body)
    rows = dataset_rows(spec)
    selected = [row for row in rows if row.get("instance_id") == spec.instance_id]
    if len(selected) != 1:
        raise ValueError("Frozen dataset does not identify exactly one grading task")
    # Egress is a setting of the backend that runs the grading sandboxes:
    # podman (the default) disconnects them before they start, AgentENV
    # creates them offline.
    name = backend_name(spec.backend)
    backend = {
        **spec.backend,
        name: {**spec.backend.get(name, {}), "allow_internet": spec.verifier_network == "allow"},
    }
    if spec.benchmark == "swe-rebench-v2":
        return grade_swerebench(spec, selected[0], backend, directory, session_factory=session_factory)
    raise ValueError(f"Unsupported official grader {spec.benchmark!r}")


PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/opt/conda/envs/testbed/bin:/opt/conda/bin:/usr/local/cargo/bin:"
    "/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)


def load_parsers(spec):
    path = Path(spec.parser_path or "")
    if (
        not path.is_file()
        or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != spec.grader_revision
    ):
        raise ValueError("SWE-rebench parser checksum differs from grader_revision")
    module_spec = importlib.util.spec_from_file_location("sprout_swerebench_parsers", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _normalize(name):
    for pattern in (
        r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$",
        r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b",
        r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$",
    ):
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    return name.strip()


def _passed(status_map, names):
    actual = {_normalize(name): value for name, value in status_map.items()}
    return [actual.get(_normalize(name)) == "PASSED" for name in names]


def _expected(task):
    return list(task.get("FAIL_TO_PASS") or []) + list(task.get("PASS_TO_PASS") or [])


def resolved(status_map, task):
    outcomes = _passed(status_map, _expected(task))
    return bool(outcomes) and all(outcomes)


def progress(status_map, task):
    """The share of the task's FAIL_TO_PASS tests that passed: 1.0 is resolved.

    Finer than the verdict, for weighting and comparing attempts: a
    continuation that gets more of the target tests through than another
    made progress even when neither resolved the task. PASS_TO_PASS tests
    pass before any change, so they would only raise every attempt's score
    alike (an empty patch scores 43/45 on a task with 2 targets among 45);
    breaking one of them is no progress at all, and scores 0.
    """
    targets = _passed(status_map, task.get("FAIL_TO_PASS") or [])
    if not targets:  # nothing to fix: the verdict is all there is
        return float(resolved(status_map, task))
    if not all(_passed(status_map, task.get("PASS_TO_PASS") or [])):
        return 0.0
    return sum(targets) / len(targets)


def expected_pass_share(status_map, task):
    """The share of all expected tests that passed, PASS_TO_PASS included: reported, not weighted on."""
    outcomes = _passed(status_map, _expected(task))
    return sum(outcomes) / len(outcomes) if outcomes else 0.0


#: What every completed grade says about the task's hidden tests beside its
#: verdict (:func:`patch_facts`): the files of the test patch, those of them
#: the model's diff names too, and the FAIL_TO_PASS tests. For the trainer and
#: the review's leak check; no analyst or reviewer prompt shows them.
GRADE_FACT_KEYS = ("test_patch_paths", "touched_test_paths", "fail_to_pass")


def patch_facts(model_patch: str, row: dict) -> dict:
    """The :data:`GRADE_FACT_KEYS` of a grade of ``model_patch`` on the task ``row``."""
    test_patch_paths = patch_paths(row["test_patch"])
    return {
        "test_patch_paths": test_patch_paths,
        "touched_test_paths": sorted(set(patch_paths(model_patch)) & set(test_patch_paths)),
        "fail_to_pass": list(row.get("FAIL_TO_PASS") or []),
    }


def patch_paths(text: str) -> list[str]:
    """Every path a patch names, old and new, sorted and unique, without /dev/null.

    Read the way ``git apply`` reads a patch. A ``diff --git`` line at column
    0 opens a file's section (no hunk line starts with it); its header block,
    up to the first hunk or binary payload, is the only place a ``---`` or
    ``+++`` line names a file, since inside a hunk the same text is a removed
    ``-- x`` or an added ``++ y``. The header line itself is read when its
    names are quoted or symmetric (``a/X b/X``, since a name may contain a
    space followed by ``b/``), so a mode change or a binary file is counted,
    and a rename by its ``rename from``/``rename to`` lines. Lines split on
    ``\\n`` alone: ``splitlines`` also splits on characters a hunk may hold.
    A text with no ``diff --git`` line is a plain unified diff, whose files
    are its ``---``, ``+++``, ``@@`` triples.
    """
    lines = text.split("\n")
    names: list[str | None] = []
    if not any(line.startswith("diff --git ") for line in lines):
        for index, line in enumerate(lines[:-2]):
            if (
                line.startswith("--- ")
                and lines[index + 1].startswith("+++ ")
                and lines[index + 2].startswith("@@")
            ):
                names += [_side(line), _side(lines[index + 1])]
        return sorted({name for name in names if name})
    in_header = False
    for line in lines:
        if line.startswith("diff --git "):
            in_header = True
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
    """The C-quoted name opening at ``text[start]``, and the index after its closing quote.

    The escapes are bytes, so a name git quoted for its non-ASCII bytes
    (``core.quotepath``, on by default) is decoded as UTF-8 once they are all
    in place. ``None`` for a quote that never closes.
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
    """A patch's ``a/X`` or ``b/X`` as ``X``; None for /dev/null."""
    if name is None or name == "/dev/null":
        return None
    return name[2:] if name[:2] in ("a/", "b/") else name


def _side(line: str) -> str | None:
    """The path of a ``--- `` or ``+++ `` line: git puts a TAB after a name holding a space, diff a date."""
    return _unprefixed(_quoted(line[4:].split("\t", 1)[0]))


def _header_paths(names: str) -> list[str | None]:
    """The paths of a ``diff --git`` header, when its names can be told apart."""
    if names.startswith('"'):
        first, end = _unquote(names)
        if first is None or names[end : end + 1] != " ":
            return []
        pair = [first, _quoted(names[end + 1 :])]
    elif names.endswith('"') and ' "' in names:  # an unquoted name holds no quote
        split = names.index(' "')
        pair = [names[:split], _unquote(names, split + 1)[0]]
    else:
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


#: Every git the grader runs trusts the task's repository whoever owns it:
#: without this, a repository owned by another uid than the runtime's makes
#: git refuse ("dubious ownership"), and the refusal would read as the
#: agent's doing -- a model_patch_unavailable 0 -- instead of failing the grade.
GIT = "git -c safe.directory='*'"

APPLY = (
    GIT + " apply -v --3way --recount --ignore-space-change --whitespace=nowarn %s"
    " || patch --batch --fuzz=5 -p1 -i %s"
)


#: The agent's diff against the base commit, written to a file: there is no
#: pipe, so git's own exit status is the command's. The options outvote the
#: repository's config, which the agent could have set: no external diff,
#: textconv, colour or fsmonitor program, and the test patch's prefixes.
DIFF = (
    GIT + " -c core.fsmonitor=false add -A && " + GIT + " -c core.quotepath=off -c core.fsmonitor=false diff"
    " --cached --binary --no-ext-diff --no-textconv --no-color --src-prefix=a/ --dst-prefix=b/ %s > %s"
)


def guest_directory() -> str:
    """Where one grading sandbox keeps the grade's files; new for every sandbox.

    Made with ``mkdir -m 700``, which refuses a path that exists, so nothing
    left in the agent's snapshot (a file or symlink planted at a fixed name)
    can stand in for a file the grader writes or reads there.
    """
    return "/tmp/sprout-grade-" + uuid4().hex


def _guest_directory(session) -> str:
    from sprout.worker.sandbox.commands import checked

    guest = guest_directory()
    checked(session, "mkdir -m 700 " + shlex.quote(guest), 120)
    return guest


def _create(session, image: str, resources: dict, failure: str) -> None:
    if not session.create(image, resources):
        reason = getattr(session, "create_error", "")
        raise RuntimeError(failure + (": " + reason if reason else ""))


def _upload(session, source: Path, destination: str) -> None:
    if not session.upload_file(source, destination):
        raise RuntimeError(
            f"Could not upload {source.name} to {destination} in the grading sandbox"
            + _transfer_error(session)
        )


def _download(session, source: str, destination: Path) -> None:
    if not session.download_file(source, destination):
        raise RuntimeError(f"Could not download {source} from the grading sandbox" + _transfer_error(session))


def _transfer_error(session) -> str:
    reason = getattr(session, "transfer_error", "")
    return ": " + reason if reason else ""


def _apply(path: str) -> str:
    return APPLY % (shlex.quote(path), shlex.quote(path))


def _zero(reason: str, error_output: str, common: dict, **extra) -> dict:
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
    spec: GradeSpec, row: dict, backend: dict, destination: Path, *, session_factory
) -> tuple[bytes | None, str]:
    """The agent's diff against the base commit, from its final snapshot, saved at ``destination``.

    ``(None, git's error)`` when the snapshot's repository cannot produce one
    (the agent removed ``.git`` or the base commit): the agent's doing, which
    the grade scores 0 instead of failing.
    """
    from sprout.worker.sandbox.commands import shell

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


def grade_swerebench(spec: GradeSpec, row: dict, backend: dict, directory: Path, *, session_factory) -> dict:
    from sprout.worker.sandbox.commands import checked, shell, shell_outcome

    parsers = load_parsers(spec)
    config = row["install_config"]
    if isinstance(config, str):
        config = json.loads(config)
    parser = parsers.NAME_TO_PARSER.get(config["log_parser"]) or getattr(parsers, config["log_parser"], None)
    if parser is None:
        raise ValueError("Unknown SWE-rebench log parser")
    commands = config["test_cmd"]
    if isinstance(commands, str):
        commands = [commands]
    if not commands or any(not isinstance(command, str) or not command.strip() for command in commands):
        raise ValueError("SWE-rebench test_cmd must contain commands")
    if not row["test_patch"].strip() or not row["base_commit"]:
        raise ValueError("SWE-rebench needs test_patch and base_commit")
    if not spec.image:
        raise ValueError("SWE-rebench grading needs the task image")
    workdir = "/" + row["repo"].split("/", 1)[1]
    script = (
        "set -uo pipefail\nexport PATH=" + shlex.quote(PATH) + "\n"
        "export CI=1 PAGER=cat MANPAGER=cat PIP_PROGRESS_BAR=off TQDM_DISABLE=1\n"
        "export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\nFAIL=0\n"
        + "\n".join(f"{command} || FAIL=1" for command in commands)
        + '\nexit "$FAIL"\n'
    )
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
        patch_file.write_bytes(b"")  # the path every result names; nothing was graded
    # test_patch is the hidden tests as applied, a path like model_patch: for the
    # review's leak judge (worker.jobs.review.leak.judge) only, never shown to the analyst.
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
        _create(
            session,
            spec.image,
            spec.resources,
            "Could not start a pristine SWE-rebench sandbox from the task image",
        )
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
        _, stderr, code, timed_out = shell_outcome(
            session,
            prefix + f"bash {shlex.quote(guest + '/eval.sh')} > {shlex.quote(guest_log)} 2>&1",
            int(spec.timeout_s),
        )
        if timed_out:
            # The log so far shows a reviewer where the tests hung; the verdict
            # does not need it, so a failed download costs only that.
            saved = session.download_file(guest_log, log)
            return _zero(
                "tests_timed_out",
                f"the tests did not finish within {int(spec.timeout_s)} s",
                common,
                **({"log": str(log)} if saved else {}),
            )
        _download(session, guest_log, log)
        if code not in (0, 1):
            # The tests' own output is in the log; stderr holds only the
            # outer shell's (a `cd` that failed), usually nothing.
            killed = f"exit code {code}" + (f"\n{stderr[-4000:]}" if stderr.strip() else "")
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
    finally:
        session.destroy()
