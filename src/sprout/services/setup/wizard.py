"""The guided setup, ``python -m sprout.services setup``: four steps, each with its questions.

1. The worker's host checks, each with its fix. The user can stop after a failed check.
2. The config files: ``init`` writes the missing ones, and the setup prints each problem in
   runstore.json with its fix.
3. The model endpoint, its model and the variable of its key, which the profiles of the actor and
   the review then reference. Before the rewrite, the old file goes to a backup directory. An
   optional request checks the endpoint.
4. The start of the Run Store and a worker, and the next commands.

Each question has a default and a flag that answers it. ``Prompter`` reads the other answers from
its input stream, or, with ``yes``, takes their defaults. The steps print through ``Prompter``.
Plain functions compute each line and each change. No line holds the value of a key.
"""

from __future__ import annotations

import json
import re
import shlex
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TextIO, TypeVar

from sprout.contract.sampling import endpoint as endpoint_prefix
from sprout.services.health import HostError, HostReport, Probe
from sprout.services.render import doctor_line
from sprout.services.setup import findings, templates
from sprout.services.spec import Spec
from sprout.services.state import Paths

#: The default of each question that has no default from the environment.
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_KEY_ENV = "SPROUT_MODEL_API_KEY"
#: The profiles of the template whose jobs call the model: the actor's rollouts and the reviews.
KEY_PROFILES = ("mini", "harness", "mini-microvm", "review")
#: The services that the last step starts, and the dependencies that ``up`` starts with them.
START = ("runstore-api", "runstore-worker")
#: The output budget of the test request. Arbitrary: enough for a short answer, and cheap.
_CHECK_MAX_TOKENS = 16
#: Arbitrary: a slow endpoint answers a short request within a minute.
_CHECK_TIMEOUT_S = 60.0
#: The fix of each host check of the worker's doctor, by the name of the check.
_HOST_FIXES = {
    "podman": "ops/setup_podman.sh",
    "store": "ops/setup_podman.sh",
    "runtime": "cd runtime && go generate ./... && go build -o sprout-runtime .",
    "keyring": "sudo sysctl kernel.keys.maxkeys=20000 kernel.keys.maxbytes=2000000",
}
_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_INDENT = "      "
_T = TypeVar("_T")


class SetupError(RuntimeError):
    """The setup cannot go on, for example because a flag has an invalid value or the input ended."""


@dataclass(frozen=True)
class Answers:
    """The answer of each flag to its question. None asks the question, or takes the default with ``yes``."""

    yes: bool = False
    go_on: bool | None = None
    model: str | None = None
    endpoint: str | None = None
    key_env: str | None = None
    add_key: bool | None = None
    remove_unset: bool | None = None
    check_endpoint: bool | None = None
    start: bool | None = None


@dataclass(frozen=True)
class Session:
    """What the setup works on: the host, the answers, the terminal and the commands of the CLI."""

    paths: Paths
    environ: Mapping[str, str]
    answers: Answers
    prompter: Prompter
    #: The report of the worker's doctor. Raises ``HostError`` when the doctor gives none.
    report: Callable[[], HostReport]
    #: Writes the missing config files and secrets as ``init`` does, and returns the spec, or None
    #: after an error on stderr.
    seed: Callable[[Callable[[str], None]], Spec | None]
    #: Starts ``START`` and their dependencies, and returns True when each one is up.
    start: Callable[[Callable[[str], None]], bool]
    now: Callable[[], datetime] = datetime.now


@dataclass(frozen=True)
class _Model:
    """The model of the experiments: its name, its endpoint, and the variable that holds its key."""

    name: str
    endpoint: str
    key_env: str


class Prompter:
    """Asks the questions of the setup on ``stdin`` and ``stdout``, or answers them from flags.

    A question that a flag answers is printed with the answer and the flag. With ``yes``, a question
    without a flag takes its default. Without ``yes``, the end of ``stdin`` raises ``SetupError``.
    """

    def __init__(self, *, stdin: TextIO, stdout: TextIO, yes: bool) -> None:
        self._stdin = stdin
        self._stdout = stdout
        self._yes = yes
        # A terminal echoes what the user types. For scripted input, the transcript shows the answer.
        self._echo = not stdin.isatty()

    def say(self, line: str = "") -> None:
        self._stdout.write(line + "\n")
        self._stdout.flush()

    def confirm(self, question: str, *, default: bool, answer: bool | None, flag: str) -> bool:
        """The answer to a yes-or-no ``question``. ``flag`` and ``--no-<flag>`` answer it."""
        hint = "[Y/n]" if default else "[y/N]"
        if answer is not None:
            shown = "yes" if answer else "no"
            return self._given("%s %s" % (question, hint), shown, flag if answer else _negated(flag), answer)
        while True:
            reply = self._ask("%s %s " % (question, hint), default="yes" if default else "no").lower()
            if reply in ("y", "yes", "n", "no"):
                return reply.startswith("y")
            self.say("Answer y or n.")

    def text(
        self,
        question: str,
        *,
        default: str,
        answer: str | None,
        flag: str,
        check: Callable[[str], str | None],
    ) -> str:
        """The answer to ``question``, which ``check`` accepts. ``flag`` answers it.

        ``check`` returns the problem of an answer, or None. A bad answer on the input asks again. A
        bad answer from ``flag`` or from the default with ``yes`` raises ``SetupError``.
        """
        if answer is not None:
            problem = check(answer)
            if problem is not None:
                raise SetupError("%s %s: %s" % (flag, shlex.quote(answer), problem))
            return self._given("%s [%s]:" % (question, default), answer, flag, answer)
        while True:
            reply = self._ask("%s [%s]: " % (question, default), default=default)
            problem = check(reply)
            if problem is None:
                return reply
            if self._yes:
                raise SetupError("%s %s: %s (pass %s)" % (question, shlex.quote(reply), problem, flag))
            self.say(problem)

    def _given(self, prompt: str, shown: str, flag: str, value: _T) -> _T:
        """``value``, after the line that shows ``prompt`` with the answer ``shown`` and its ``flag``."""
        self.say("%s %s  (%s)" % (prompt, shown, flag))
        return value

    def _ask(self, prompt: str, *, default: str) -> str:
        """The stripped reply to ``prompt``, or ``default`` for an empty reply or with ``yes``."""
        if self._yes:
            return self._given(prompt.rstrip(), default, "--yes", default)
        self._stdout.write(prompt)
        self._stdout.flush()
        line = self._stdin.readline()
        if not line:
            self._stdout.write("\n")
            raise SetupError(
                "the input ended before an answer: pass --yes and the flags to set up without a terminal"
            )
        if self._echo:
            self._stdout.write(line if line.endswith("\n") else line + "\n")
        return line.strip() or default


# --- the steps ----------------------------------------------------------------


def run(session: Session) -> int:
    """Run the four steps. Return 1 when the user or a failure stopped the setup, or a start failed."""
    say = session.prompter.say
    say("Sprout setup, 4 steps. Each question shows its default in brackets. Enter takes it.")
    try:
        if not _host_step(session):
            say("stopped: fix the host, then run `python -m sprout.services setup` again")
            return 1
        spec = _config_step(session)
        if spec is None:
            say("stopped: services.yaml is not valid, as the error above says")
            return 1
        model = _model_step(session, _Backup(directory=session.paths.config, now=session.now))
        started = _finish_step(session, spec=spec, model=model)
    except SetupError as error:
        say("stopped: %s" % error)
        return 1
    return 0 if started else 1


def _host_step(session: Session) -> bool:
    """Print the worker's host checks, and return False when the user stops after a failed one."""
    say = session.prompter.say
    say("")
    say("1/4 Host checks (python -m sprout.worker doctor)")
    try:
        checks = session.report().checks
    except HostError as error:
        checks = {"worker": Probe(False, str(error))}
    for line in _host_lines(checks):
        say(line)
    if all(check.ok or check.warning for check in checks.values()):
        return True
    return session.prompter.confirm(
        "A host check failed. Go on with the config files?",
        default=True,
        answer=session.answers.go_on,
        flag="--continue-after-failure",
    )


def _host_lines(checks: Mapping[str, Probe]) -> list[str]:
    """A doctor line for each check, and a fix line under each check that did not pass.

    The fix line is left out when the detail of the check already gives the fix.
    """
    lines = []
    for name, check in checks.items():
        lines.append(doctor_line(check.word, name, check.detail))
        fix = _HOST_FIXES.get(name)
        if not check.ok and fix is not None and fix not in check.detail:
            lines.append("%sfix: %s" % (_INDENT, fix))
    return lines


def _config_step(session: Session) -> Spec | None:
    """Write the missing config files, print each problem in runstore.json, and return the spec."""
    say = session.prompter.say
    say("")
    say("2/4 Config files (%s)" % session.paths.config)
    spec = session.seed(say)
    if spec is None:
        return None
    runstore = (session.paths.config / findings.RUNSTORE).read_text(encoding="utf-8")
    found = findings.host_findings(runstore=runstore, environ=session.environ)
    for finding in found:
        say(doctor_line(finding.status, findings.RUNSTORE, finding.problem))
        say("%sfix: %s" % (_INDENT, finding.fix))
    if not found:
        say("ok    no problem found in runstore.json")
    return spec


def _model_step(session: Session, backup: _Backup) -> _Model | None:
    """Ask for the model, reference its key in the profiles, and offer a test request."""
    prompter, answers, environ = session.prompter, session.answers, session.environ
    prompter.say("")
    prompter.say("3/4 Model endpoint")
    url = prompter.text(
        "Model endpoint URL, empty to skip this step",
        default=environ.get("QWEN_BASE_URL") or environ.get("OPENAI_BASE_URL") or "",
        answer=answers.endpoint,
        flag="--model-endpoint",
        check=_endpoint_problem,
    )
    if not url:
        prompter.say("%sskipped: no model endpoint" % _INDENT)
        return None
    name = prompter.text(
        "Model name", default=DEFAULT_MODEL, answer=answers.model, flag="--model", check=_name_problem
    )
    key_env = prompter.text(
        "Environment variable that holds the key",
        default=DEFAULT_KEY_ENV,
        answer=answers.key_env,
        flag="--model-key-env",
        check=_variable_problem,
    )
    model = _Model(name=name, endpoint=endpoint_prefix(url), key_env=key_env)
    prompter.say(_INDENT + _key_state(key_env, environ))
    text = (session.paths.config / findings.RUNSTORE).read_text(encoding="utf-8")
    try:
        changed = _reference_key(session, text=text, key_env=key_env)
    except findings.ConfigError as error:
        prompter.say("%scannot reference the key: %s. Edit the file by hand." % (_INDENT, error))
        changed = text
    _write(
        session.paths,
        backup=backup,
        texts={findings.RUNSTORE: text},
        changed={findings.RUNSTORE: changed},
        say=prompter.say,
    )
    _offer_check(session, model=model)
    return model


def _reference_key(session: Session, *, text: str, key_env: str) -> str:
    """``text`` with the key reference that the user adds and the unset references that they remove."""
    prompter, answers = session.prompter, session.answers
    lacking = _profiles_without(text, variable=key_env)
    if lacking and prompter.confirm(
        "Add %s to the worker_env of %s?" % (_reference(key_env), ", ".join(lacking)),
        default=True,
        answer=answers.add_key,
        flag="--add-key",
    ):
        text = _add_reference(text, variable=key_env, profiles=lacking)
    unset = findings.unset_references(findings.load_runstore(text)["profiles"], session.environ)
    for variable, profiles in unset.items():
        if variable != key_env and prompter.confirm(
            "%s is not set in this shell. Remove its reference from the worker_env of %s?"
            % (variable, ", ".join(profiles)),
            default=False,
            answer=answers.remove_unset,
            flag="--remove-unset",
        ):
            text = _remove_reference(text, variable=variable)
    return text


def _offer_check(session: Session, *, model: _Model) -> None:
    """Send the test request if the user asks for it, and print how it went."""
    prompter = session.prompter
    if not prompter.confirm(
        "Send one test request to %s/v1/chat/completions? It is one chat completion with max_tokens %d. "
        "It costs a request, and it can count against the rate limit of a free tier."
        % (model.endpoint, _CHECK_MAX_TOKENS),
        default=False,
        answer=session.answers.check_endpoint,
        flag="--check-endpoint",
    ):
        return
    key = session.environ.get(model.key_env)
    if key is None:
        prompter.say("%sskipped: %s is not set in this shell" % (_INDENT, model.key_env))
        return
    prompter.say(_INDENT + _check_endpoint(endpoint=model.endpoint, model=model.name, key=key))


def _finish_step(session: Session, *, spec: Spec, model: _Model | None) -> bool:
    """Offer to start the Run Store and a worker, then print the next commands.

    Returns False when a service that the user chose to start did not come up.
    """
    prompter = session.prompter
    prompter.say("")
    prompter.say("4/4 Start")
    if model is not None:
        prompter.say(
            "The worker takes the environment of the shell that starts it, so that shell must set %s."
            % model.key_env
        )
    prompter.say(
        "A worker that runs already reads changed profiles only after "
        "`python -m sprout.services restart runstore-worker`."
    )
    if prompter.confirm(
        "Start the Run Store and a worker now (python -m sprout.services up %s)?" % " ".join(START),
        default=False,
        answer=session.answers.start,
        flag="--start",
    ):
        started = session.start(prompter.say)
    else:
        started = True
    prompter.say("")
    prompter.say("Next, in the shell of the experiment:")
    for line in _next_commands(model=model, runstore_url=_runstore_url(spec)):
        prompter.say("  " + line)
    return started


# --- what the steps compute ---------------------------------------------------


def _profiles_without(text: str, *, variable: str) -> list[str]:
    """Each profile of ``KEY_PROFILES`` in the runstore.json ``text`` whose worker_env lacks ``variable``."""
    profiles = findings.load_runstore(text)["profiles"]
    return [
        name
        for name in KEY_PROFILES
        if isinstance(profiles.get(name), dict) and variable not in (profiles[name].get("worker_env") or {})
    ]


def _add_reference(text: str, *, variable: str, profiles: Sequence[str]) -> str:
    """The runstore.json ``text``, where the worker_env of each of ``profiles`` references ``variable``."""
    config = findings.load_runstore(text)
    for name in profiles:
        profile = config["profiles"][name]
        profile["worker_env"] = {**(profile.get("worker_env") or {}), variable: {"$env": variable}}
    return templates.json_text(config)


def _remove_reference(text: str, *, variable: str) -> str:
    """The runstore.json ``text`` without each worker_env entry that is a reference to ``variable``."""
    config = findings.load_runstore(text)
    for profile in config["profiles"].values():
        worker_env = profile.get("worker_env") if isinstance(profile, dict) else None
        if isinstance(worker_env, dict):
            profile["worker_env"] = {k: v for k, v in worker_env.items() if v != {"$env": variable}}
    return templates.json_text(config)


def _runstore_url(spec: Spec) -> str:
    """The URL of the Run Store API that services.yaml serves: its ``runstore-api`` on its ``port``."""
    ports = [service.port for service in spec.services if service.name == "runstore-api" and service.port]
    return "http://127.0.0.1:%d" % ports[0] if ports else "<Run Store URL>"


def _next_commands(*, model: _Model | None, runstore_url: str) -> list[str]:
    """The ``env`` line, and an example run of ``examples.qwen.run`` with ``--plan`` on ``model``."""
    if model is None:
        chosen = "--model <model> --model-endpoint <url> --model-key-env <key variable>"
    else:
        chosen = "--model %s --model-endpoint %s --model-key-env %s" % (
            shlex.quote(model.name),
            shlex.quote(model.endpoint),
            shlex.quote(model.key_env),
        )
    example = [
        "python -m examples.qwen.run --dataset deepswe --method baseline",
        "--tasks-dir <tasks directory> --task <task> --output <output directory>",
        chosen,
        "--runstore-url %s --plan" % runstore_url,
    ]
    return ['eval "$(python -m sprout.services env)"', " ".join(example)]


def _check_endpoint(*, endpoint: str, model: str, key: str) -> str:
    """Send one short chat completion to ``endpoint`` with ``key``, and return a line on how it went.

    The line holds no part of ``key``, even when the endpoint echoes it.
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the word ok."}],
        "max_tokens": _CHECK_MAX_TOKENS,
    }
    request = urllib.request.Request(
        endpoint + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_CHECK_TIMEOUT_S) as response:
            status, reply = response.status, response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        status, reply = error.code, error.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, ValueError) as error:
        return _redact("fail  no answer from %s: %s" % (endpoint, error), key)
    if status == 200 and '"choices"' in reply:
        line = "ok    HTTP 200: the model answered"
    else:
        line = "fail  HTTP %d: %s" % (status, " ".join(reply.split())[:300])
    return _redact(line, key)


class _Backup:
    """The copies of the files that one setup session rewrites, in one timestamped directory.

    The directory is ``backup-<YYYYmmdd-HHMMSS>`` beside the files. Each file is copied once, before
    its first rewrite, so the directory holds the files as they were before the session.
    """

    def __init__(self, *, directory: Path, now: Callable[[], datetime]) -> None:
        self._parent = directory
        self._now = now
        self._directory: Path | None = None

    def save(self, path: Path, say: Callable[[str], None]) -> None:
        """Copy ``path`` into the backup directory, unless an earlier call copied it."""
        if self._directory is None:
            self._directory = _new_directory(
                self._parent, "backup-%s" % self._now().strftime("%Y%m%d-%H%M%S")
            )
            say("backup %s" % self._directory)
        if not (self._directory / path.name).exists():
            shutil.copy2(path, self._directory / path.name)


def _new_directory(parent: Path, name: str) -> Path:
    """A new directory ``parent/name``, or ``parent/name-2`` and so on when the name is taken."""
    for suffix in ("", *("-%d" % n for n in range(2, 100))):
        candidate = parent / (name + suffix)
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise SetupError("%s holds 99 backup directories named %s" % (parent, name))


def _write(
    paths: Paths,
    *,
    backup: _Backup,
    texts: Mapping[str, str],
    changed: Mapping[str, str],
    say: Callable[[str], None],
) -> None:
    """Write each file of ``changed`` whose text differs from ``texts``, after its backup."""
    for name, text in changed.items():
        if text == texts.get(name):
            continue
        path = paths.config / name
        backup.save(path, say)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
        say("wrote  %s" % path)


def _key_state(variable: str, environ: Mapping[str, str]) -> str:
    if variable in environ:
        return "%s is set in this shell." % variable
    return "%s is not set in this shell. Export it before you start the worker." % variable


def _reference(variable: str) -> str:
    return json.dumps({variable: {"$env": variable}})


def _name_problem(value: str) -> str | None:
    return None if value.strip() else "the model name is empty"


def _endpoint_problem(value: str) -> str | None:
    """Why ``value`` is not an endpoint URL that a rollout request accepts, or None. Empty skips the step."""
    if not value:
        return None
    try:
        endpoint_prefix(value)
    except ValueError as error:
        return str(error)
    return None


def _variable_problem(value: str) -> str | None:
    return None if _VARIABLE.fullmatch(value) else "%r is not the name of an environment variable" % value


def _redact(text: str, secret: str) -> str:
    return text.replace(secret, "***") if secret else text


def _negated(flag: str) -> str:
    return "--no-" + flag[2:]
