"""Backend selection (harness_sandbox/backends.py).

The session names no concrete pool: it hands its execution section here and gets
back a pool built from the named backend's settings -- `MicroVMPool` from the
`microvm` section, `PodmanPool` from the `podman` one. Every call site once named
a pool class outright, so switching one was an edit in several files.

Covered:
- no config means podman: the default backend
- each pool is built from its own section
- a missing required setting names the setting, not a connection failure
- a mistyped setting is an error, not a silent default
- an unknown backend lists the backends there are
- env vars as a fallback; api_key_file so a key need not sit in a config
- one AgentENV server for the callers that assume one, AENV_SERVER_URL first
- the VM lease covers microvm only and leaves a podman config alone
- no call site names a concrete pool any more

The podman pool checks its store's mount with ``findmnt`` at construction, so
these run with the fake findmnt and btrfs from ``tests/conftest.py`` on PATH
(the ``store`` fixture).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout_sandbox import MicroVMPool, PodmanPool
from sprout.harness_sandbox.backends import (BACKENDS, DEFAULT_AENV_SERVER_URL, BackendError,
                                             backend_name, build_pool, default_aenv_server_url,
                                             snapshot_store, with_sandbox_budget)

#: Stand-ins, in the parametrized configs below, for the per-test store and
#: runtime binary: both live under tmp_path, so a row cannot name them.
STORE, RUNTIME_BIN = "<store>", "<runtime_bin>"


def _config(template: dict, store: Path, runtime_bin: Path) -> dict:
    """``template`` with STORE and RUNTIME_BIN replaced by this test's real paths."""
    real = {STORE: str(store), RUNTIME_BIN: str(runtime_bin)}
    config = dict(template)
    for name, section in template.items():
        if isinstance(section, dict):
            config[name] = {key: real.get(value, value) if isinstance(value, str) else value
                            for key, value in section.items()}
    return config


def _environment(monkeypatch, env: dict) -> None:
    """Set each variable for the test, or unset it for a value of ``None``."""
    for name, value in env.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


# --------------------------------------------------------------------------- #
#  Defaults
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("config, name", [
    ({}, "podman"), ({"backend": "  MicroVM "}, "microvm"), ({"backend": "Podman"}, "podman")])
def test_backend_name_defaults_to_podman_and_normalizes(config, name):
    assert backend_name(config) == name


def test_the_registry_has_both_backends():
    assert set(BACKENDS) == {"microvm", "podman"}


def test_microvm_builds_from_its_own_section():
    pool = build_pool({
        "backend": "microvm",
        "microvm": {"server_url": "http://127.0.0.1:8000",
                    "runtime_port": 4000, "sandbox_ttl": 900},
    })
    assert isinstance(pool, MicroVMPool)
    assert pool.server_url == "http://127.0.0.1:8000"
    assert pool.runtime_port == 4000
    assert pool.sandbox_ttl == 900


def test_podman_builds_from_its_own_section(store, runtime_bin, tmp_path):
    pool = build_pool({
        "backend": "podman",
        "podman": {"store": str(store), "runtime_bin": str(runtime_bin),
                   "podman_bin": "/opt/podman/bin/podman",
                   "allow_internet": False, "socket_dir": str(tmp_path / "socks")},
    })
    assert isinstance(pool, PodmanPool)
    assert pool.store.path == store
    assert pool.runtime_bin == runtime_bin.resolve()
    assert pool.podman_bin == "/opt/podman/bin/podman"
    assert pool.allow_internet is False
    assert pool.socket_dir == tmp_path / "socks"


def test_podman_defaults(store, runtime_bin, monkeypatch):
    """What a bare section gets: podman on PATH, sockets under the store,
    egress allowed. No network setting: every sandbox gets its own, so a
    config still naming the one network a store's sandboxes once shared is
    refused rather than silently ignored."""
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "podman")
    pool = build_pool({"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin)}})
    assert pool.podman_bin == "podman"
    assert pool.socket_dir == store / "sockets"
    assert pool.allow_internet is None
    with pytest.raises(BackendError, match="unknown podman setting.*network"):
        build_pool({"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin),
                                                    "network": "lab"}})


def test_a_bare_section_runs_the_podman_sprout_podman_bin_names(store, runtime_bin, monkeypatch):
    """ops/setup_podman.sh installs podman 6 off PATH and prints the variable;
    `sprout.services env` exports it; a section that names no podman_bin runs it."""
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "/home/me/.local/opt/podman/bin/podman")
    pool = build_pool({"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin)}})
    assert pool.podman_bin == "/home/me/.local/opt/podman/bin/podman"
    # an explicit podman_bin in the section still wins
    pool = build_pool({"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin),
                                                       "podman_bin": "podman"}})
    assert pool.podman_bin == "podman"


def test_podman_accepts_the_top_level_runtime_bin(store, runtime_bin):
    """The CLI hands --runtime-bin over at the top level too, as the template
    builder reads it; either spelling names the same binary."""
    pool = build_pool({"backend": "podman", "runtime_bin": str(runtime_bin), "podman": {"store": str(store)}})
    assert isinstance(pool, PodmanPool) and pool.runtime_bin == runtime_bin.resolve()


def test_a_tilde_in_a_podman_path_is_the_home_directory(store, runtime_bin, monkeypatch):
    """A YAML config names `~/sprout-btrfs` where a shell would have expanded it;
    unexpanded, the store would be refused as `<cwd>/~/sprout-btrfs`."""
    monkeypatch.setenv("HOME", str(store.parent))
    pool = build_pool({"backend": "podman",
                       "podman": {"store": "~/" + store.name, "runtime_bin": "~/" + runtime_bin.name,
                                  "socket_dir": "~/socks"}})
    assert pool.store.path == store and pool.runtime_bin == runtime_bin.resolve()
    assert pool.socket_dir == store.parent / "socks"
    assert snapshot_store({"backend": "podman", "podman": {"store": "~/" + store.name}}) == str(store)


def test_snapshot_store_is_the_podman_store_and_nothing_for_microvm(store):
    """What the orchestrator writes on `sandbox.created`, so sweep can find a
    podman run's subvolumes from its journal alone."""
    assert snapshot_store({"backend": "podman", "podman": {"store": str(store)}}) == str(store)
    assert snapshot_store({"backend": "podman", "podman": {}}) is None
    assert snapshot_store({"backend": "microvm", "microvm": {"server_url": "http://x"}}) is None
    assert snapshot_store({}) is None


# --------------------------------------------------------------------------- #
#  Misconfiguration is reported, never guessed
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("config, env, message", [
    # Naming no backend selects podman, which needs no service; its pool still
    # needs a runtime binary and a store, so an empty config names what is missing.
    pytest.param({}, {}, "needs podman.runtime_bin", id="no-config-is-podman-without-its-binary"),
    # Without this the microVM pool would build and fail later as a connection
    # error, which reads as "the host is down" rather than "you did not say where".
    pytest.param({"backend": "microvm"}, {"AENV_SERVER_URL": None}, "microvm.server_url",
                 id="microvm-without-a-server"),
    # The binary is mounted into every container; without it the container would
    # start with no entrypoint and the failure would read as a bad image.
    pytest.param({"backend": "podman", "podman": {"store": STORE}}, {}, "podman.runtime_bin",
                 id="podman-without-a-runtime-binary"),
    pytest.param({"backend": "podman", "podman": {"store": STORE, "runtime_bin": "/nope/sprout-runtime"}}, {},
                 "runtime_bin does not exist", id="podman-runtime-binary-that-does-not-exist"),
    # A missing store names the setting and the script that makes one.
    pytest.param({"backend": "podman", "podman": {"runtime_bin": RUNTIME_BIN}}, {},
                 r"podman\.store.*ops/btrfs_store\.sh", id="podman-without-a-store"),
    # The pool's refusal, naming the mount it found, as a BackendError: the run's
    # error then says what to fix rather than that the image is bad. (The full
    # wording is the SDK's, pinned in tests/sdk/test_podman.py.)
    pytest.param({"backend": "podman", "podman": {"store": STORE, "runtime_bin": RUNTIME_BIN}},
                 {"FAKE_FINDMNT_FSTYPE": "ext4"}, "backend 'podman': store .* is on ext4",
                 id="podman-store-off-btrfs"),
    # A mistyped setting is an error, not a silent default: `sandbox_tll: 900`
    # would leave the server's 15 s TTL in place and look like a VM that dies
    # mid-turn; `allow_internt: false` would leave every sandbox online and look
    # like a no-network task that ran online; `squash_at` is a key that is gone.
    pytest.param({"backend": "microvm", "microvm": {"server_url": "http://x", "sandbox_tll": 900}}, {},
                 "sandbox_tll", id="microvm-sandbox_tll"),
    pytest.param({"backend": "podman",
                  "podman": {"store": STORE, "runtime_bin": RUNTIME_BIN, "allow_internt": False}},
                 {}, "unknown podman setting.*allow_internt", id="podman-allow_internt"),
    pytest.param({"backend": "podman",
                  "podman": {"store": STORE, "runtime_bin": RUNTIME_BIN, "squash_at": 5}},
                 {}, "unknown podman setting.*squash_at", id="podman-squash_at"),
    # A scalar where a section is expected, rather than its characters reported
    # as unknown settings.
    pytest.param({"backend": "microvm", "microvm": "http://x"}, {}, "must be a mapping",
                 id="microvm-section-is-a-scalar"),
    pytest.param({"backend": "podman", "podman": "runtime/sprout-runtime"}, {}, "must be a mapping",
                 id="podman-section-is-a-scalar"),
    # A named key file that is not there, rather than a fall-through to an empty
    # key that would read as a bad key (401) at the first spawn.
    pytest.param({"backend": "microvm", "microvm": {"server_url": "http://x", "api_key_file": "/nope/key"}},
                 {}, "api_key_file not found", id="microvm-key-file-that-does-not-exist"),
])
def test_a_misconfiguration_is_named_not_guessed(config, env, message, store, runtime_bin, monkeypatch):
    """Every misconfiguration build_pool can see is a BackendError naming the
    setting; none builds a pool that fails later as something else."""
    _environment(monkeypatch, env)
    with pytest.raises(BackendError, match=message):
        build_pool(_config(config, store, runtime_bin))


def test_an_unknown_backend_lists_the_backends():
    """A config written for a backend that does not exist fails loudly rather
    than quietly running on the wrong one, and says what does exist."""
    with pytest.raises(BackendError, match="unknown backend 'docker'.*'microvm'.*'podman'"):
        build_pool({"backend": "docker"})


# --------------------------------------------------------------------------- #
#  Credentials
# --------------------------------------------------------------------------- #

#: Where a row's key file is: under HOME, which the test points at tmp_path, so
#: the `~` of the backends.py example is exercised too.
KEY_FILE = "~/aenv-key"


def _sent_api_key(pool) -> str:
    """The key as the server will see it — the X-API-KEY header, which is what
    AgentENV validates (a Bearer Authorization header is not)."""
    return pool._client.headers.get("X-API-KEY", "")


@pytest.mark.parametrize("config, key_file, env, server_url, key", [
    # Nothing in the config: endpoint and key from the environment, as the CLI
    # leaves them so a key is never typed where shell history would keep it.
    pytest.param({"backend": "microvm"}, None,
                 {"AENV_SERVER_URL": "http://from-env:8000", "AENV_API_KEY": "key-from-env"},
                 "http://from-env:8000", "key-from-env", id="from-the-environment"),
    # A key written into a YAML config tends to reach a commit; a file keeps it
    # out, `~` expanded and the trailing newline stripped.
    pytest.param({"backend": "microvm", "microvm": {"server_url": "http://x", "api_key_file": KEY_FILE}},
                 "secret-from-file\n", {"AENV_API_KEY": None},
                 "http://x", "secret-from-file", id="from-a-file"),
    # Both named: the config's wins, so a stale file never shadows an explicit key.
    pytest.param({"backend": "microvm", "microvm": {"server_url": "http://x", "api_key": "inline",
                                                    "api_key_file": KEY_FILE}},
                 "from-file", {}, "http://x", "inline", id="config-beats-file"),
])
def test_the_api_key_comes_from_config_then_file_then_env(config, key_file, env, server_url, key,
                                                          tmp_path: Path, monkeypatch):
    """The order _read_api_key documents, as the server will see the result."""
    monkeypatch.setenv("HOME", str(tmp_path))
    _environment(monkeypatch, env)
    if key_file is not None:
        (tmp_path / Path(KEY_FILE).name).write_text(key_file)
    pool = build_pool(config)
    assert isinstance(pool, MicroVMPool)
    assert pool.server_url == server_url
    assert _sent_api_key(pool) == key


def test_the_callers_that_assume_a_server_assume_one(monkeypatch):
    """The Harbor environment and the cleanup client fall back to a server when
    none is named, and a trial's memory snapshot is released through the second:
    they once fell back to different ports. AgentENV's own default is 8000;
    AENV_SERVER_URL comes first and a named URL before both."""
    from sprout.harness_orchestrator.reap import AgentEnvClient

    monkeypatch.delenv("AENV_SERVER_URL", raising=False)
    assert default_aenv_server_url() == DEFAULT_AENV_SERVER_URL == "http://127.0.0.1:8000"
    assert AgentEnvClient().base_url == DEFAULT_AENV_SERVER_URL
    monkeypatch.setenv("AENV_SERVER_URL", "http://bench:18000/")
    assert default_aenv_server_url() == "http://bench:18000/"
    assert AgentEnvClient().base_url == "http://bench:18000"
    assert AgentEnvClient("http://named:1").base_url == "http://named:1"


# --------------------------------------------------------------------------- #
#  The VM lease is a microvm matter
# --------------------------------------------------------------------------- #

def test_the_sandbox_budget_leaves_a_podman_config_alone(store, runtime_bin):
    """A container has no TTL. The orchestrator applies the lease to every run's
    backend, so a podman config must pass through rather than gain a `microvm`
    section or trip over one it does not have."""
    config = {"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin)}}
    assert with_sandbox_budget(config, 1800) == config
    assert with_sandbox_budget({"backend": "microvm"}, 1800) == {
        "backend": "microvm", "microvm": {"sandbox_ttl": 2400}}


def test_no_template_is_built_for_podman(store, runtime_bin):
    """Templates exist because a microVM cold-started from a plain image has no
    runtime; the podman pool mounts it into every container, so the session must
    not go looking for AgentENV to build one."""
    from sprout.harness_sandbox.templates import builder_from_backend

    assert builder_from_backend({"backend": "podman",
                                 "podman": {"store": str(store), "runtime_bin": str(runtime_bin)}}) is None


# --------------------------------------------------------------------------- #
#  Harness plumbing
# --------------------------------------------------------------------------- #

def test_no_call_site_names_a_concrete_pool():
    """The point of this module: one place decides how a pool is built, so its
    settings are a config change rather than an edit in several files."""
    repo = Path(__file__).resolve().parents[2]
    session = (repo / "src" / "sprout" / "harness_sandbox" / "session.py").read_text()
    assert "build_pool(" in session, "the session should build its pool via backends.py"
    for path in sorted((repo / "src" / "sprout" / "harness").rglob("*.py")):
        if "tests" in path.parts or path.name == "backends.py":
            continue
        text = path.read_text()
        assert "MicroVMPool(" not in text, f"{path.name} constructs a MicroVMPool"
        assert "PodmanPool(" not in text, f"{path.name} constructs a PodmanPool"


@pytest.mark.parametrize("backend, pool_type", [("podman", PodmanPool), ("microvm", MicroVMPool)])
def test_what_the_cli_hands_over_is_what_build_pool_takes(backend, pool_type, store, runtime_bin,
                                                          monkeypatch):
    """`harness run --backend B --store S --runtime-bin R` is exactly the section
    build_pool takes: nothing else is needed, nothing else is set. For microvm
    that section carries `from_image` (a name on a command line is an OCI
    reference, and only the cold-start path accepts one); unknown keys are
    rejected rather than ignored, so build_pool has to declare it or every CLI
    microVM run would be refused as a typo."""
    from sprout.harness.cli import _backend_section

    # The endpoint is the one setting the CLI leaves to the environment.
    monkeypatch.setenv("AENV_SERVER_URL", "http://from-env:8000")
    sections = {"podman": {"store": str(store), "runtime_bin": str(runtime_bin)},
                "microvm": {"from_image": True, "runtime_bin": str(runtime_bin)}}
    config = _backend_section(backend, str(runtime_bin), str(store))
    assert config == {"backend": backend, backend: sections[backend]}
    assert isinstance(build_pool(config), pool_type)


def test_both_clients_mount_the_runtime_at_one_path():
    """The SDK may not import from `sprout`, so the podman pool carries its own
    copy of the template's RUNTIME_PATH; this keeps the two from drifting."""
    from sprout.harness_sandbox import templates
    from sprout_sandbox import podman

    assert podman.RUNTIME_PATH == templates.RUNTIME_PATH
