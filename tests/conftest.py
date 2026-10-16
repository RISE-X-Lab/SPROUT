"""Tests run against the source tree: ``src/`` is on the import path and, for the
child processes some tests spawn, on PYTHONPATH.

Also the fakes the podman backend's tests share: a ``podman``, a ``btrfs`` and
a ``findmnt`` executable, placed first on PATH, that record every argv they are
called with and answer the way the real ones do (podman 4.9 and 6.1 answer
alike for what the pool reads; btrfs-progs 6.6, util-linux 2.39), so what is
verified is the exact command line and how its
output is read. Each also appends to one shared log, so the order of calls
ACROSS tools is observable too (``recorded_sequence``): a container must be
removed before its rootfs subvolume is deleted, and the two are different
tools. Environment knobs make them misbehave the way the real tools do; each is
documented on the fake. ``tests/sdk/test_podman_live.py`` runs the same code
against the tools themselves.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_paths = [str(ROOT / "src"), str(ROOT)]
_current = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(_paths + ([_current] if _current else []))

from sprout_sandbox.podman_cli import default_podman_bin  # noqa: E402  (src is on the path above)

#: What the fake podman prints for the containers it "creates".
FAKE_CONTAINER = "a1" * 32
FAKE_EXTRACTION_CONTAINER = "c3" * 32
FAKE_EXPORT_BYTES = 4096

FAKE_PODMAN = r'''#!%(python)s
"""A podman that records what it was asked and answers like the real one.

FAKE_PODMAN_FAIL       comma-separated commands that exit 125
FAKE_PODMAN_HANG       commands that never return
FAKE_PODMAN_UNKNOWN    image references `image inspect` does not know until pulled
FAKE_PODMAN_USER       the OCI User every image carries (default: none, i.e. root)
FAKE_PODMAN_STATE      what a container inspect reports ("running 0")
FAKE_PODMAN_NETWORKS   the networks that exist before any `network create`: a JSON list of
                       {name, store, created}, each labelled like a sandbox's network of that store
                       (default: none); create and rm keep the set in the fake's state directory
FAKE_PODMAN_NET_REFUSE network connect/disconnect refuse: already connected / not connected
FAKE_PODMAN_GONE       `rm` finds no such container
FAKE_PODMAN_CONTAINERS what `ps` lists and `inspect --format json` describes: a JSON list of
                       {name, store, sockets, created}, each a managed container of that store
                       whose socket directory is `sockets` (default: none); `container exists`
                       answers yes for one of them until `rm` removes it
FAKE_PODMAN_NESTED     comma-separated paths, relative to the searched root, that `unshare find
                       ... -inum 256` reports as nested subvolumes when they exist as directories
"""
import hashlib, json, os, shutil, sys, time

argv = sys.argv[1:]
with open(os.environ["FAKE_PODMAN_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\n")
with open(os.environ["FAKE_PODMAN_LOG"] + ".cwd", "a") as log:
    log.write(json.dumps([argv[0] if argv else "", os.getcwd()]) + "\n")
with open(os.environ["FAKE_TOOL_LOG"], "a") as log:
    log.write(json.dumps(["podman"] + argv) + "\n")
state = os.path.dirname(os.environ["FAKE_PODMAN_LOG"])
command = argv[0] if argv else ""
if command in os.environ.get("FAKE_PODMAN_FAIL", "").split(","):
    sys.stderr.write("Error: simulated %%s failure\n" %% command)
    sys.exit(125)
if command in os.environ.get("FAKE_PODMAN_HANG", "").split(","):
    time.sleep(30)
pulled = os.path.join(state, "pulled")

if command == "image" and argv[1:2] == ["inspect"]:
    target = argv[-1]
    known = set(open(pulled).read().split()) if os.path.exists(pulled) else set()
    if target in os.environ.get("FAKE_PODMAN_UNKNOWN", "").split(",") and target not in known:
        sys.stderr.write("Error: %%s: image not known\n" %% target)
        sys.exit(125)
    digest = hashlib.sha256(target.encode()).hexdigest()
    print(json.dumps([{
        "Id": digest,
        "RepoDigests": ["%%s@sha256:%%s" %% (target.split(":")[0], digest)],
        "Config": {
            "Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
            "WorkingDir": "/testbed", "User": os.environ.get("FAKE_PODMAN_USER", ""),
            "Entrypoint": None, "Cmd": ["python3"],
        },
    }]))
elif command == "pull":
    with open(pulled, "a") as fh:
        fh.write(argv[-1] + "\n")
    print(argv[-1])
elif command == "create":
    extraction = "--entrypoint" in argv and argv[argv.index("--entrypoint") + 1] == "/bin/true"
    print("%(extraction)s" if extraction else "%(container)s")
elif command == "export":
    sys.stdout.buffer.write(b"\0" * %(export_bytes)d)
elif command == "unshare" and argv[1:2] == ["tar"]:
    data = sys.stdin.buffer.read()
    dest = argv[argv.index("-C") + 1]
    os.makedirs(os.path.join(dest, "etc"), exist_ok=True)
    with open(os.path.join(dest, "etc", "extracted"), "w") as fh:
        fh.write(str(len(data)))
elif command == "unshare" and argv[1:2] == ["find"]:
    root = argv[2]
    for rel in filter(None, os.environ.get("FAKE_PODMAN_NESTED", "").split(",")):
        if os.path.isdir(os.path.join(root, rel)):
            print(os.path.join(root, rel))
elif command == "unshare" and argv[1:3] == ["btrfs", "property"]:
    pass                                   # `property set <nested> ro false` inside the namespace
elif command == "unshare" and argv[1:4] == ["btrfs", "subvolume", "delete"]:
    shutil.rmtree(argv[4])                 # a nested subvolume, deleted inside the namespace
elif command == "rm":
    if os.environ.get("FAKE_PODMAN_GONE"):
        sys.stderr.write('Error: no container with name or ID "%%s" found: no such container\n' %% argv[-1])
        sys.exit(1)
    with open(os.path.join(state, "removed"), "a") as fh:
        fh.write(argv[-1] + "\n")
    print(argv[-1])
elif command == "container" and argv[1:2] == ["exists"]:
    # A FAKE_PODMAN_CONTAINERS entry exists until `rm` removes it.
    removed = set(open(os.path.join(state, "removed")).read().split()) if os.path.exists(os.path.join(state, "removed")) else set()
    names = {c["name"] for c in json.loads(os.environ.get("FAKE_PODMAN_CONTAINERS", "[]"))}
    sys.exit(0 if argv[-1] in names - removed else 1)
elif command == "run":
    print("%(container)s")
elif command == "start":
    print(argv[-1])
elif command == "network" and argv[1:2] in (["exists"], ["create"], ["rm"], ["ls"]):
    state_file = os.path.join(state, "networks.json")
    if os.path.exists(state_file):
        networks = json.load(open(state_file))
    else:
        networks = {n["name"]: {"labels": {"sprout.managed": "1", "sprout.store": n["store"]},
                                "created": n["created"]}
                    for n in json.loads(os.environ.get("FAKE_PODMAN_NETWORKS", "[]"))}
    name = argv[-1]
    if argv[1] == "exists":
        sys.exit(0 if name in networks else 1)
    if argv[1] == "ls":
        wanted = {a.split("=", 1)[1] for a in argv[2:] if a.startswith("label=")}
        print(json.dumps([{"name": n, "driver": "bridge", "created": d["created"], "labels": d["labels"]}
                          for n, d in networks.items()
                          if wanted <= {"%%s=%%s" %% item for item in d["labels"].items()}]))
        sys.exit(0)
    if argv[1] == "create":
        if name in networks:
            sys.stderr.write("Error: network name %%s already used: network already exists\n" %% name)
            sys.exit(125)
        labels = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "--label")
        networks[name] = {"labels": labels, "created": "2023-11-14T22:13:20.123456789Z",
                          "options": argv[2:-1]}
    elif name not in networks:
        sys.stderr.write("Error: unable to find network with name or ID %%s: network not found\n" %% name)
        sys.exit(125)
    else:
        del networks[name]
    json.dump(networks, open(state_file, "w"))
    print(name)
elif command == "network" and argv[1:2] in (["connect"], ["disconnect"]):
    if os.environ.get("FAKE_PODMAN_NET_REFUSE"):
        if argv[1] == "connect":
            sys.stderr.write("Error: container %%s is already connected to network %%s: "
                             "network is already connected\n" %% (argv[-1], argv[-2]))
        else:
            sys.stderr.write("Error: container %%s is not connected to network %%s: "
                             "network not found\n" %% (argv[-1], argv[-2]))
        sys.exit(125)
elif command == "ps":
    print(json.dumps([{
        "Id": hashlib.sha256(c["name"].encode()).hexdigest(), "Names": [c["name"]],
        "Labels": {"sprout.managed": "1", "sprout.store": c["store"]},
        "Mounts": ["/run/sprout"], "Created": c["created"], "State": "running",
    } for c in json.loads(os.environ.get("FAKE_PODMAN_CONTAINERS", "[]"))]))
elif command == "inspect" and "json" in argv:
    found = [c for c in json.loads(os.environ.get("FAKE_PODMAN_CONTAINERS", "[]")) if c["name"] == argv[-1]]
    if not found:
        sys.stderr.write('Error: no such container "%%s"\n' %% argv[-1])
        sys.exit(125)
    print(json.dumps([{
        "Config": {"Labels": {"sprout.managed": "1", "sprout.store": found[0]["store"]}},
        "Mounts": [{"Source": found[0]["sockets"], "Destination": "/run/sprout"}],
        "State": {"Status": "running"},
    }]))
elif command == "inspect":
    print(os.environ.get("FAKE_PODMAN_STATE", "running 0"))
elif command == "logs":
    print("fake logs")
elif command == "cp":
    pass
else:
    sys.stderr.write("fake podman: unknown command %%r\n" %% (argv,))
    sys.exit(2)
'''

FAKE_BTRFS = r'''#!%(python)s
"""A btrfs that keeps subvolumes as plain directories and records every argv.

FAKE_BTRFS_FAIL        comma-separated subcommands (create, snapshot, delete, set) that exit 1
"""
import json, os, shutil, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_BTRFS_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\n")
with open(os.environ["FAKE_TOOL_LOG"], "a") as log:
    log.write(json.dumps(["btrfs"] + argv) + "\n")
subcommand = argv[1] if len(argv) > 1 else ""
if subcommand in os.environ.get("FAKE_BTRFS_FAIL", "").split(","):
    sys.stderr.write("ERROR: simulated %%s failure\n" %% subcommand)
    sys.exit(1)

if argv[:2] == ["subvolume", "create"]:
    os.mkdir(argv[2])
    print("Create subvolume '%%s'" %% argv[2])
elif argv[:2] == ["subvolume", "snapshot"]:
    source, destination = [a for a in argv[2:] if a != "-r"]
    shutil.copytree(source, destination, symlinks=True)
    print("Create a %%ssnapshot of '%%s' in '%%s'" %% (
        "readonly " if "-r" in argv else "", source, destination))
elif argv[:2] == ["subvolume", "delete"]:
    shutil.rmtree(argv[2])
    sys.stderr.write("WARNING: cannot read default subvolume id: Operation not permitted\n")
    print("Delete subvolume 999 (no-commit): '%%s'" %% argv[2])
elif argv[:2] == ["property", "set"]:
    pass
else:
    sys.stderr.write("fake btrfs: unknown command %%r\n" %% (argv,))
    sys.exit(2)
'''

#: What the real findmnt reports for a store ops/btrfs_store.sh made.
STORE_MOUNT_OPTIONS = "rw,noatime,ssd,discard=async,space_cache=v2,user_subvol_rm_allowed,subvolid=5,subvol=/"

FAKE_FINDMNT = r'''#!%(python)s
"""A findmnt answering `-no FSTYPE,OPTIONS --target <path>`: the store's mount check.

FAKE_FINDMNT_FSTYPE    the filesystem type reported (default btrfs)
FAKE_FINDMNT_OPTIONS   the mount options reported (default: those ops/btrfs_store.sh sets)
"""
import json, os, sys

argv = sys.argv[1:]
with open(os.environ["FAKE_FINDMNT_LOG"], "a") as log:
    log.write(json.dumps(argv) + "\n")
with open(os.environ["FAKE_TOOL_LOG"], "a") as log:
    log.write(json.dumps(["findmnt"] + argv) + "\n")
if argv[:2] != ["-no", "FSTYPE,OPTIONS"] or "--target" not in argv:
    sys.stderr.write("fake findmnt: unknown command %%r\n" %% (argv,))
    sys.exit(2)
print("%%s %%s" %% (os.environ.get("FAKE_FINDMNT_FSTYPE", "btrfs"),
                  os.environ.get("FAKE_FINDMNT_OPTIONS", "%(mount_options)s")))
'''


def _install(tmp_path: Path, monkeypatch, name: str, source: str, log_var: str) -> Path:
    script = tmp_path / "bin" / name
    script.parent.mkdir(exist_ok=True)
    script.write_text(source % {
        "python": sys.executable, "container": FAKE_CONTAINER,
        "extraction": FAKE_EXTRACTION_CONTAINER, "export_bytes": FAKE_EXPORT_BYTES,
        "mount_options": STORE_MOUNT_OPTIONS})
    script.chmod(0o755)
    monkeypatch.setenv(log_var, str(tmp_path / ("%s.log" % name)))
    monkeypatch.setenv("FAKE_TOOL_LOG", str(tmp_path / "tools.log"))
    # the resolver would prefer an installed podman 6 or $SPROUT_PODMAN_BIN; the bare name
    # means PATH, where the fake is
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "podman")
    path = os.environ["PATH"]
    if not path.startswith(str(script.parent) + os.pathsep):
        monkeypatch.setenv("PATH", "%s%s%s" % (script.parent, os.pathsep, path))
    return script


def recorded_calls(tmp_path: Path, name: str) -> "list[list[str]]":
    """Every argv the fake ``name`` saw, in order."""
    log = tmp_path / ("%s.log" % name)
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


def recorded_cwds(tmp_path: Path) -> "list[tuple[str, str]]":
    """``(command, working directory)`` of every call the fake podman saw, in order."""
    log = tmp_path / "podman.log.cwd"
    if not log.exists():
        return []
    return [tuple(json.loads(line)) for line in log.read_text().splitlines()]


def sandbox_networks() -> "set[str]":
    """The networks the podman pool made -- one per sandbox, labelled like its
    container -- for the live tests' leak checks."""
    try:
        listed = subprocess.run([default_podman_bin(), "network", "ls", "-q", "--filter", "label=sprout.managed=1"],
                                capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        listed = []
    return set(listed)


def podman_store_in_use(store: Path) -> "str | None":
    """Why the live podman tests must not run now, or ``None``.

    Their fixtures diff the whole host's containers and the store before and
    after each test and remove what appeared, so a run using the store at the
    same time would have its containers and snapshots destroyed.
    """
    sandboxes = store / "sandboxes"
    live = sorted(p.name for p in sandboxes.iterdir() if p.is_dir()) if sandboxes.is_dir() else []
    if live:
        return "the store has live sandboxes (%s...)" % ", ".join(live[:3])
    try:
        managed = subprocess.run([default_podman_bin(), "ps", "-a", "-q", "--filter", "label=sprout.managed=1"],
                                 capture_output=True, text=True, timeout=60).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        managed = []
    if managed:
        return "%d sprout-managed containers exist" % len(managed)
    return None


def fake_networks(tmp_path: Path) -> "dict[str, dict]":
    """The networks the fake podman holds now, by name: each one's labels and creation options."""
    state = tmp_path / "networks.json"
    return json.loads(state.read_text()) if state.exists() else {}


def recorded_sequence(tmp_path: Path) -> "list[list[str]]":
    """Every call to every fake, in order, each prefixed with the tool's name."""
    log = tmp_path / "tools.log"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines()]


@pytest.fixture
def fake_podman(tmp_path: Path, monkeypatch) -> Path:
    """The fake podman, first on PATH, with its argv log wired up."""
    return _install(tmp_path, monkeypatch, "podman", FAKE_PODMAN, "FAKE_PODMAN_LOG")


@pytest.fixture
def fake_btrfs(tmp_path: Path, monkeypatch) -> Path:
    """The fake btrfs, first on PATH, with its argv log wired up."""
    return _install(tmp_path, monkeypatch, "btrfs", FAKE_BTRFS, "FAKE_BTRFS_LOG")


@pytest.fixture
def fake_findmnt(tmp_path: Path, monkeypatch) -> Path:
    """The fake findmnt, first on PATH: the store is on btrfs with user_subvol_rm_allowed unless told otherwise."""
    return _install(tmp_path, monkeypatch, "findmnt", FAKE_FINDMNT, "FAKE_FINDMNT_LOG")


@pytest.fixture
def store(tmp_path: Path, fake_btrfs: Path, fake_findmnt: Path) -> Path:
    """A store directory the fake btrfs and findmnt vouch for."""
    path = tmp_path / "store"
    path.mkdir()
    return path


@pytest.fixture
def runtime_bin(tmp_path: Path) -> Path:
    path = tmp_path / "sprout-runtime"
    path.write_bytes(b"\x7fELF fake runtime")
    return path
