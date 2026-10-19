"""A fake ``podman`` executable. It records each call and answers the way podman 4.9 and 6.1 answer.

It runs no container and holds no image. It answers each command from its argv, from the knobs below
and from its state files in the records directory ``$FAKE_TOOLS_DIR``: the images that it pulled, the
containers that it removed and the networks that it holds. It gives the output fields and the error
texts that the code reads. ``tests/support/test_fakes.py`` runs the same calls against the real
podman. ``tests.support.tools.FakePodman`` sets the knobs:

    FAKE_PODMAN_FAIL        commands (the first argv word) that exit 125 with an error
    FAKE_PODMAN_HANG        commands that sleep 30 s before they answer
    FAKE_PODMAN_UNKNOWN     image references that `image inspect` does not know until a `pull`
    FAKE_PODMAN_USER        the OCI User of every image (default: none, so root)
    FAKE_PODMAN_STATE       what `inspect --format` reports for any container (default "running 0")
    FAKE_PODMAN_CONTAINERS  JSON list of {name, store, sockets, created}: sandbox containers that exist
    FAKE_PODMAN_NETWORKS    JSON list of {name, store, created}: sandbox networks at the first call
    FAKE_PODMAN_NET_REFUSE  `network connect` and `network disconnect` refuse, as when already done
    FAKE_PODMAN_GONE        `rm` finds no such container
    FAKE_PODMAN_NESTED      relative paths that `unshare find` reports, besides the fake btrfs's subvolumes
"""

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

#: What `create` and `run` print for a sandbox container.
CONTAINER_ID = "a1" * 32
#: What `create --entrypoint /bin/true` prints: the container that an image is exported from.
EXTRACTION_CONTAINER_ID = "c3" * 32
#: The size of the archive that `export` writes. `unshare tar` writes it into ``etc/extracted``.
EXPORT_BYTES = 4096
#: The creation time of each network that `network create` makes, in podman's format.
NETWORK_CREATED = "2023-11-14T22:13:20.123456789Z"
#: The labels of a pool's sandbox containers and networks (sprout_sandbox.podman_containers).
MANAGED = "sprout.managed"
STORE = "sprout.store"


def main(argv: list[str]) -> int:
    records = Path(os.environ["FAKE_TOOLS_DIR"])
    _record(records, argv)
    command = argv[0] if argv else ""
    if command in _knob_list("FAKE_PODMAN_FAIL"):
        return _error("Error: simulated %s failure" % command, 125)
    if command in _knob_list("FAKE_PODMAN_HANG"):
        time.sleep(30)
    handler = _HANDLERS.get(command)
    if handler is None:
        return _unknown(argv)
    return handler(argv, records)


def _record(records: Path, argv: list[str]) -> None:
    for name, entry in (
        ("podman.log", argv),
        ("podman.cwd.log", [argv[0] if argv else "", os.getcwd()]),
        ("tools.log", ["podman", *argv]),
    ):
        with open(records / name, "a") as log:
            log.write(json.dumps(entry) + "\n")


def _knob_list(name: str) -> list[str]:
    return [item for item in os.environ.get(name, "").split(",") if item]


def _error(message: str, code: int) -> int:
    sys.stderr.write(message + "\n")
    return code


def _unknown(argv: list[str]) -> int:
    return _error("fake podman: unknown command %r" % (argv,), 2)


def _lines(path: Path) -> set[str]:
    return set(path.read_text().splitlines()) if path.exists() else set()


def _image(argv: list[str], records: Path) -> int:
    if argv[1:2] != ["inspect"]:
        return _unknown(argv)
    target = argv[-1]
    if target in _knob_list("FAKE_PODMAN_UNKNOWN") and target not in _lines(records / "podman.pulled"):
        return _error("Error: %s: image not known" % target, 125)
    digest = hashlib.sha256(target.encode()).hexdigest()
    repository = target.split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    config = {
        "Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
        "WorkingDir": "/testbed",
        "User": os.environ.get("FAKE_PODMAN_USER", ""),
        "Entrypoint": None,
        "Cmd": ["python3"],
    }
    print(
        json.dumps([{"Id": digest, "RepoDigests": ["%s@sha256:%s" % (repository, digest)], "Config": config}])
    )
    return 0


def _pull(argv: list[str], records: Path) -> int:
    with open(records / "podman.pulled", "a") as pulled:
        pulled.write(argv[-1] + "\n")
    print(argv[-1])
    return 0


def _create(argv: list[str], records: Path) -> int:
    extraction = "--entrypoint" in argv and argv[argv.index("--entrypoint") + 1] == "/bin/true"
    print(EXTRACTION_CONTAINER_ID if extraction else CONTAINER_ID)
    return 0


def _export(argv: list[str], records: Path) -> int:
    sys.stdout.buffer.write(b"\0" * EXPORT_BYTES)
    return 0


def _unshare(argv: list[str], records: Path) -> int:
    if argv[1:2] == ["tar"]:
        data = sys.stdin.buffer.read()
        etc = Path(argv[argv.index("-C") + 1]) / "etc"
        etc.mkdir(parents=True, exist_ok=True)
        (etc / "extracted").write_text(str(len(data)))
    elif argv[1:2] == ["find"]:
        for path in _nested_subvolumes(Path(argv[2]), records):
            print(path)
    elif argv[1:3] == ["btrfs", "property"]:
        pass
    elif argv[1:4] == ["btrfs", "subvolume", "delete"]:
        shutil.rmtree(argv[4])
    else:
        return _unknown(argv)
    return 0


def _nested_subvolumes(root: Path, records: Path) -> list[Path]:
    """The directories under ``root`` that are subvolumes: the knob's, then the fake btrfs's."""
    named = [root / relative for relative in _knob_list("FAKE_PODMAN_NESTED")]
    made = [Path(line) for line in sorted(_lines(records / "btrfs.subvolumes"))]
    found = named + [path for path in made if root in path.parents and path not in named]
    return [path for path in found if path.is_dir()]


def _rm(argv: list[str], records: Path) -> int:
    if os.environ.get("FAKE_PODMAN_GONE"):
        return _error('Error: no container with ID or name "%s" found: no such container' % argv[-1], 1)
    with open(records / "podman.removed", "a") as removed:
        removed.write(argv[-1] + "\n")
    print(argv[-1])
    return 0


def _containers() -> list[dict]:
    return json.loads(os.environ.get("FAKE_PODMAN_CONTAINERS", "[]"))


def _container_labels(container: dict) -> dict[str, str]:
    return {MANAGED: "1", STORE: container["store"]}


def _container(argv: list[str], records: Path) -> int:
    if argv[1:2] != ["exists"]:
        return _unknown(argv)
    names = {container["name"] for container in _containers()}
    return 0 if argv[-1] in names - _lines(records / "podman.removed") else 1


def _run(argv: list[str], records: Path) -> int:
    print(CONTAINER_ID)
    return 0


def _start(argv: list[str], records: Path) -> int:
    print(argv[-1])
    return 0


def _label_filters(argv: list[str]) -> set[str]:
    """The ``key=value`` of each ``--filter label=key=value`` in ``argv``."""
    return {arg.split("=", 1)[1] for arg in argv[1:] if arg.startswith("label=")}


def _matches(labels: dict[str, str], wanted: set[str]) -> bool:
    return wanted <= {"%s=%s" % item for item in labels.items()}


def _ps(argv: list[str], records: Path) -> int:
    listed = [c for c in _containers() if _matches(_container_labels(c), _label_filters(argv))]
    ids = [hashlib.sha256(c["name"].encode()).hexdigest() for c in listed]
    if "-q" in argv:
        # Without --no-trunc, podman prints each id cut to 12 characters.
        if ids:
            print("\n".join(identifier[:12] for identifier in ids))
        return 0
    entries = [
        {
            "Id": identifier,
            "Names": [container["name"]],
            "Labels": _container_labels(container),
            "Mounts": ["/run/sprout"],
            "Created": container["created"],
            "State": "running",
        }
        for identifier, container in zip(ids, listed, strict=True)
    ]
    print(json.dumps(entries))
    return 0


def _inspect(argv: list[str], records: Path) -> int:
    if "json" not in argv:
        print(os.environ.get("FAKE_PODMAN_STATE", "running 0"))
        return 0
    found = [container for container in _containers() if container["name"] == argv[-1]]
    if not found:
        return _error('Error: no such container "%s"' % argv[-1], 125)
    entry = {
        "Config": {"Labels": _container_labels(found[0])},
        "Mounts": [{"Source": found[0]["sockets"], "Destination": "/run/sprout"}],
        "State": {"Status": "running"},
    }
    print(json.dumps([entry]))
    return 0


def _networks(records: Path) -> dict[str, dict]:
    state = records / "podman.networks.json"
    if state.exists():
        return json.loads(state.read_text())
    given = json.loads(os.environ.get("FAKE_PODMAN_NETWORKS", "[]"))
    return {n["name"]: {"labels": {MANAGED: "1", STORE: n["store"]}, "created": n["created"]} for n in given}


def _network(argv: list[str], records: Path) -> int:
    subcommand, name = (argv[1] if len(argv) > 1 else ""), argv[-1]
    if subcommand in ("connect", "disconnect"):
        return _network_change(subcommand, argv)
    if subcommand not in ("exists", "create", "rm", "ls"):
        return _unknown(argv)
    networks = _networks(records)
    if subcommand == "exists":
        return 0 if name in networks else 1
    if subcommand == "ls":
        wanted = _label_filters(argv)
        listed = [
            {"name": key, "driver": "bridge", "created": value["created"], "labels": value["labels"]}
            for key, value in networks.items()
            if _matches(value["labels"], wanted)
        ]
        print(json.dumps(listed))
        return 0
    if subcommand == "create":
        if name in networks:
            return _error("Error: network name %s already used: network already exists" % name, 125)
        labels = dict(argv[i + 1].split("=", 1) for i, arg in enumerate(argv) if arg == "--label")
        networks[name] = {"labels": labels, "created": NETWORK_CREATED, "options": argv[2:-1]}
    elif name not in networks:
        return _error("Error: unable to find network with name or ID %s: network not found" % name, 1)
    else:
        del networks[name]
    (records / "podman.networks.json").write_text(json.dumps(networks))
    print(name)
    return 0


def _network_change(subcommand: str, argv: list[str]) -> int:
    if not os.environ.get("FAKE_PODMAN_NET_REFUSE"):
        return 0
    if subcommand == "connect":
        message = "Error: container %s is already connected to network %s: network is already connected"
    else:
        message = "Error: container %s is not connected to network %s: network not found"
    return _error(message % (argv[-1], argv[-2]), 125)


def _logs(argv: list[str], records: Path) -> int:
    print("fake logs")
    return 0


def _cp(argv: list[str], records: Path) -> int:
    return 0


_HANDLERS = {
    "image": _image,
    "pull": _pull,
    "create": _create,
    "export": _export,
    "unshare": _unshare,
    "rm": _rm,
    "container": _container,
    "run": _run,
    "start": _start,
    "network": _network,
    "ps": _ps,
    "inspect": _inspect,
    "logs": _logs,
    "cp": _cp,
}

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
