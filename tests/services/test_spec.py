"""services.yaml: the packaged template is a valid spec, and a wrong one says what and where."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.services import spec as spec_module
from sprout.services.spec import (Context, ExpandError, SpecError, expand, load_spec, parse_spec, podman_bin_for,
                                  store_for, template_path)
from sprout.services.state import Paths


def _service(**fields):
    body = {"kind": "process", "command": ["true"]}
    body.update(fields)
    return body


def test_the_packaged_template_loads_with_every_service():
    spec = load_spec(template_path("services.yaml"))
    assert spec.names == ("db", "proxy", "runstore-api", "runstore-worker", "rl-driver", "podman", "store", "aenv")
    kinds = {service.name: service.kind for service in spec.services}
    assert kinds == {"db": "podman", "proxy": "process", "runstore-api": "process", "runstore-worker": "process",
                     "rl-driver": "process", "podman": "check", "store": "check", "aenv": "external"}
    assert not spec.get("aenv").enabled and spec.get("proxy").timeout_s == 120
    # dependencies first, declaration order among the rest
    assert [s.name for s in spec.order(["rl-driver"])] == ["db", "runstore-api", "rl-driver"]
    assert [s.name for s in spec.order()] == ["db", "proxy", "runstore-api", "podman", "store", "runstore-worker", "rl-driver"]
    assert [s.name for s in spec.with_dependents(["db"])] == ["db", "runstore-api", "runstore-worker", "rl-driver"]
    # a consumer inherits its dependencies' exports, transitively: the worker's sandboxes run on the
    # podman and store an experiment shell gets from `env`
    worker = dict(spec.exports_for(spec.get("runstore-worker")))
    assert set(worker) == {"SPROUT_RUNSTORE_DSN", "SPROUT_RUNSTORE_TOKEN", "SPROUT_PODMAN_BIN", "SPROUT_PODMAN_STORE"}
    assert worker["SPROUT_PODMAN_BIN"] == "{podman_bin}" and worker["SPROUT_PODMAN_STORE"] == "{store}"
    assert spec.secret_names() == ("db-password", "litellm-master-key", "runstore-token", "rl-driver-token")
    # Both authenticated services probe with their own token: an unauthenticated probe of the
    # RL driver's /health answers 401 and would report a healthy driver as down.
    for name, secret in (("runstore-api", "runstore-token"), ("rl-driver", "rl-driver-token")):
        assert dict(spec.get(name).health.headers) == {"Authorization": f"Bearer {{secret:{secret}}}"}
    assert spec.get("db").volumes == ("{state}/db/data:/var/lib/postgresql/data:Z",)
    assert spec.get("proxy").unset_env == ("FORCE_COLOR",)
    # the three places a secret may stand, all used by the template
    assert spec.get("db").env[-1] == ("POSTGRES_PASSWORD", "{secret:db-password}")
    assert spec.get("runstore-api").health.headers == (("Authorization", "Bearer {secret:runstore-token}"),)
    assert dict(spec.get("proxy").exports)["OPENAI_API_KEY"] == "{secret:litellm-master-key}"


def test_an_unknown_dependency_names_the_service_and_the_field():
    with pytest.raises(SpecError, match=r"service 'b', field 'depends_on': unknown service 'zzz'"):
        parse_spec({"services": {"a": _service(), "b": _service(depends_on=["zzz"])}})


def test_a_dependency_cycle_is_refused_with_the_loop_spelled_out():
    with pytest.raises(SpecError, match=r"field 'depends_on': dependency cycle a -> b -> a"):
        parse_spec({"services": {"a": _service(depends_on=["b"]), "b": _service(depends_on=["a"])}})


@pytest.mark.parametrize("body, field", [
    (_service(kind="systemd"), "field 'kind': 'systemd' is not one of process, podman, external, check"),
    (_service(health={"kind": "tcp"}), "field 'health.kind': 'tcp' is not one of http, pg_isready, process, none"),
    (_service(health={"kind": "http"}), "field 'health.url'"),
    (_service(port="4000"), "field 'port': must be an integer port"),
    (_service(port=70000), "field 'port'"),
    ({"kind": "process"}, "field 'command': a process needs a non-empty argv list"),
    ({"kind": "process", "commands": ["true"]}, "field 'commands': not a field of a process service"),
    (_service(**{"depends-on": ["a"]}), "field 'depends-on': not a field of a process service"),
    ({"kind": "podman", "image": "x"}, "field 'container'"),
    ({"kind": "podman", "image": "x", "container": "{state}-db"}, "field 'container': .*placeholders are not expanded"),
    ({"kind": "podman", "image": "{env:REGISTRY}/x", "container": "c"}, "field 'image': .*placeholders are not expanded"),
    ({"kind": "check", "check": "disk"}, "field 'check': 'disk' is not one of btrfs_store"),
    (_service(env={"X": "{secrets:db}"}), r"field 'env.X': unknown placeholder \{secrets:db\}"),
    (_service(prepare=[{"unless": "x"}]), r"field 'prepare\[0\]'"),
    (_service(timeout_s=0), "field 'timeout_s'"),
    # a secret belongs in a value that becomes an environment, never in something logged as typed
    (_service(command=["sleep", "30", "{secret:db-password}"]), "field 'command': a secret may not appear here"),
    (_service(prepare=[{"run": ["echo", "{secret:x}"]}]), r"field 'prepare\[0\].run': a secret may not appear here"),
    (_service(prepare=[{"run": ["true"], "unless": "{secret:x}"}]), r"field 'prepare\[0\].unless': a secret"),
    ({"kind": "podman", "image": "x", "container": "c", "volumes": ["{secret:x}:/d"]}, "field 'volumes': a secret"),
    ({"kind": "podman", "image": "x", "container": "c", "publish": "{secret:x}"}, "field 'publish': a secret"),
    (_service(health={"kind": "http", "url": "http://127.0.0.1:1/{secret:x}"}), "field 'health.url': a secret"),
    ({"kind": "check", "check": "btrfs_store", "path": "{secret:x}"}, "field 'path': a secret"),
])
def test_a_bad_value_names_the_service_and_the_field(body, field):
    with pytest.raises(SpecError, match=field) as caught:
        parse_spec({"services": {"a": {"kind": "process", "command": ["true"]}, "svc": body}}, "my.yaml")
    assert str(caught.value).startswith("my.yaml: service 'svc', field ")


def test_a_secret_is_accepted_where_it_becomes_an_environment_value():
    spec = parse_spec({"services": {"svc": _service(env={"KEY": "{secret:k}"}, exports={"TOKEN": "x {secret:t}"},
                                                    health={"kind": "http", "url": "http://127.0.0.1:1/",
                                                            "headers": {"Authorization": "Bearer {secret:t}"}})}})
    assert spec.secret_names() == ("k", "t")


def test_placeholders_expand_against_the_context(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    paths = Paths(config=tmp_path / "c", state=tmp_path / "s", repo=tmp_path / "r")
    ctx = Context(paths, "/venv/bin/python", lambda name: "S-" + name, {"X": "1"})
    assert expand("{python} -m x", ctx) == "/venv/bin/python -m x"
    assert expand("{state}/db:{config}/f:{repo}", ctx) == "%s/s/db:%s/c/f:%s/r" % (tmp_path, tmp_path, tmp_path)
    assert expand("pw={secret:db-password} x={env:X}", ctx) == "pw=S-db-password x=1"
    assert expand("~/sprout-btrfs", ctx) == str(tmp_path / "sprout-btrfs")
    assert expand('{"mini": {"cwd": "/testbed"}}', ctx) == '{"mini": {"cwd": "/testbed"}}'   # JSON is left alone
    with pytest.raises(ExpandError, match="NOPE is not set"):
        expand("{env:NOPE}", ctx)

    def missing(name: str) -> str:
        raise ValueError("secret %r does not exist yet: run `python -m sprout.services init`" % name)

    with pytest.raises(ExpandError, match="secret 'x' does not exist yet"):     # what `require` says, as one kind of error
        expand("{secret:x}", Context(paths, "/venv/bin/python", missing, {}))


def test_the_podman_and_store_placeholders_follow_the_entry_points_rule(tmp_path, monkeypatch):
    """`{podman_bin}`: $SPROUT_PODMAN_BIN, else the podman ops/setup_podman.sh installed
    under ~/.local/opt/podman, else `podman` on PATH; `{store}`: $SPROUT_PODMAN_STORE, else
    ~/sprout-btrfs. What the two checks export, so `env` hands every entry point the same."""
    paths = Paths(config=tmp_path / "c", state=tmp_path / "s", repo=tmp_path / "r")
    installed = tmp_path / "opt" / "podman" / "bin" / "podman"
    monkeypatch.setattr(spec_module, "installed_podman", lambda: installed)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    ctx = Context(paths, "/venv/bin/python", lambda name: "S-" + name, {})
    assert expand("{podman_bin}", ctx) == "podman" and podman_bin_for({}) == "podman"
    assert expand("{store}", ctx) == str(tmp_path / "home" / "sprout-btrfs") and store_for({}) == tmp_path / "home" / "sprout-btrfs"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\n")
    installed.chmod(0o755)
    assert expand("{podman_bin}", ctx) == str(installed)
    named = Context(paths, "/venv/bin/python", lambda name: "S-" + name,
                    {"SPROUT_PODMAN_BIN": "/elsewhere/podman", "SPROUT_PODMAN_STORE": "/mnt/store"})
    assert expand("{podman_bin}:{store}", named) == "/elsewhere/podman:/mnt/store"
    # a check may export; the other kinds already could
    spec = parse_spec({"services": {"podman": {"kind": "check", "check": "podman",
                                                 "exports": {"SPROUT_PODMAN_BIN": "{podman_bin}"}}}}, "t.yaml")
    assert dict(spec.get("podman").exports) == {"SPROUT_PODMAN_BIN": "{podman_bin}"}
    with pytest.raises(SpecError, match="not one of btrfs_store, podman"):
        parse_spec({"services": {"x": {"kind": "check", "check": "docker"}}}, "t.yaml")


def test_the_services_and_the_sandbox_package_name_the_same_installed_podman():
    """services imports no layer it starts, so the install path of ops/setup_podman.sh is
    spelled in both packages; they must agree or `env` would export one podman and the
    pool pick another."""
    from sprout_sandbox.podman_cli import installed_podman as sandbox_installed_podman
    assert spec_module.installed_podman() == sandbox_installed_podman()
