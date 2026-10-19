"""``services.yaml``: a wrong file is refused at load, the services are ordered by their dependencies,
and a placeholder becomes its value only when a command runs.

The packaged template is one valid file. Its tests pin the wiring between its services and the
programs that they start: a wrong copy of a variable name passes the loader and fails only on a host.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sprout.services.spec import (
    Context,
    ExpandError,
    Spec,
    SpecError,
    expand,
    load_spec,
    parse_spec,
    template_path,
)
from sprout.services.state import Paths


def process(**fields):
    """A valid process service, with ``fields`` over its defaults."""
    return {"kind": "process", "command": ["true"], **fields}


def container(**fields):
    """A valid podman service, with ``fields`` over its defaults."""
    return {"kind": "podman", "image": "docker.io/library/postgres:16-alpine", "container": "db", **fields}


def names(services) -> tuple[str, ...]:
    return tuple(service.name for service in services)


def template() -> Spec:
    return load_spec(template_path("services.yaml"))


# --- the dependency graph -----------------------------------------------------


def test_order_puts_each_dependency_before_its_dependent_and_keeps_the_file_order_otherwise():
    spec = parse_spec(
        {
            "services": {
                "worker": process(depends_on=["api"]),
                "proxy": process(),
                "api": process(depends_on=["db"]),
                "db": process(),
            }
        }
    )

    assert names(spec.order()) == ("db", "api", "worker", "proxy")


def test_order_of_a_named_service_adds_its_dependencies_and_nothing_else():
    spec = parse_spec({"services": {"db": process(), "proxy": process(), "api": process(depends_on=["db"])}})

    assert names(spec.order(["api"])) == ("db", "api")


@pytest.mark.parametrize("wanted, expected", [(None, ("db", "api")), (["aenv"], ("aenv",))])
def test_a_disabled_service_is_left_out_unless_it_is_named(wanted, expected):
    spec = parse_spec(
        {"services": {"db": process(), "aenv": process(enabled=False), "api": process(depends_on=["db"])}}
    )

    assert names(spec.order(wanted)) == expected


def test_with_dependents_adds_each_service_that_depends_on_a_named_one_transitively_in_file_order():
    spec = parse_spec(
        {
            "services": {
                "driver": process(depends_on=["api"]),
                "api": process(depends_on=["db"]),
                "db": process(),
                "proxy": process(),
            }
        }
    )

    assert names(spec.with_dependents(["db"])) == ("driver", "api", "db")


def test_a_process_gets_the_exports_of_its_dependencies_transitively_and_its_own_replace_theirs():
    spec = parse_spec(
        {
            "services": {
                "db": process(exports={"DSN": "db-dsn", "LEVEL": "db"}),
                "api": process(depends_on=["db"], exports={"TOKEN": "api-token", "LEVEL": "api"}),
                "worker": process(depends_on=["api"], exports={"LEVEL": "worker"}),
                "proxy": process(exports={"KEY": "proxy-key"}),
            }
        }
    )

    exports = spec.exports_for(spec.get("worker"))

    assert dict(exports) == {"DSN": "db-dsn", "TOKEN": "api-token", "LEVEL": "worker"}


def test_secret_names_lists_each_secret_once_in_the_order_of_first_use():
    """``init`` makes these secrets. A secret may stand in ``env``, ``exports`` and ``health.headers``."""
    probe = {
        "kind": "http",
        "url": "http://127.0.0.1:1/",
        "headers": {"Authorization": "Bearer {secret:probe}"},
    }
    spec = parse_spec(
        {
            "services": {
                "a": process(
                    env={"PASSWORD": "{secret:db-password}"},
                    exports={"BOTH": "{secret:token} {secret:db-password}"},
                ),
                "b": process(env={"AGAIN": "{secret:token}"}, health=probe),
            }
        }
    )

    assert spec.secret_names() == ("db-password", "token", "probe")


# --- what the loader refuses --------------------------------------------------


@pytest.mark.parametrize(
    "body, field",
    [
        (["true"], "kind"),
        (process(kind="systemd"), "kind"),
        (process(**{"depends-on": ["a"]}), "depends-on"),
        (process(enabled="no"), "enabled"),
        (process(depends_on="a"), "depends_on"),
        (process(depends_on=["zzz"]), "depends_on"),
        (process(port="4000"), "port"),
        (process(port=True), "port"),
        (process(port=0), "port"),
        (process(port=65536), "port"),
        (process(timeout_s=0), "timeout_s"),
        (process(timeout_s=True), "timeout_s"),
        (process(env=["X=1"]), "env"),
        (process(env={"X": ["a"]}), "env.X"),
        ({"kind": "process"}, "command"),
        (process(command=["echo", {"a": 1}]), "command"),
        (process(prepare={"run": ["true"]}), "prepare"),
        (process(prepare=[{"unless": "/path/to/done"}]), "prepare[0]"),
        (process(prepare=[{"run": []}]), "prepare[0].run"),
        (process(health="http"), "health"),
        (process(health={"kind": "tcp"}), "health.kind"),
        (process(health={"kind": "none", "timeout": 3}), "health.timeout"),
        (process(health={"kind": "http"}), "health.url"),
        (process(health={"kind": "pg_isready"}), "health.port"),
        (container(image=""), "image"),
        (container(image="{env:REGISTRY}/postgres"), "image"),
        (container(container=""), "container"),
        (container(container="{state}-db"), "container"),
        # YAML reads an unquoted 15432:5432 as the base-60 number 931352.
        (container(publish=931352), "publish"),
        ({"kind": "check", "check": "disk"}, "check"),
        (process(env={"X": "{secrets:db}"}), "env.X"),
        (process(command=["sleep", "{secret:x}"]), "command"),
        (process(unset_env=["{secret:x}"]), "unset_env"),
        (process(prepare=[{"run": ["echo", "{secret:x}"]}]), "prepare[0].run"),
        (process(prepare=[{"run": ["true"], "unless": "{secret:x}"}]), "prepare[0].unless"),
        (container(volumes=["{secret:x}:/data"]), "volumes"),
        (container(publish="{secret:x}"), "publish"),
        (process(health={"kind": "http", "url": "http://127.0.0.1:1/{secret:x}"}), "health.url"),
        ({"kind": "check", "check": "btrfs_store", "path": "{secret:x}"}, "path"),
    ],
)
def test_a_wrong_field_is_refused_with_the_file_the_service_and_the_field_named(body, field):
    data = {"services": {"a": process(), "svc": body}}

    with pytest.raises(SpecError) as caught:
        parse_spec(data, "my.yaml")

    assert str(caught.value).startswith("my.yaml: service 'svc', field '%s': " % field), caught.value


def test_a_dependency_cycle_is_refused_with_the_loop_spelled_out():
    data = {
        "services": {
            "a": process(depends_on=["b"]),
            "b": process(depends_on=["c"]),
            "c": process(depends_on=["a"]),
        }
    }

    with pytest.raises(SpecError) as caught:
        parse_spec(data, "my.yaml")

    assert str(caught.value) == "my.yaml: service 'a', field 'depends_on': dependency cycle a -> b -> c -> a"


@pytest.mark.parametrize("name", ["Web", "1web", "web_api", "-web"])
def test_a_service_name_that_is_not_lower_case_letters_digits_and_dashes_is_refused(name):
    with pytest.raises(SpecError, match=r"^my\.yaml: service name '%s': " % re.escape(name)):
        parse_spec({"services": {name: process()}}, "my.yaml")


@pytest.mark.parametrize(
    "data, problem",
    [
        (["services"], "the file must be a mapping"),
        ({"service": {"a": process()}}, r"unknown top-level keys \['service'\]"),
        ({"python": 3, "services": {"a": process()}}, "`python` must be a string"),
        ({"services": {}}, "`services` must be a non-empty mapping"),
    ],
)
def test_a_wrong_file_shape_is_refused_with_the_file_named(data, problem):
    with pytest.raises(SpecError, match=r"^my\.yaml: %s" % problem):
        parse_spec(data, "my.yaml")


def test_a_file_that_is_not_yaml_is_refused_with_its_path(tmp_path):
    path = tmp_path / "services.yaml"
    path.write_text("services: [\n")

    with pytest.raises(SpecError, match=r"^%s: not valid YAML: " % re.escape(str(path))):
        load_spec(path)


# --- what the loader accepts --------------------------------------------------


@pytest.mark.parametrize("port", [1, 65535])
def test_a_port_at_either_limit_is_accepted(port):
    spec = parse_spec({"services": {"svc": process(port=port)}})

    assert spec.get("svc").port == port


def test_a_number_in_an_argv_or_an_environment_is_read_as_its_string():
    """YAML reads the 4000 of ``--port 4000`` and a bare ``user: 70`` as numbers."""
    spec = parse_spec({"services": {"svc": process(command=["litellm", "--port", 4000], env={"UID": 70})}})

    service = spec.get("svc")
    assert (service.command, service.env) == (("litellm", "--port", "4000"), (("UID", "70"),))


def test_a_json_object_in_an_argument_is_not_taken_for_a_placeholder():
    argument = '{"mini": {"cwd": "/testbed"}}'

    spec = parse_spec({"services": {"svc": process(command=["run", argument])}})

    assert spec.get("svc").command == ("run", argument)


def test_a_pg_isready_probe_of_a_container_runs_in_that_container_by_default():
    """The probe falls back to ``podman exec <container> pg_isready`` when the host has no pg_isready."""
    probe = {"kind": "pg_isready", "port": 15432}

    spec = parse_spec({"services": {"db": container(container="sprout-services-db", health=probe)}})

    assert spec.get("db").health.container == "sprout-services-db"


# --- placeholders -------------------------------------------------------------


def context(*, environ=None, secret=None, host=None) -> Context:
    """A context that resolves each placeholder to a placeholder path or a value named after it."""
    hosts = {"podman_bin": "/path/to/podman", "store": "/path/to/store"}
    return Context(
        paths=Paths(config=Path("/path/to/config"), state=Path("/path/to/state"), repo=Path("/path/to/repo")),
        python="/path/to/python",
        secret=secret or (lambda name: "value-of-" + name),
        host=host or hosts.__getitem__,
        environ={"AENV_SERVER_URL": "http://127.0.0.1:18000"} if environ is None else environ,
    )


@pytest.mark.parametrize(
    "text, expected",
    [
        ("{python} -m sprout.worker", "/path/to/python -m sprout.worker"),
        ("{repo}/runtime", "/path/to/repo/runtime"),
        ("{config}/litellm.yaml", "/path/to/config/litellm.yaml"),
        ("{state}/db/data:/var/lib/postgresql/data:Z", "/path/to/state/db/data:/var/lib/postgresql/data:Z"),
        ("password={secret:db-password}", "password=value-of-db-password"),
        ("{env:AENV_SERVER_URL}/sandboxes", "http://127.0.0.1:18000/sandboxes"),
        ("{podman_bin} {store}", "/path/to/podman /path/to/store"),
        ("~/sprout-btrfs", "/path/to/home/sprout-btrfs"),
        ('{"mini": {"cwd": "/testbed"}}', '{"mini": {"cwd": "/testbed"}}'),
    ],
)
def test_each_placeholder_becomes_its_value_from_the_context(text, expected, monkeypatch):
    monkeypatch.setenv("HOME", "/path/to/home")

    assert expand(text, context()) == expected


def no_secret(name: str) -> str:
    raise ValueError("secret %r does not exist yet: run `python -m sprout.services init`" % name)


def no_doctor(name: str) -> str:
    raise ValueError("python -m sprout.worker doctor --json gave no report (exit 1): boom")


@pytest.mark.parametrize(
    "text, message",
    [
        ("{env:AENV_API_KEY}", "environment variable AENV_API_KEY is not set"),
        (
            "{secret:db-password}",
            "secret 'db-password' does not exist yet: run `python -m sprout.services init`",
        ),
        ("{store}", "python -m sprout.worker doctor --json gave no report (exit 1): boom"),
    ],
)
def test_a_placeholder_that_this_run_cannot_resolve_is_one_expand_error_with_the_reason(text, message):
    ctx = context(environ={}, secret=no_secret, host=no_doctor)

    with pytest.raises(ExpandError) as caught:
        expand(text, ctx)

    assert str(caught.value) == message


# --- the packaged template ----------------------------------------------------


def received_variables(spec: Spec, name: str) -> set[str]:
    """The names of the variables that the CLI adds to the environment of the process ``name``."""
    service = spec.get(name)
    return set(dict(spec.exports_for(service))) | set(dict(service.env))


@pytest.mark.parametrize(
    "service, variables",
    [
        # sprout/runstore/__main__.py reads both.
        ("runstore-api", {"SPROUT_RUNSTORE_DSN", "SPROUT_RUNSTORE_TOKEN"}),
        # The worker reads the DSN. Its podman pool reads the podman and the store.
        ("runstore-worker", {"SPROUT_RUNSTORE_DSN", "SPROUT_PODMAN_BIN", "SPROUT_PODMAN_STORE"}),
        ("proxy", {"LITELLM_MASTER_KEY"}),
    ],
)
def test_each_template_process_receives_the_variables_that_its_program_reads(service, variables):
    received = received_variables(template(), service)

    assert variables - received == set()


@pytest.mark.parametrize(
    "service, token", [("runstore-api", "SPROUT_RUNSTORE_TOKEN"), ("rl-driver", "SPROUT_RL_DRIVER_TOKEN")]
)
def test_each_template_service_with_a_token_probes_with_the_token_that_it_is_started_with(service, token):
    """Each route of these services needs the token. A healthy service answers 401 to a probe without it."""
    entry = template().get(service)

    assert dict(entry.health.headers) == {"Authorization": "Bearer " + dict(entry.exports)[token]}


def test_the_template_gives_the_clients_of_the_proxy_its_master_key():
    proxy = template().get("proxy")

    exports = dict(proxy.exports)
    assert (exports["OPENAI_API_KEY"], exports["SPROUT_LITELLM_KEY"]) == (
        dict(proxy.env)["LITELLM_MASTER_KEY"],
    ) * 2


def dsn_fields(dsn: str) -> dict[str, str]:
    return dict(pair.split("=", 1) for pair in dsn.split())


def test_the_template_database_publishes_probes_and_exports_one_host_port():
    """The host's own PostgreSQL owns 5432, so the container is published on 15432."""
    db = template().get("db")

    dsn = dsn_fields(dict(db.exports)["SPROUT_RUNSTORE_DSN"])
    assert (db.publish, db.health.port, db.port, dsn["port"]) == (
        "127.0.0.1:15432:5432",
        15432,
        15432,
        "15432",
    )
