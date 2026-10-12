"""Per-image template builds: naming, staging, reuse, and failure surfaces."""

import pytest

from sprout.harness.sandbox.templates import (RUNTIME_PATH, TemplateBuilder, TemplateError,
                                builder_from_backend, template_name)


IMAGE = "swebench/sweb.eval.x86_64.django__django-11848:latest"


def test_disk_shape_changes_identity_and_reaches_staging_and_template(monkeypatch, runtime_bin):
    resources = {"cpu": 2, "memory_mb": 2048, "disk_size_mb": 65536}
    assert template_name(IMAGE, "runtime", 3000, resources) != template_name(
        IMAGE, "runtime", 3000, {**resources, "disk_size_mb": 131072})
    payloads = []

    class CaptureClient(FakeClient):
        def post(self, path, json=None, **kwargs):
            if path in {"/sandboxes-cold", "/v3/templates"}:
                payloads.append((path, json))
            return super().post(path, json=json, **kwargs)

    client = CaptureClient()
    builder = TemplateBuilder("http://s", "key", runtime_bin)
    builder._stage_runtime(client, IMAGE, "template", resources)
    builder._build_from(client, "template", IMAGE, "staged-snap", resources)
    assert len(payloads) == 2
    assert all(body["diskSizeMB"] == 65536 for path, body in payloads)


def test_names_are_legal_stable_and_input_sensitive():
    first = template_name(IMAGE, "fp1", 3000)
    assert first == template_name(IMAGE, "fp1", 3000), "same inputs, same name"
    # Image names carry characters a template alias cannot.
    assert all(c.isalnum() or c == "-" for c in first), first
    assert len(first) <= 128

    # A different runtime binary or port must not reuse a template built with
    # the old one.
    assert first != template_name(IMAGE, "fp2", 3000)
    assert first != template_name(IMAGE, "fp1", 3001)
    assert first != template_name(IMAGE + "@v2", "fp1", 3000)


def test_image_env_changes_the_start_command_and_the_name(monkeypatch, runtime_bin):
    # Measured: the guest agent rebuilds PATH, dropping a venv's bin, so
    # `pytest` was "not found" in a verifier that the image's own PATH would
    # have satisfied. With image_env the runtime is launched under the image's
    # ENV -- and under a different template name, so a plain template built
    # earlier is never mistaken for one that carries the environment.
    import sprout.harness.sandbox.templates as templates
    env = ["PATH=/opt/venv/bin:/usr/bin:/bin", "LANG=C.UTF-8", "MSG=has space"]
    monkeypatch.setattr(templates, "image_config_env", lambda image, regctl, **kw: env)
    client = FakeClient(exists=False)
    b = builder(monkeypatch, client, runtime_bin)
    b.image_env, b.regctl_bin = True, runtime_bin      # any existing file will do
    name = b.template_for(IMAGE)

    payload = client.build_payload
    assert payload["startCmd"] == (
        "env PATH=/opt/venv/bin:/usr/bin:/bin LANG=C.UTF-8 'MSG=has space' "
        f"{RUNTIME_PATH} --port 3000")
    assert name != template_name(IMAGE, b._fingerprint, 3000)
    assert name == template_name(IMAGE, b._fingerprint, 3000,
                                 salt=templates.env_salt(env))
    # Off by default: the SWE-bench templates keep their names and their startCmd.
    plain = builder(monkeypatch, FakeClient(exists=False), runtime_bin)
    assert plain.template_for(IMAGE) == template_name(IMAGE, plain._fingerprint, 3000)


def test_image_env_without_regctl_is_refused_at_wiring(monkeypatch, runtime_bin):
    import sprout.harness.sandbox.templates as templates
    monkeypatch.setattr(templates, "find_regctl", lambda: None)
    monkeypatch.setattr(templates, "ensure_ripgrep", lambda: None)
    with pytest.raises(TemplateError, match="regctl"):
        builder_from_backend({"backend": "microvm",
                              "microvm": {"runtime_bin": str(runtime_bin),
                                          "server_url": "http://s",
                                          "image_env": True}})


class FakeResponse:
    def __init__(self, status_code, body=None, text=""):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self.text = text

    def json(self):
        return self._body


class FakeClient:
    """Scripts the calls a two-stage build makes, and records them."""

    def __init__(self, exists=False, statuses=("ready",), create_status=202,
                 create_text=""):
        self.exists = exists
        self.statuses = list(statuses)
        self.create_status = create_status
        self.create_text = create_text
        self.calls: list[tuple[str, str]] = []
        self.build_payload: dict | None = None
        self.uploaded: bytes | None = None
        self.deleted: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, path, **kwargs):
        self.calls.append(("GET", path))
        if path.startswith("/snapshots/"):
            # Sandbox-sourced snapshots only; a built template answers 404
            # here (it is template-sourced), which is exactly the trap the
            # builder must not fall into.
            return FakeResponse(404)
        if path.startswith("/templates/aliases/"):
            if "sprout-swebench-" not in path:
                return FakeResponse(404)
            return FakeResponse(200 if self.exists else 404)
        if path.endswith("/status"):
            status = self.statuses.pop(0) if self.statuses else "ready"
            return FakeResponse(200, {"status": status})
        raise AssertionError(path)

    def post(self, path, json=None, files=None, **kwargs):
        self.calls.append(("POST", path))
        if path == "/sandboxes-cold":
            return FakeResponse(201, {"sandboxID": "staging-sb"})
        if path == "/files":
            self.uploaded = files["file"][1].read()
            return FakeResponse(200, [{"path": RUNTIME_PATH}])
        if path.endswith("/snapshots"):
            assert json.get("diskOnly") is True, "staging pays for no memory image"
            assert "name" not in json, (
                "a staged alias would collide on retry after a failed build")
            return FakeResponse(201, {"snapshotID": "staged-snap"})
        if path == "/v3/templates":
            return FakeResponse(self.create_status,
                                {"templateID": "tid", "buildID": "bid"},
                                text=self.create_text)
        self.build_payload = json
        return FakeResponse(202)

    def delete(self, path, **kwargs):
        self.calls.append(("DELETE", path))
        self.deleted.append(path)
        return FakeResponse(204)


def builder(monkeypatch, client, runtime_bin):
    import sprout.harness.sandbox.templates as templates
    monkeypatch.setattr(templates.httpx, "Client", lambda **kw: client)
    monkeypatch.setattr(templates.time, "sleep", lambda _s: None)
    return TemplateBuilder(server_url="http://server", api_key="k",
                           runtime_bin=runtime_bin)


def test_missing_runtime_binary_fails_at_construction(tmp_path):
    with pytest.raises(TemplateError, match="runtime binary not found"):
        TemplateBuilder(server_url="http://server", api_key="k",
                        runtime_bin=tmp_path / "no-such-file")


def test_existing_template_is_reused_without_building(monkeypatch, runtime_bin):
    client = FakeClient(exists=True)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert [m for m, _ in client.calls] == ["GET"], (
        "an image reference cannot be a snapshot name, so only the template "
        "is looked up; no build was started")


def test_build_stages_the_runtime_then_declares_startup(monkeypatch, runtime_bin):
    client = FakeClient(exists=False)
    builder(monkeypatch, client, runtime_bin).template_for(IMAGE)

    # Stage 1: the binary went in through the file service, and the staging
    # sandbox was not leaked.
    assert client.uploaded == runtime_bin.read_bytes()
    assert client.deleted == ["/sandboxes/staging-sb"]

    # Stage 2: the committed template starts from the staged snapshot, knows
    # how to relaunch the runtime, and restores the executable bit the upload
    # dropped.
    payload = client.build_payload
    assert payload["fromTemplate"] == "staged-snap"
    assert payload["startCmd"] == f"{RUNTIME_PATH} --port 3000"
    # Cold boots re-run startCmd, so readiness must mean the port answers.
    assert "/dev/tcp/127.0.0.1/3000" in payload["readyCmd"]
    assert f"chmod +x {RUNTIME_PATH}" in payload["steps"][0]["args"][0]


def test_custom_port_changes_template_identity_startup_and_readiness(monkeypatch, runtime_bin):
    client = FakeClient(exists=False)
    configured = builder(monkeypatch, client, runtime_bin)
    configured.runtime_port = 34122
    name = configured.template_for(IMAGE)
    assert name != template_name(IMAGE, configured._fingerprint, 3000)
    assert client.build_payload["startCmd"] == f"{RUNTIME_PATH} --port 34122"
    assert "/dev/tcp/127.0.0.1/34122" in client.build_payload["readyCmd"]


def test_staging_sandbox_is_deleted_even_when_snapshotting_fails(
        monkeypatch, runtime_bin):
    class SnapshotFails(FakeClient):
        def post(self, path, json=None, files=None, **kwargs):
            if path.endswith("/snapshots"):
                self.calls.append(("POST", path))
                return FakeResponse(500, text="boom")
            return super().post(path, json=json, files=files, **kwargs)

    client = SnapshotFails(exists=False)
    with pytest.raises(TemplateError, match="could not snapshot"):
        builder(monkeypatch, client, runtime_bin).template_for(IMAGE)
    assert client.deleted == ["/sandboxes/staging-sb"], "no leaked VM"


def test_pending_build_is_waited_for(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=("building", "building", "ready"))
    builder(monkeypatch, client, runtime_bin).template_for(IMAGE)
    assert sum(1 for m, p in client.calls if p.endswith("/status")) == 3


def test_failed_build_is_reported(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=("error",))
    with pytest.raises(TemplateError, match="template build failed"):
        builder(monkeypatch, client, runtime_bin).template_for(IMAGE)


@pytest.mark.parametrize("create_status, create_text", [
    (409, ""),
    # The collision also arrives as a 400 whose text names the alias
    # conflict; a plain 400 must still raise, so the text match is load-bearing.
    (400, "alias 'x' already points to 'y', cannot rebind"),
])
def test_conflicting_create_defers_to_the_other_builder(
        monkeypatch, runtime_bin, create_status, create_text):
    # Two workers racing on the same image: the loser reuses the winner's.
    client = FakeClient(exists=False, create_status=create_status,
                        create_text=create_text)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert client.build_payload is None


def test_build_timeout_fails_rather_than_hangs(monkeypatch, runtime_bin):
    client = FakeClient(exists=False, statuses=tuple(["building"] * 50))
    b = builder(monkeypatch, client, runtime_bin)
    b.build_timeout = 0.0
    with pytest.raises(TemplateError, match="did not finish"):
        b.template_for(IMAGE)


# --- config wiring --------------------------------------------------------- #

def test_builder_only_with_a_runtime_bin(runtime_bin):
    assert builder_from_backend({}) is None
    assert builder_from_backend({"backend": "microvm", "microvm": {}}) is None

    built = builder_from_backend({"backend": "microvm", "microvm": {
        "server_url": "http://server/", "api_key": "k",
        "runtime_bin": str(runtime_bin), "runtime_port": 4000}})
    assert built is not None
    assert built.server_url == "http://server", "trailing slash trimmed"
    assert built.runtime_port == 4000


def test_runtime_bin_without_a_server_is_an_error(runtime_bin):
    with pytest.raises(TemplateError, match="server_url"):
        builder_from_backend({"backend": "microvm",
                              "microvm": {"runtime_bin": str(runtime_bin)}})


def test_a_name_the_backend_already_knows_is_used_as_is(monkeypatch, runtime_bin):
    """A replay hands `create()` a checkpoint snapshot id; building a template
    from it would try to cold-start a snapshot as an image and fail. The
    catalog is the authority: a known snapshot passes through untouched."""
    class KnowsSnapshot(FakeClient):
        def get(self, path, **kwargs):
            if path == "/snapshots/01a0-checkpoint":
                self.calls.append(("GET", path))
                return FakeResponse(200, {"snapshotID": "01a0-checkpoint"})
            return super().get(path, **kwargs)

    client = KnowsSnapshot(exists=False)
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for("01a0-checkpoint") == "01a0-checkpoint"
    assert client.build_payload is None, "no build for an existing snapshot"
    # And it is cached like any other resolution.
    before = len(client.calls)
    assert b.template_for("01a0-checkpoint") == "01a0-checkpoint"
    assert len(client.calls) == before


def test_a_built_template_is_found_even_though_snapshots_says_404(
        monkeypatch, runtime_bin):
    """The regression that shipped a double build: /snapshots/{name} answers
    only for sandbox-sourced snapshots, so a built (template-sourced) template
    looks missing there and the second run rebuilds into a name collision."""
    client = FakeClient(exists=True)     # exists=True now means: alias lookup hits
    b = builder(monkeypatch, client, runtime_bin)
    assert b.template_for(IMAGE) == template_name(IMAGE, b._fingerprint, 3000)
    assert client.build_payload is None, "no rebuild of an existing template"
    assert any(p.startswith("/templates/aliases/") for _, p in client.calls)
    # And it is cached like a snapshot id: a second request asks nothing.
    before = len(client.calls)
    b.template_for(IMAGE)
    assert len(client.calls) == before, "the resolved name is cached"


def test_ripgrep_is_staged_and_part_of_the_identity(monkeypatch, runtime_bin,
                                                    tmp_path):
    """rg rides along with the runtime so it is on every sandbox's PATH;
    provisioning it inside the sandbox instead cost ~89 MiB of disk writes in
    the episode's first checkpoint (measured). Baked-in content is identity, so
    a template built without rg is not reused once rg is available."""
    rg = tmp_path / "rg"
    rg.write_bytes(b"\x7fELF fake rg")

    client = FakeClient(exists=False)
    b_with = builder(monkeypatch, client, runtime_bin)
    b_with.ripgrep_bin = None  # rebuild fingerprints via a fresh instance
    b_with = TemplateBuilder(server_url="http://server", api_key="k",
                             runtime_bin=runtime_bin, ripgrep_bin=rg)
    import sprout.harness.sandbox.templates as templates
    monkeypatch.setattr(templates.httpx, "Client", lambda **kw: client)
    b_with.template_for(IMAGE)

    uploads = [p for m, p in client.calls if p == "/files"]
    assert len(uploads) == 2, "runtime and rg both staged"
    assert "chmod +x /usr/local/bin/rg" in client.build_payload["steps"][0]["args"][0]

    # Unlike the runtime, rg is optional: a path to a vanished binary (the
    # ~/.cache entry ensure_ripgrep returns can be cleaned) degrades to None
    # instead of failing every build, and names the without-rg template.
    b_without = TemplateBuilder(server_url="http://server", api_key="k",
                                runtime_bin=runtime_bin,
                                ripgrep_bin=tmp_path / "no-such-rg")
    assert b_without.ripgrep_bin is None, "a vanished cache entry must not fail builds"
    assert b_with._fingerprint != b_without._fingerprint, (
        "with/without rg are different templates")


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeClient:
    """Answers the two lookups a template check makes."""

    def __init__(self, aliases, statuses):
        self.aliases = aliases        # name -> template id
        self.statuses = statuses      # template id -> build status
        self.gets = []

    def get(self, path, **kwargs):
        self.gets.append(path)
        if path.startswith("/templates/aliases/"):
            name = path.rsplit("/", 1)[1]
            if name in self.aliases:
                return _Resp(200, {"templateID": self.aliases[name]})
            return _Resp(404)
        if path.startswith("/templates/") and path.endswith("/status"):
            tid = path.split("/")[2]
            return _Resp(200, {"status": self.statuses.get(tid, "ready")})
        raise AssertionError(path)


def bare_builder():
    """A TemplateBuilder with no I/O wiring: these tests exercise lookup
    logic, which must not need a server or a runtime binary."""
    from sprout.harness.sandbox.templates import TemplateBuilder
    return TemplateBuilder.__new__(TemplateBuilder)


def test_a_template_whose_build_failed_is_not_usable():
    """Existence was answering a weaker question than the caller asks: a
    failed build still leaves its alias resolvable, so every later run adopted
    a template it could not spawn from (HTTP 500, "snapshot ... is not
    ready"). One failed build took out all 20 tasks of a batch."""
    b = bare_builder()
    client = _FakeClient({"t": "id-broken"}, {"id-broken": "error"})
    assert not b._template_exists(client, "t")

    ok = _FakeClient({"t": "id-good"}, {"id-good": "ready"})
    assert b._template_exists(ok, "t")

    # Only an explicit failure disqualifies: an unrecognised or absent status
    # must not make a working template look broken, or every template built
    # before the status endpoint existed would be rebuilt under a new name.
    unknown = _FakeClient({"t": "id-odd"}, {"id-odd": "some-new-state"})
    assert b._template_exists(unknown, "t")


def test_a_poisoned_name_is_routed_around_not_retried():
    """Template aliases cannot be rebound -- re-creating one is refused with
    "cannot rebind" -- so a name owned by a failed build is permanently
    unusable and the only way forward is a different name."""
    b = bare_builder()
    built = []
    b._stage_runtime = lambda c, image, name, resources=None: built.append(name) or "snap"
    b._build_from = lambda c, name, image, staged, resources=None: None

    client = _FakeClient({"base": "id-broken"}, {"id-broken": "error"})
    name = b._usable_template(client, "base", "img", None)
    assert name == "base-r1", "the next name, not the poisoned one"
    assert built == ["base-r1"]


def test_the_search_for_a_name_is_bounded():
    """Needing many variants means something fails repeatably, and a loud
    error beats a longer search."""
    from sprout.harness.sandbox.templates import MAX_TEMPLATE_ATTEMPTS, TemplateError
    import pytest

    b = bare_builder()
    aliases = {"base": "x0", **{f"base-r{i}": f"x{i}"
                                for i in range(1, MAX_TEMPLATE_ATTEMPTS + 1)}}
    client = _FakeClient(aliases, {v: "error" for v in aliases.values()})
    with pytest.raises(TemplateError, match="no usable template name"):
        b._usable_template(client, "base", "img", None)
