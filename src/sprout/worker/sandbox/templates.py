"""MicroVM templates built on demand from a benchmark's per-instance images.

``MicroVMPool.spawn`` starts from a template or a snapshot. Only ``spawn_from_image`` accepts
an OCI reference, and a cold start from a plain image does not bring up the runtime, because
the backend runs no startup command for it. So each raw per-instance image, such as each
SWE-bench image, needs a template whose startup command launches the runtime.

:class:`TemplateBuilder` returns the name of such a template for an image, and builds the
template on the first request. The name is a hash of the image, the runtime binary, the port,
the shape and the start environment. So a second run of an instance reuses the template, and
a changed runtime binary gets a new one.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from sprout.worker.sandbox.template_recipe import (
    DEFAULT_RUNTIME_PORT,
    RIPGREP_PATH,
    RUNTIME_PATH,
    TemplateError,
    env_salt,
    image_config_env,
    runtime_fingerprint,
    start_command,
    template_name,
)
from sprout_sandbox import MicroVMPool

#: How many suffixed template names to try before giving up. Each costs a lookup, and a need for
#: more than a couple means a build fails repeatably, which a louder error reports better.
MAX_TEMPLATE_ATTEMPTS = 6

#: Poll interval and ceiling of a build. A cold image pull and its conversion take most of a
#: build. The ceiling makes a stuck build fail the run instead of hanging it.
BUILD_POLL_SECONDS = 3.0
BUILD_TIMEOUT_SECONDS = 1800.0
COLD_START_TIMEOUT_SECONDS = 3600


def _could_be_snapshot_name(name: str) -> bool:
    """Whether ``name`` fits the backend's grammar of a snapshot id or alias.

    The grammar is ASCII letters, digits, hyphens and underscores. A name with ``/``, ``:`` or ``@``
    skips the catalog lookup, which would answer it with an error, not a 404. A bare name such as
    ``ubuntu`` still goes to the catalog, misses, and gets a template built."""
    return bool(name) and all(c.isalnum() or c in "-_" for c in name)


@dataclass
class TemplateBuilder:
    """Builds and reuses the microVM templates of benchmark images.

    An instance caches each name that it resolved, so a batch builds each distinct image once. A
    template that already exists under the name is reused, not rebuilt.
    """

    server_url: str
    api_key: str
    runtime_bin: Path
    #: A ripgrep binary to install next to the runtime, for ``rg`` in the sandbox shell. None
    #: leaves the shell without ``rg``. The ``grep_files`` tool does not need it.
    ripgrep_bin: Path | None = None
    runtime_port: int = DEFAULT_RUNTIME_PORT
    request_timeout: float = 120.0
    build_timeout: float = BUILD_TIMEOUT_SECONDS
    #: Launch the runtime under the image's OCI ENV (read with ``regctl_bin``).
    #: Changes the template's name, so it never collides with a plain build.
    image_env: bool = False
    regctl_bin: Path | None = None
    _resolved: dict[str, str] = field(default_factory=dict)
    _env_cache: dict[str, list[str]] = field(default_factory=dict)
    _fingerprint: str = ""

    def _env_for(self, image: str) -> list[str] | None:
        """The image's ENV when ``image_env`` is on and ``image`` is an image reference, else None.

        Raises TemplateError if ``image_env`` is on and no regctl binary was given."""
        if not self.image_env or _could_be_snapshot_name(image):
            return None
        if image not in self._env_cache:
            if self.regctl_bin is None:
                raise TemplateError("image_env requested but no regctl binary given")
            self._env_cache[image] = image_config_env(image, self.regctl_bin)
        return self._env_cache[image]

    def __post_init__(self) -> None:
        self.runtime_bin = Path(self.runtime_bin)
        if not self.runtime_bin.is_file():
            raise TemplateError(
                f"runtime binary not found at {self.runtime_bin}; "
                "build it (cd runtime && go generate ./... && go build -o sprout-runtime .) or point "
                "microvm.runtime_bin at one"
            )
        self._fingerprint = runtime_fingerprint(self.runtime_bin)
        if self.ripgrep_bin is not None:
            self.ripgrep_bin = Path(self.ripgrep_bin)
            if not self.ripgrep_bin.is_file():
                self.ripgrep_bin = None
        # Baked-in content is part of the template's identity: a template
        # without rg must not be reused once rg is available.
        if self.ripgrep_bin is not None:
            self._fingerprint += ":" + runtime_fingerprint(self.ripgrep_bin)

    def template_for(self, image: str, resources: dict[str, Any] | None = None) -> str:
        """The name of a template for ``image``, built on the first request.

        A name that the backend already knows, such as a checkpoint snapshot or an earlier template,
        comes back as it is. Only the catalog says what is a snapshot, not the string's shape. Each
        other name is an image reference, and gets a template built for it."""
        if image in self._resolved:
            return self._resolved[image]

        env = self._env_for(image)
        base = template_name(
            image, self._fingerprint, self.runtime_port, resources, salt=env_salt(env) if env else ""
        )
        name = base
        with self._client() as client:
            if _could_be_snapshot_name(image) and self._known(client, image):
                # A snapshot keeps the shape that it was captured with, and a resumed run continues
                # in the machine that it ran on. So ``resources`` does not apply here.
                name = image
            else:
                name = self._usable_template(client, base, image, resources)
        self._resolved[image] = name
        return name

    # --- internals ---

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.server_url, headers={"X-API-Key": self.api_key}, timeout=self.request_timeout
        )

    def _known(self, client: httpx.Client, name: str) -> bool:
        """Whether the backend already knows ``name`` as something launchable.

        Sandbox-sourced snapshots (a replay's checkpoint id) and built
        templates live behind different lookup endpoints, so both are asked.
        """
        return self._lookup(client, f"/snapshots/{name}", name) or self._template_exists(client, name)

    def _usable_template(
        self, client: httpx.Client, base: str, image: str, resources: dict[str, Any] | None
    ) -> str:
        """The name of a usable template for ``image``: ``base``, else its first free suffixed name.

        A failed build leaves its alias bound to a template that never spawns, and an alias cannot be
        rebound. Only a new name avoids it. Raises TemplateError after :data:`MAX_TEMPLATE_ATTEMPTS`."""
        for attempt in range(MAX_TEMPLATE_ATTEMPTS):
            name = base if attempt == 0 else f"{base}-r{attempt}"
            if self._template_exists(client, name):
                if attempt:
                    logging.getLogger(__name__).warning("template %s was unusable; using %s", base, name)
                return name
            if not self._name_taken(client, name):
                # A template step cannot copy a local file, and an isolated guest cannot download
                # one. So the runtime goes into a staged snapshot, and the template builds from it.
                staged = self._stage_runtime(client, image, name, resources)
                self._build_from(client, name, image, staged, resources)
                return name
            # Taken but unusable: a failed build owns this name for good.
            logging.getLogger(__name__).warning(
                "template %s exists but its build failed; trying the next name", name
            )
        raise TemplateError(
            f"no usable template name for {image}: {MAX_TEMPLATE_ATTEMPTS} "
            f"variants of {base} are taken by failed builds"
        )

    def _name_taken(self, client: httpx.Client, name: str) -> bool:
        """Whether the alias resolves at all, usable or not."""
        return self._lookup(client, f"/templates/aliases/{name}", name)

    def _template_exists(self, client: httpx.Client, name: str) -> bool:
        """Whether a template of this name exists and its build is not known to have failed.

        The alias endpoint, because ``/snapshots/{name}`` answers only for sandbox-sourced snapshots.
        An alias also resolves after a failed build, so the build status is read too."""
        resp = client.get(f"/templates/aliases/{name}")
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise TemplateError(f"could not check {name}: HTTP {resp.status_code} {resp.text[:200]}")
        template_id = (resp.json() or {}).get("templateID")
        if not template_id:
            # The alias resolves but does not say to what. That is no evidence of a failed build,
            # as in _build_succeeded below.
            return True
        return self._build_succeeded(client, template_id)

    def _build_succeeded(self, client: httpx.Client, template_id: str) -> bool:
        """Whether the template's build is not known to have failed.

        Only an explicit failure status disqualifies a template. A missing status, an unknown state
        or a missing endpoint does not: a bad template then fails at spawn, and names do not multiply."""
        try:
            resp = client.get(f"/templates/{template_id}/builds/{template_id}/status")
        except Exception:
            return True
        if resp.status_code != 200:
            return True
        status = str((resp.json() or {}).get("status") or "").lower()
        return status not in ("error", "failed", "failure", "cancelled", "canceled")

    def _lookup(self, client: httpx.Client, path: str, name: str) -> bool:
        """True for HTTP 200 at ``path``, False for 404. Raises TemplateError for another status."""
        resp = client.get(path)
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        raise TemplateError(f"could not check {name}: HTTP {resp.status_code} {resp.text[:200]}")

    def _stage_runtime(
        self, client: httpx.Client, image: str, name: str, resources: dict[str, Any] | None = None
    ) -> str:
        """Cold-start ``image``, install the runtime, snapshot the sandbox, and return the snapshot's id.

        The binary goes in through the backend's file service, not through the runtime, which is
        not installed yet."""
        # Only a cold start can choose a shape. The template built from this snapshot inherits it,
        # and so does every sandbox started from that template.
        payload: dict[str, Any] = {
            "image": image,
            "timeout": COLD_START_TIMEOUT_SECONDS,
            "autoPause": False,
        }
        if resources:
            if resources.get("cpu"):
                payload["cpuCount"] = int(resources["cpu"])
            if resources.get("memory_mb"):
                payload["memoryMB"] = int(resources["memory_mb"])
            if resources.get("disk_size_mb"):
                payload["diskSizeMB"] = int(resources["disk_size_mb"])
        created = client.post(
            "/sandboxes-cold", json=payload, timeout=max(self.request_timeout, COLD_START_TIMEOUT_SECONDS)
        )
        if created.status_code != 201:
            raise TemplateError(
                f"could not cold-start {image}: HTTP {created.status_code} {created.text[:200]}"
            )
        sandbox_id = created.json()["sandboxID"]
        try:
            self._upload(client, sandbox_id, self.runtime_bin, RUNTIME_PATH)
            if self.ripgrep_bin is not None:
                self._upload(client, sandbox_id, self.ripgrep_bin, RIPGREP_PATH)
            # Unnamed on purpose: the build uses it by id. An alias would make a retry after a
            # half-failed build collide with the leftover of the earlier attempt.
            snapshot = client.post(f"/sandboxes/{sandbox_id}/snapshots", json={"diskOnly": True})
            if snapshot.status_code != 201:
                raise TemplateError(
                    f"could not snapshot the staged runtime for {image}: "
                    f"HTTP {snapshot.status_code} {snapshot.text[:200]}"
                )
            snapshot_id: str = snapshot.json()["snapshotID"]
            return snapshot_id
        finally:
            # The staging sandbox is not needed on either path. Left running, it would hold a VM
            # for the rest of the run.
            client.delete(f"/sandboxes/{sandbox_id}")

    def _upload(self, client: httpx.Client, sandbox_id: str, source: Path, dest: str) -> None:
        # The backend's file service listens on its own port inside the guest,
        # reached through the same proxy as the runtime.
        headers = {
            MicroVMPool.SANDBOX_ID_HEADER: sandbox_id,
            MicroVMPool.TARGET_PORT_HEADER: MicroVMPool.ENVD_PORT,
        }
        with open(source, "rb") as handle:
            resp = client.post(
                "/files",
                params={"path": dest},
                headers=headers,
                files={"file": (source.name, handle, "application/octet-stream")},
                timeout=max(self.request_timeout, 300.0),
            )
        if resp.status_code not in (200, 201, 204):
            raise TemplateError(
                f"could not upload {source.name} to {sandbox_id}: HTTP {resp.status_code} {resp.text[:200]}"
            )

    def _build_from(
        self,
        client: httpx.Client,
        name: str,
        image: str,
        staged_snapshot: str,
        resources: dict[str, Any] | None = None,
    ) -> None:
        """Create the template ``name`` from ``staged_snapshot``, with the runtime's start command.

        Returns when the build is ready, or when another build of ``name`` exists already."""
        # The template must declare the staged snapshot's shape: the backend refuses a snapshot-based
        # build that changes it ("snapshot-based build cannot change CPU or memory").
        payload: dict[str, Any] = {"name": name}
        if resources:
            if resources.get("cpu"):
                payload["cpuCount"] = int(resources["cpu"])
            if resources.get("memory_mb"):
                payload["memoryMB"] = int(resources["memory_mb"])
            if resources.get("disk_size_mb"):
                payload["diskSizeMB"] = int(resources["disk_size_mb"])
        created = client.post("/v3/templates", json=payload)
        if created.status_code == 409 or (created.status_code == 400 and "already points" in created.text):
            # Another worker or an earlier attempt created it first. Its build is the one to use.
            return
        if created.status_code != 202:
            raise TemplateError(
                f"could not create template {name} for {image}: "
                f"HTTP {created.status_code} {created.text[:200]}"
            )
        body = created.json()
        template_id, build_id = body["templateID"], body["buildID"]

        started = client.post(
            f"/v2/templates/{template_id}/builds/{build_id}",
            json={
                "fromTemplate": staged_snapshot,
                # One step, to make the uploaded binary executable: uploads do
                # not carry the mode bit.
                "steps": [
                    {
                        "type": "RUN",
                        "args": [f"chmod +x {RUNTIME_PATH}; chmod +x {RIPGREP_PATH} 2>/dev/null || true"],
                    }
                ],
                "startCmd": start_command(self.runtime_port, self._env_for(image)),
                # Cold boots re-run startCmd, so readiness has to mean "the
                # runtime is accepting connections", not "the process exists".
                "readyCmd": f"timeout 1 bash -c '</dev/tcp/127.0.0.1/{self.runtime_port}'",
            },
        )
        if started.status_code not in (200, 202, 204):
            raise TemplateError(
                f"could not start build for {name} ({image}): HTTP {started.status_code} {started.text[:200]}"
            )

        self._await_build(client, template_id, build_id, name, image)

    def _await_build(
        self, client: httpx.Client, template_id: str, build_id: str, name: str, image: str
    ) -> None:
        """Poll the build until it is ready. Raises TemplateError if it fails or exceeds ``build_timeout``."""
        deadline = time.monotonic() + self.build_timeout
        last = ""
        while time.monotonic() < deadline:
            resp = client.get(f"/templates/{template_id}/builds/{build_id}/status")
            if resp.status_code != 200:
                raise TemplateError(
                    f"could not read build status for {name}: HTTP {resp.status_code} {resp.text[:200]}"
                )
            status = str(resp.json().get("status") or "")
            last = status
            if status == "ready":
                return
            if status in ("error", "failed"):
                raise TemplateError(f"template build failed for {name} ({image}): {resp.text[:400]}")
            time.sleep(BUILD_POLL_SECONDS)
        raise TemplateError(
            f"template build for {name} ({image}) did not finish in "
            f"{self.build_timeout:.0f}s (last status: {last or 'unknown'})"
        )
