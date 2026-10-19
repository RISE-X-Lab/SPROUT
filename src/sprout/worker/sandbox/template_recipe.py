"""The recipe of a per-image microVM template, and the OCI config of an image.

A recipe is the files baked into a template, the command that starts the runtime, and a
content-addressed name for the recipe. The command can start the runtime under the image's OCI
``ENV``. The OCI config comes from the image's registry, read with regctl.
Why a template can need the image's ``ENV``: ``docs/RUNSTORE.md``, "Image USER and ENV".
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import httpx

#: Where the runtime binary lands in a template.
RUNTIME_PATH = "/usr/local/bin/sprout-runtime"
#: The port that the runtime listens on when the backend section sets no ``runtime_port``.
DEFAULT_RUNTIME_PORT = 3000

#: Where a template keeps its ripgrep, so that ``rg`` is on the PATH of each sandbox shell.
#: ``grep_files`` runs the copy embedded in the runtime, not this one.
RIPGREP_PATH = "/usr/local/bin/rg"
RIPGREP_VERSION = "14.1.1"
RIPGREP_URL = (
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    f"{RIPGREP_VERSION}/ripgrep-{RIPGREP_VERSION}-x86_64-unknown-linux-musl.tar.gz"
)


class TemplateError(RuntimeError):
    """A template cannot be built, or an image's OCI config cannot be read."""


def template_name(
    image: str, runtime_fingerprint: str, port: int, resources: dict[str, Any] | None = None, salt: str = ""
) -> str:
    """A stable, legal template name for one image, runtime, port, shape (``resources``) and ``salt``.

    The name is a hash: template names have a length limit and a narrow grammar, and image names have ``/``,
    ``:`` and ``.``. A template fixes a microVM's CPU and memory, so each shape gets its own name. ``salt`` is
    anything else that changes what the template does, such as its ENV. An empty ``salt`` adds nothing."""
    shape = ""
    if resources:
        shape = f"{resources.get('cpu', '')}x{resources.get('memory_mb', '')}"
        if "disk_size_mb" in resources:
            shape += f"x{resources['disk_size_mb']}"
    parts = [image, runtime_fingerprint, str(port), shape]
    if salt:
        parts.append(salt)
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]
    return f"sprout-swebench-{digest}"


# --- the image's own environment ---------------------------------------------

#: Where AgentENV keeps the copies of regctl that it downloads.
REGCTL_DEPS_GLOB = "/var/lib/aenv/deps/regctl/*/regctl"


def find_regctl() -> Path | None:
    """This host's regctl: ``$SPROUT_REGCTL`` if it is a file, else regctl on PATH, else AgentENV's copy.

    None when there is none. A host that runs microVM sandboxes has AgentENV's copy, because AgentENV resolves
    images with it. That copy has the server's registry logins. Docker Hub, public ECR and ghcr need none."""
    import glob
    import os
    import shutil

    candidate = os.environ.get("SPROUT_REGCTL")
    if candidate and Path(candidate).is_file():
        return Path(candidate)
    found = shutil.which("regctl")
    if found:
        return Path(found)
    for path in sorted(glob.glob(REGCTL_DEPS_GLOB), reverse=True):
        if Path(path).is_file():
            return Path(path)
    return None


def image_config_env(image: str, regctl: Path, timeout: float = 120.0) -> list[str]:
    """The ``KEY=VALUE`` entries of the ``Env`` in the image's OCI config, in order.

    Raises ``TemplateError`` when ``regctl`` cannot read the config within ``timeout`` seconds.
    """
    env = _image_config(image, regctl, timeout).get("Env") or []
    return [str(e) for e in env if "=" in str(e)]


def registry_config(image: str | None) -> dict[str, Any]:
    """The OCI config of ``image`` in its registry, such as its ``Env``, ``User`` and ``WorkingDir``.

    This host's regctl (``find_regctl``) reads it. Raises ``TemplateError`` when ``image`` is None or empty,
    when no regctl is found, or when regctl cannot read the config.
    """
    if not image:
        raise TemplateError("no OCI image reference names the image whose config is to be read")
    regctl = find_regctl()
    if regctl is None:
        raise TemplateError(
            f"no regctl was found (SPROUT_REGCTL, PATH, or {REGCTL_DEPS_GLOB}); "
            f"the config of {image} cannot be read"
        )
    return _image_config(image, regctl)


def _image_config(image: str, regctl: Path, timeout: float = 120.0) -> dict[str, Any]:
    """The ``Config`` object of the image's OCI config, read with ``regctl``."""
    import json
    import subprocess

    try:
        proc = subprocess.run(
            [str(regctl), "image", "config", image, "--format", "{{json .Config}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TemplateError(f"could not read {image}'s config with regctl: {exc}") from exc
    if proc.returncode != 0:
        raise TemplateError(f"regctl could not read {image}'s config: {proc.stderr.strip()[:300]}")
    try:
        config = json.loads(proc.stdout.strip() or "null") or {}
    except ValueError as exc:
        raise TemplateError(f"regctl returned no JSON for {image}: {proc.stdout[:200]}") from exc
    if not isinstance(config, dict):
        raise TemplateError(f"regctl returned no config object for {image}: {proc.stdout[:200]}")
    return config


def start_command(port: int, env: list[str] | None = None) -> str:
    """The template's ``startCmd``: the runtime on ``port``, under the ``KEY=VALUE`` entries of ``env``.

    The entries apply to the runtime process, so every tool call gets them. Only the listed keys change. The
    variables that the guest agent adds, such as its ``E2B_*`` variables and ``HOME``, stay.
    """
    import shlex

    base = f"{RUNTIME_PATH} --port {port}"
    if not env:
        return base
    return "env " + " ".join(shlex.quote(item) for item in env) + " " + base


def env_salt(env: list[str]) -> str:
    """The ``salt`` of ``template_name`` for a template whose runtime starts under ``env``."""
    return "env:" + hashlib.sha256("\0".join(env).encode()).hexdigest()[:12]


def runtime_fingerprint(runtime_bin: Path) -> str:
    """A short content hash of the binary ``runtime_bin``, so that a rebuilt binary gives a new name."""
    digest = hashlib.sha256()
    with open(runtime_bin, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def ensure_ripgrep(cache_dir: Path | None = None) -> Path | None:
    """A ripgrep binary for templates, downloaded once into ``cache_dir`` (``~/.cache/sprout-swebench``).

    Returns ``None``, and prints a warning, when it cannot be fetched. Templates then still build, but without
    ``rg`` on the sandbox PATH, so their environment differs from one built with it.
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sprout-swebench"
    binary = cache_dir / f"rg-{RIPGREP_VERSION}"
    if binary.is_file():
        return binary
    import io
    import tarfile

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            payload = client.get(RIPGREP_URL)
            payload.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(payload.content), mode="r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/rg") and m.isfile())
            extracted = tar.extractfile(member)
            assert extracted is not None
            tmp = binary.with_suffix(".tmp")
            tmp.write_bytes(extracted.read())
            tmp.chmod(0o755)
            tmp.rename(binary)
        return binary
    except Exception as exc:  # noqa: BLE001 -- optional asset, never fatal
        print(
            f"note: could not fetch ripgrep to bake into templates ({exc}); "
            "sandboxes built from these templates have no rg on PATH"
        )
        return None
