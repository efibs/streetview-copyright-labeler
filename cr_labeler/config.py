"""Secure resolution of the optional Google Maps API key.

The key is only ever needed to turn coordinates into a panorama id.  Tile
fetching -- the entire image pipeline -- never uses it, so most runs need no key
at all.

Deliberately absent: a ``--api-key`` command line flag.  That would put the
secret into shell history, into ``ps`` output, and into any CI log that echoes
its command line.  The four sources below all keep it out of the process table.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

ENV_VAR = "GSV_API_KEY"
KEYRING_SERVICE = "cr_labeler"
KEYRING_USER = "google_maps_api_key"


class MissingApiKey(RuntimeError):
    """Raised when a coordinate lookup is needed but no key is configured."""


@dataclass(frozen=True)
class ApiKey:
    """A resolved key and where it came from.  Never log ``value``."""

    value: str
    source: str

    def __repr__(self) -> str:  # keeps the secret out of tracebacks and reprs
        return f"ApiKey(source={self.source!r}, value='***')"

    __str__ = __repr__


def _from_file(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def _from_dotenv(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == ENV_VAR:
            return value.strip().strip("'\"") or None
    return None


def _from_keyring() -> str | None:
    try:
        import keyring  # optional dependency
    except ImportError:
        return None
    try:
        return keyring.get_password(KEYRING_SERVICE, KEYRING_USER) or None
    except Exception as exc:  # a misconfigured backend must not break the run
        log.debug("keyring lookup failed: %s", exc)
        return None


def resolve_api_key(key_file: Path | None = None, dotenv: Path | None = None) -> ApiKey | None:
    """Find an API key, or return ``None`` if none is configured.

    Order: explicit ``--api-key-file``, then ``$GSV_API_KEY``, then ``.env`` in
    the working directory, then the OS keyring.
    """
    if key_file:
        value = _from_file(Path(key_file))
        if value:
            return ApiKey(value, f"--api-key-file {key_file}")
        raise MissingApiKey(f"--api-key-file {key_file} is missing or empty")

    value = os.environ.get(ENV_VAR, "").strip()
    if value:
        return ApiKey(value, f"${ENV_VAR}")

    candidate = Path(dotenv) if dotenv else Path.cwd() / ".env"
    value = _from_dotenv(candidate)
    if value:
        return ApiKey(value, str(candidate))

    value = _from_keyring()
    if value:
        return ApiKey(value, "OS keyring")

    return None


NO_KEY_HELP = f"""\
No panoId on this entry, and no Google Maps API key is configured, so the
panorama cannot be resolved from its coordinates.

Either add a "panoId" to the entry, or configure a key in one of:
  1. export {ENV_VAR}=...                (simplest; keep it out of shell history)
  2. a .env file beside your input       (add .env to .gitignore -- this repo does)
  3. --api-key-file /path/outside/repo   (file readable only by you: chmod 600)
  4. python -m keyring set {KEYRING_SERVICE} {KEYRING_USER}

Street View *metadata* requests are documented by Google as free of charge, but
they still require a billing-enabled project. Confirm against current pricing
before enabling: https://developers.google.com/maps/documentation/streetview/usage-and-billing
"""
