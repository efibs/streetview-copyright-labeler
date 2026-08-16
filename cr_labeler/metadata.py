"""Coordinate to panorama-id lookup via the official Street View metadata API.

This is the only module that touches the API key, and it is only reached for
entries that carry no ``panoId``.  Google's undocumented internal endpoints
(``photometa``, ``SingleImageSearch``) were tested and now reject all requests,
so the official API is the one route that actually works.

Metadata requests are documented as free of charge; see ``config.NO_KEY_HELP``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from .config import ApiKey

log = logging.getLogger(__name__)

METADATA_URL = "https://maps.googleapis.com/maps/api/streetview/metadata"


class LookupFailed(RuntimeError):
    """Raised when a coordinate cannot be resolved to a panorama."""


@dataclass(frozen=True)
class PanoramaRef:
    """A resolved panorama."""

    pano_id: str
    date: str | None
    lat: float | None
    lng: float | None


def lookup(
    lat: float,
    lng: float,
    api_key: ApiKey,
    radius: int = 50,
    session: requests.Session | None = None,
    timeout: float = 20.0,
) -> PanoramaRef:
    """Resolve the newest panorama near ``(lat, lng)``.

    Google returns the default (newest) panorama for a location, which is what
    the input format implies when it gives coordinates without a ``panoId``.
    """
    http = session or requests.Session()
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "source": "outdoor",
        "key": api_key.value,
    }
    try:
        response = http.get(METADATA_URL, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise LookupFailed(f"metadata request failed for ({lat}, {lng}): {exc}") from exc

    if response.status_code != 200:
        raise LookupFailed(
            f"metadata request for ({lat}, {lng}) returned HTTP {response.status_code}"
        )

    payload = response.json()
    status = payload.get("status")
    if status == "ZERO_RESULTS":
        raise LookupFailed(f"no Street View coverage within {radius} m of ({lat}, {lng})")
    if status == "REQUEST_DENIED":
        # Never echo the key itself, only the reason.
        raise LookupFailed(
            "Google denied the metadata request -- check that the key is valid and "
            f"that the Street View Static API is enabled: {payload.get('error_message', '')}"
        )
    if status != "OK":
        raise LookupFailed(f"metadata lookup for ({lat}, {lng}) returned status {status}")

    pano_id = payload.get("pano_id")
    if not pano_id:
        raise LookupFailed(f"metadata response for ({lat}, {lng}) carried no pano_id")

    location = payload.get("location") or {}
    return PanoramaRef(
        pano_id=pano_id,
        date=payload.get("date"),
        lat=location.get("lat"),
        lng=location.get("lng"),
    )
