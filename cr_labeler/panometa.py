"""Panorama metadata from Google's internal ``photometa`` endpoint.

Gives two things the tile server cannot: the panorama's true dimensions, and
the ids of neighbouring panoramas.

Dimensions settle the camera generation, which Vali has no property for and
which capture year does not predict -- Australian coverage recorded as 2009 is
second generation, while genuine first-generation imagery sits at 3328x1664:

    3328 x 1664    generation 1   (2007-2009)
    13312 x 6656   generation 2/3
    16384 x 8192   generation 4

The endpoint also carries a ``"(c) YYYY Google"`` string, which is *not* the
watermark year and must not be used as one: it is the year the response was
generated.  Checked against ground truth it read 2026 for all twenty
panoramas, matching only the four whose true year happened to be 2026.  The
year this project reads is composited into the imagery itself.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import requests

log = logging.getLogger(__name__)

METADATA_URL = "https://www.google.com/maps/photometa/v1"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# The endpoint takes a protobuf-ish query string.  The trailing 9m36 block is
# what makes it return the full record; without it the response is a stub with
# no dimensions and no neighbours.
_PB = (
    "!1m4!1smaps_sv.tactile!11m2!2m1!1b1!2m2!1sen!2sus!3m3!1m2!1e2!2s{}"
    "!4m57!1e1!1e2!1e3!1e4!1e5!1e6!1e8!1e12!2m1!1e1!4m1!1i48!5m1!1e1!5m1!1e2!6m1!1e1!6m1!1e2"
    "!9m36!1m3!1e2!2b1!3e2!1m3!1e2!2b0!3e3!1m3!1e3!2b1!3e2!1m3!1e3!2b0!3e3!1m3!1e8!2b0!3e3"
    "!1m3!1e1!2b0!3e3!1m3!1e4!2b0!3e3!1m3!1e10!2b1!3e2!1m3!1e10!2b0!3e3"
)

_SIZE = re.compile(r"\[2,2,\[(\d+),(\d+)\]")
_PANO = re.compile(r'\[2,"([A-Za-z0-9_-]{22})"\]')
_DATE = re.compile(r"\[(\d{4}),(\d{1,2})\]\]")

GEN1_SIZE = (3328, 1664)


@dataclass(frozen=True)
class PanoInfo:
    """What the metadata endpoint knows about a panorama."""

    pano_id: str
    width: int | None
    height: int | None
    neighbours: tuple[str, ...]

    @property
    def generation(self) -> int | None:
        """Camera generation inferred from the panorama's true resolution."""
        if not self.width:
            return None
        if self.width <= 3328:
            return 1
        if self.width <= 13312:
            return 3  # generations 2 and 3 share this size; not separable here
        return 4

    @property
    def is_gen1(self) -> bool:
        return self.generation == 1


def fetch(pano_id: str, session: requests.Session | None = None, timeout: float = 25.0) -> PanoInfo | None:
    """Look up one panorama.  Returns ``None`` if the endpoint gives nothing."""
    http = session or requests.Session()
    http.headers.setdefault("User-Agent", USER_AGENT)
    try:
        response = http.get(
            METADATA_URL,
            params={"authuser": "0", "hl": "en", "pb": _PB.format(pano_id)},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.debug("metadata request failed for %s: %s", pano_id, exc)
        return None
    if response.status_code != 200:
        return None

    body = response.text
    size = _SIZE.search(body)
    height, width = (int(size.group(1)), int(size.group(2))) if size else (None, None)
    neighbours = tuple(dict.fromkeys(_PANO.findall(body)))
    return PanoInfo(pano_id=pano_id, width=width, height=height, neighbours=neighbours)
