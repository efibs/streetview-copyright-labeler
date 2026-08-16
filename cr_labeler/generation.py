"""Detect first-generation Street View panoramas.

Vali cannot filter on camera generation -- it has no such property, confirmed
against v1.0.0 -- and capture year is a poor stand-in: Australian coverage
recorded as 2009 is second generation, while genuine 2008 first-generation
imagery sits behind the time machine at locations Vali records under a later
year.

Resolution is the reliable signal.  First-generation panoramas are 3328x1664,
which is fully served by zoom 3; asking for zoom 4 returns a valid HTTP 200
carrying a *blank* tile of a fixed small size.  Later generations have real
content there.  One request settles it.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

from .fetch import TileFetcher

log = logging.getLogger(__name__)

# A blank filler tile is uniform; real imagery never is.  Measured: filler tiles
# come back at std 0.0 and ~1.2 kB, real tiles at std 10-50 and 11-77 kB.
BLANK_STD = 0.5
GEN1_MAX_ZOOM = 3


def tile_has_content(fetcher: TileFetcher, pano_id: str, zoom: int) -> bool:
    """Whether the tile server returns real imagery at ``zoom``.

    Samples a mid-panorama tile: the edges of the grid can legitimately be
    empty at the poles, but the middle cannot.
    """
    cols, rows = 2**zoom, 2 ** (zoom - 1)
    blob = fetcher._tile(pano_id, cols // 2, rows // 2, zoom)
    if not blob:
        return False
    try:
        image = Image.open(io.BytesIO(blob))
        image.load()
    except Exception:  # truncated or non-image payload
        return False
    return float(np.asarray(image.convert("L"), np.float32).std()) > BLANK_STD


def max_zoom(fetcher: TileFetcher, pano_id: str, ceiling: int = 5) -> int:
    """Highest zoom carrying real imagery, or 0 if the panorama is unavailable."""
    best = 0
    for zoom in range(1, ceiling + 1):
        if tile_has_content(fetcher, pano_id, zoom):
            best = zoom
        elif best:
            break  # content then blank: the resolution ceiling has been found
    return best


def is_gen1(fetcher: TileFetcher, pano_id: str) -> bool:
    """First generation: real content at zoom 3, none at zoom 4."""
    return tile_has_content(fetcher, pano_id, GEN1_MAX_ZOOM) and not tile_has_content(
        fetcher, pano_id, GEN1_MAX_ZOOM + 1
    )
