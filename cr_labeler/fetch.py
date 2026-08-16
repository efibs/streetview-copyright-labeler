"""Panorama tile retrieval and stitching.

Tiles come from Google's public Street View tile endpoint.  It needs no API key
-- only a browser ``User-Agent`` -- but it does refuse requests that lack one,
which is why the header is set unconditionally.
"""

from __future__ import annotations

import errno
import hashlib
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import requests
from PIL import Image

from .geometry import DEFAULT_ZOOM, TILE_PX, pano_grid

log = logging.getLogger(__name__)

TILE_URL = "https://streetviewpixels-pa.googleapis.com/v1/tile"
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Pillow refuses very large images by default; a zoom-5 panorama is 134 MPx and
# is a file we constructed ourselves, so the guard is not meaningful here.
Image.MAX_IMAGE_PIXELS = None


class PanoramaUnavailable(RuntimeError):
    """Raised when too few tiles could be retrieved to analyse a panorama."""


@dataclass
class Panorama:
    """A stitched equirectangular panorama and how complete it is."""

    pano_id: str
    image: Image.Image
    zoom: int
    tiles_ok: int
    tiles_total: int

    @property
    def completeness(self) -> float:
        return self.tiles_ok / max(self.tiles_total, 1)


class TileCache:
    """Content-addressed on-disk tile cache.

    Optional, but makes re-runs over the same panorama list effectively free,
    which matters while tuning thresholds.
    """

    def __init__(self, root: Path | None):
        self.root = Path(root) if root else None
        self.writable = self.root is not None
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, pano_id: str, x: int, y: int, zoom: int) -> Path:
        assert self.root is not None
        digest = hashlib.sha1(f"{pano_id}:{zoom}:{x}:{y}".encode()).hexdigest()
        return self.root / digest[:2] / f"{digest}.jpg"

    def get(self, pano_id: str, x: int, y: int, zoom: int) -> bytes | None:
        if not self.root:
            return None
        path = self._path(pano_id, x, y, zoom)
        try:
            return path.read_bytes()
        except OSError:
            return None

    def put(self, pano_id: str, x: int, y: int, zoom: int, blob: bytes) -> None:
        if not self.root or not self.writable:
            return
        path = self._path(pano_id, x, y, zoom)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        except OSError as exc:  # a broken cache must never break a run
            if exc.errno == errno.ENOSPC:
                # Every subsequent tile would fail the same way and say so.
                # Stop writing, keep reading what is already there, and let the
                # run continue -- a full disk should cost the cache, not the
                # twenty hours of work behind it.
                self.writable = False
                log.warning(
                    "disk full at %s -- tile caching disabled for the rest of "
                    "this run; labelling continues",
                    self.root,
                )
            else:
                log.debug("tile cache write failed: %s", exc)


class TileFetcher:
    """Fetches and stitches panoramas, with retry and optional caching."""

    def __init__(
        self,
        cache: TileCache | None = None,
        tile_workers: int = 32,
        retries: int = 3,
        timeout: float = 20.0,
    ):
        self.cache = cache or TileCache(None)
        self.tile_workers = tile_workers
        self.retries = retries
        self.timeout = timeout
        self._local = threading.local()
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    @property
    def pool(self) -> ThreadPoolExecutor:
        """One tile pool for the whole run, not one per panorama.

        Building a pool per call cost more than the work it did: a zoom-2 band
        is four tiles, and spawning threads for them dominated the fetch.  A
        shared pool also bounds how many requests are in flight across every
        worker at once, which is the number Google throttles on.
        """
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(
                        max_workers=self.tile_workers, thread_name_prefix="tile"
                    )
        return self._pool

    def close(self) -> None:
        with self._pool_lock:
            if self._pool is not None:
                self._pool.shutdown(wait=False)
                self._pool = None

    @property
    def _session(self) -> requests.Session:
        # One session per thread: requests.Session is not thread-safe, but a
        # per-thread session still gets us connection pooling.
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update({"User-Agent": USER_AGENT})
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8, pool_maxsize=64, max_retries=0
            )
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _tile(self, pano_id: str, x: int, y: int, zoom: int) -> bytes | None:
        cached = self.cache.get(pano_id, x, y, zoom)
        if cached is not None:
            return cached

        params = {
            "cb_client": "maps_sv.tactile",
            "panoid": pano_id,
            "x": x,
            "y": y,
            "zoom": zoom,
        }
        for attempt in range(self.retries):
            try:
                response = self._session.get(
                    TILE_URL, params=params, timeout=self.timeout
                )
            except requests.RequestException as exc:
                log.debug("tile %s (%d,%d) request failed: %s", pano_id, x, y, exc)
            else:
                if response.status_code == 200:
                    self.cache.put(pano_id, x, y, zoom, response.content)
                    return response.content
                # 400/404 mean this tile does not exist -- panoramas are ragged
                # at the poles -- so there is nothing to retry.
                if response.status_code in (400, 404):
                    return None
                log.debug(
                    "tile %s (%d,%d) http %d", pano_id, x, y, response.status_code
                )
            time.sleep(0.25 * (2**attempt))
        return None

    def _tile_image(self, pano_id: str, x: int, y: int, zoom: int) -> Image.Image | None:
        """Retrieve one tile and decode it, both inside the worker thread.

        Decoding used to happen serially while stitching; a band of tiles is
        several megapixels of JPEG, and Pillow drops the GIL to decode, so it
        parallelises for free by doing it here.
        """
        blob = self._tile(pano_id, x, y, zoom)
        if not blob:
            return None
        try:
            tile = Image.open(io.BytesIO(blob))
            tile.load()
        except Exception as exc:  # truncated or non-image payload
            log.debug("tile %s (%d,%d) decode failed: %s", pano_id, x, y, exc)
            return None
        return tile

    def fetch(
        self,
        pano_id: str,
        zoom: int = DEFAULT_ZOOM,
        rows: str = "top",
        min_completeness: float = 0.3,
    ) -> Panorama:
        """Fetch and stitch a panorama.

        ``rows`` selects which band of tile rows to download.  ``top`` keeps the
        upper half, which holds ~96% of watermark instances and halves both
        download and correlation cost.  ``all`` downloads everything.
        """
        cols, total_rows = pano_grid(zoom)
        if rows == "top":
            row_range = range(0, max(1, total_rows // 2))
        elif rows == "all":
            row_range = range(total_rows)
        else:
            raise ValueError(f"unknown rows mode: {rows!r}")

        wanted = [(x, y) for x in range(cols) for y in row_range]
        canvas = Image.new("RGB", (cols * TILE_PX, len(row_range) * TILE_PX))
        offset = row_range.start
        ok = 0

        futures = {
            self.pool.submit(self._tile_image, pano_id, x, y, zoom): (x, y)
            for x, y in wanted
        }
        for future in futures:
            x, y = futures[future]
            tile = future.result()
            if tile is None:
                continue
            canvas.paste(tile, (x * TILE_PX, (y - offset) * TILE_PX))
            ok += 1

        pano = Panorama(
            pano_id=pano_id,
            image=canvas,
            zoom=zoom,
            tiles_ok=ok,
            tiles_total=len(wanted),
        )
        if pano.completeness < min_completeness:
            raise PanoramaUnavailable(
                f"only {ok}/{len(wanted)} tiles retrieved for panorama {pano_id!r} "
                f"-- it may not exist, or may not be a Street View panorama"
            )
        return pano
