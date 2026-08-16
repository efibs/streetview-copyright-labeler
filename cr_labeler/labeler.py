"""End-to-end labelling: panorama in, year tag out."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from .bank import Bank
from .classify import NO_WATERMARK, UNKNOWN, Verdict, classify
from .composite import Template, build_composites
from .config import NO_KEY_HELP, ApiKey
from .fetch import PanoramaUnavailable, TileFetcher
from .geoguessr_io import Location
from .geometry import DEFAULT_ZOOM, DETECT_THRESHOLD
from .metadata import LookupFailed, lookup
from .signal import highpass

# When to spend a more expensive rung on an apparently-answered reading.  Both
# thresholds sit at the far tail of what correct readings look like: across 349
# zoom-2 readings the thinnest margin was 0.006 and the 1st percentile 0.015.
CONFIDENT_INSTANCES = 2
CONFIDENT_MARGIN = 0.02

log = logging.getLogger(__name__)


@dataclass
class LabelResult:
    """The outcome for one input row."""

    index: int
    pano_id: str | None
    label: str
    verdict: Verdict | None
    error: str | None = None
    composite: np.ndarray | None = None
    ground_truth: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Labeler:
    """Runs the pipeline over many locations, in parallel."""

    def __init__(
        self,
        bank: Bank,
        fetcher: TileFetcher,
        api_key: ApiKey | None = None,
        zoom: int = DEFAULT_ZOOM,
        rows: str = "top",
        threshold: float = DETECT_THRESHOLD,
        escalate: bool = True,
    ):
        self.bank = bank
        self.fetcher = fetcher
        self.api_key = api_key
        self.zoom = zoom
        self.rows = rows
        self.threshold = threshold
        self.escalate = escalate
        # Styles that share a rendering share an anchor -- refining a style by
        # layout splits its templates but leaves the wordmark identical. Detect
        # once per distinct anchor: classify() considers every style against
        # whatever composite it is given, so duplicates only cost time.
        distinct: dict[bytes, Template] = {}
        for style, anchor in sorted(bank.anchors.items()):
            distinct.setdefault(anchor.tobytes(), Template(array=anchor, style=style))
        self._anchors = list(distinct.values())
        self._meta_session = requests.Session()

    # ---- single panorama --------------------------------------------------

    def _analyse(self, pano_id: str, rows: str, zoom: int | None = None) -> tuple[Verdict, np.ndarray]:
        """Classify a panorama, letting the anchors compete on the final verdict.

        Each anchor produces its own composite and each composite is classified;
        the most watermark-like one wins.  Judging anchors any earlier -- on
        detection count or correlation mass -- lets a wrong-style anchor that
        merely fires on texture beat the one that actually found the watermark.
        """
        panorama = self.fetcher.fetch(pano_id, zoom=zoom or self.zoom, rows=rows)
        field = highpass(panorama.image)

        best: tuple[Verdict, np.ndarray] | None = None
        for result in build_composites(field, self._anchors, threshold=self.threshold):
            verdict = classify(result.composite, result.instances, self.bank)
            if best is None or verdict.quality > best[0].quality:
                best = (verdict, result.composite)

        assert best is not None, "build_composites returns one result per anchor"
        return best

    def _unsettled(self, verdict: Verdict) -> bool:
        """Whether a reading is worth spending a more expensive rung on.

        Beyond the obvious "found nothing", two readings look answered but are
        not.  A composite averaged from a *single* instance has nothing to have
        agreed with, so consensus never vetted it; and a near-tied margin means
        the runner-up year fit almost as well.  Both were caught in the field:
        one panorama read 2025 from one instance at margin 0.007, where every
        higher rung read 2026 from 9, 29 and 58 instances and the glyphs plainly
        say 2026.

        Neither test fires often -- a single instance is 8% of zoom-2 readings
        and a margin this thin is 2% -- and the rung is only kept if it scores
        better, so this can correct such a reading but never invent one.
        """
        return (
            verdict.label in (UNKNOWN, NO_WATERMARK)
            or verdict.instances < CONFIDENT_INSTANCES
            or verdict.margin < CONFIDENT_MARGIN
        )

    def _ladder(self) -> list[tuple[str, int]]:
        """The (rows, zoom) steps to try when a reading comes back weak.

        Starting below zoom 3 is a throughput trade: zoom 2 needs a quarter of
        the tiles, which is what the network actually costs, but it finds a
        median 11 instances against 28 and so reads confidently less often.
        The extra ``top`` rung at zoom 3 is what makes that safe -- anything
        zoom 2 could not settle is re-read at exactly the resolution the
        default uses, before the more expensive full-sphere rungs.

        At the default zoom this returns the same two steps it always has.
        """
        base = max(self.zoom, DEFAULT_ZOOM)
        steps = [("top", DEFAULT_ZOOM)] if self.zoom < DEFAULT_ZOOM else []
        steps.append(("all", base))
        steps.append(("all", base + 1))
        return steps

    def label_pano(self, pano_id: str) -> tuple[Verdict, np.ndarray]:
        """Classify one panorama, escalating to the full sphere if unsure."""
        verdict, composite = self._analyse(pano_id, self.rows)

        # Escalate a weak reading, in two steps, each kept only if it is more
        # watermark-like than what came before.
        #
        # Retrying on "None" as well as "unknown" matters: "no watermark found"
        # and "no watermark exists" are indistinguishable from one reading, and
        # on a heavily textured panorama the scene swamps the filter entirely.
        #
        # The second step raises the *zoom*.  The overlay is composited at a
        # fixed pixel size whatever the zoom, so zoom 4 covers four times the
        # area at the same glyph size and finds far more instances of it --
        # measured across 56 abstentions, mean instances went 10.3 -> 29.1 and
        # 44 of them resolved to the correct year.  Zoom 5 is worse, not better
        # (28 right but 12 wrong), and pooling z3 with z4 is worse than z4 alone
        # (39 vs 44) because the noisier z3 patches dilute the average.
        if not (self.escalate and self.rows == "top"):
            return verdict, composite

        for rows, zoom in self._ladder():
            if not self._unsettled(verdict):
                break
            log.debug("escalating %s to rows=%s zoom=%s", pano_id, rows, zoom)
            retry, retry_composite = self._analyse(pano_id, rows, zoom)
            if retry.quality > verdict.quality:
                verdict, composite = retry, retry_composite

        return verdict, composite

    # ---- batch ------------------------------------------------------------

    def _resolve(self, location: Location) -> str:
        if location.pano_id:
            return location.pano_id
        if location.lat is None or location.lng is None:
            raise ValueError(f"{location.describe()} has neither panoId nor coordinates")
        if not self.api_key:
            raise ValueError(f"{location.describe()}: {NO_KEY_HELP}")
        reference = lookup(location.lat, location.lng, self.api_key, session=self._meta_session)
        log.info("%s resolved to panoId %s", location.describe(), reference.pano_id)
        return reference.pano_id

    def label_one(self, location: Location) -> LabelResult:
        pano_id: str | None = None
        try:
            pano_id = self._resolve(location)
            verdict, composite = self.label_pano(pano_id)
            return LabelResult(
                index=location.index,
                pano_id=pano_id,
                label=verdict.label,
                verdict=verdict,
                composite=composite,
                ground_truth=location.ground_truth(),
            )
        except (PanoramaUnavailable, LookupFailed, ValueError) as exc:
            return LabelResult(
                index=location.index,
                pano_id=pano_id,
                label=UNKNOWN,
                verdict=None,
                error=str(exc),
                ground_truth=location.ground_truth(),
            )

    def label_all(
        self,
        locations: list[Location],
        workers: int = 8,
        progress=None,
    ) -> list[LabelResult]:
        results: list[LabelResult] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(self.label_one, locations):
                results.append(result)
                if progress:
                    progress(result)
        return sorted(results, key=lambda r: r.index)


def save_composite(composite: np.ndarray, path: Path, scale: int = 4) -> None:
    """Write a composite as a viewable PNG, contrast-stretched."""
    path.parent.mkdir(parents=True, exist_ok=True)
    spread = float(np.ptp(composite))
    if spread < 1e-9:
        image = Image.new("L", (composite.shape[1], composite.shape[0]), 0)
    else:
        normalised = (composite - composite.min()) / spread * 255.0
        image = Image.fromarray(normalised.astype(np.uint8))
    image.resize(
        (image.width * scale, image.height * scale), Image.LANCZOS
    ).save(path)
