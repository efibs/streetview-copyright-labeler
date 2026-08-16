"""Building a template bank from panoramas.

Two entry points:

* :func:`build_seed_bank` -- from labelled panoramas (``GT_*`` tags).  This is
  how the shipped bank was made and how you extend it with newly labelled data.
* :func:`harvest` -- from *unlabelled* panorama ids.  Composites are clustered,
  and each cluster is one (style, year).  You label the handful of cluster
  centroids, not the thousands of panoramas.

Neither trains anything.  A "template" is the mean of real composites.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .bank import Bank, average_composites, cluster_composites, derive_digit_slots
from .composite import Template, build_composite
from .discover import (
    SEED_WORD,
    BootstrapFailed,
    cut_anchor,
    find_seed_font,
    invariant_window,
    seed_composite,
)
from .fetch import PanoramaUnavailable, TileFetcher
from .geometry import ANCHOR_CX, ANCHOR_CY, DEFAULT_ZOOM, DETECT_THRESHOLD, PATCH_W
from .signal import best_alignment, declutter, highpass, normalise

log = logging.getLogger(__name__)

# Minimum gain before the year range is declared to span two render styles
# rather than one.  A genuine style change separates far more strongly than any
# boundary drawn through a single consistent rendering.
STYLE_SPLIT_GAIN = 0.10
# Within a style, the same year clusters at ~0.87 and different years at ~0.61.
YEAR_CUT = 0.78
# How far left of the wordmark the "(c) YYYY " block reaches, in pixels.  Wide
# enough to cover both renderings without spilling into the wordmark itself.
YEAR_SPAN = 62


@dataclass
class Sample:
    """One analysed panorama on the way into a bank."""

    pano_id: str
    composite: np.ndarray
    instances: int
    label: str | None = None
    style: str | None = None


@dataclass
class HarvestReport:
    """What came out of a harvest, for the operator to label."""

    clusters: list[list[Sample]] = field(default_factory=list)
    failures: list[tuple[str, str]] = field(default_factory=list)


def _fetch_field(fetcher: TileFetcher, pano_id: str, zoom: int, rows: str):
    panorama = fetcher.fetch(pano_id, zoom=zoom, rows=rows)
    return highpass(panorama.image)


def discover_samples(
    pano_ids: list[str],
    fetcher: TileFetcher,
    seed_font,
    zoom: int = DEFAULT_ZOOM,
    rows: str = "top",
    workers: int = 8,
) -> tuple[list[Sample], list[tuple[str, str]]]:
    """Produce a first composite per panorama using the synthetic seed."""
    samples: list[Sample] = []
    failures: list[tuple[str, str]] = []

    def one(pano_id: str) -> Sample | tuple[str, str]:
        try:
            field_ = _fetch_field(fetcher, pano_id, zoom, rows)
        except PanoramaUnavailable as exc:
            return (pano_id, str(exc))
        found = seed_composite(field_, seed_font)
        if not found:
            return (pano_id, "synthetic seed found no watermark instances")
        composite, count = found
        return Sample(pano_id=pano_id, composite=composite, instances=count)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(one, pano_ids):
            if isinstance(outcome, Sample):
                samples.append(outcome)
            else:
                failures.append(outcome)
                log.warning("seed pass failed for %s: %s", *outcome)

    return samples, failures


def group_by_style(samples: list[Sample], year_region: slice) -> dict[str, list[Sample]]:
    """Split labelled samples into render styles by finding the era boundary.

    Google changed the watermark rendering once in the covered range: the older
    form sets the year tight against the copyright symbol, the newer one adds a
    space.  Because the change happened at a point in *time*, style is a
    function of the year, so the split is found by testing every possible
    boundary year and keeping the one that best separates the similarity
    structure.

    That is far steadier than clustering the composites directly: adjacent years
    differ by a single digit and cluster together no matter how the threshold is
    set, whereas this only has to find one cut point on a line.
    """
    labelled = [s for s in samples if s.label and s.label.isdigit()]
    if len(labelled) < 2:
        return {"modern": list(samples)}

    # Compare *where the ink sits*, not what it says.  Collapsing the year block
    # to a column-energy profile keeps the layout -- which is what the style is
    # -- and discards the digit identities, which are not.  Comparing the block
    # pixel for pixel instead rewards grouping adjacent years together, and
    # finds the boundary between 2023 and 2024 rather than the real one.
    vectors = {}
    for sample in labelled:
        profile = np.abs(declutter(sample.composite)[:, year_region]).sum(axis=0)
        vectors[id(sample)] = normalise(profile)

    def cohesion(group: list[Sample]) -> float:
        if len(group) < 2:
            return 0.0
        pairs = [
            float(vectors[id(a)] @ vectors[id(b)])
            for i, a in enumerate(group)
            for b in group[i + 1 :]
        ]
        return float(np.mean(pairs))

    def separation(a: list[Sample], b: list[Sample]) -> float:
        pairs = [float(vectors[id(x)] @ vectors[id(y)]) for x in a for y in b]
        return float(np.mean(pairs)) if pairs else 0.0

    years = sorted({int(s.label) for s in labelled})
    best: tuple[float, int] | None = None
    for boundary in years[1:]:
        older = [s for s in labelled if int(s.label) < boundary]
        newer = [s for s in labelled if int(s.label) >= boundary]
        if len(older) < 2 or len(newer) < 2:
            continue
        gain = (cohesion(older) + cohesion(newer)) / 2 - separation(older, newer)
        if best is None or gain > best[0]:
            best = (gain, boundary)

    if best is None or best[0] < STYLE_SPLIT_GAIN:
        log.info("single render style across %d panoramas", len(labelled))
        return {"modern": list(samples)}

    gain, boundary = best
    log.info("render style boundary at %d (separation gain %.2f)", boundary, gain)
    return {
        "legacy": [s for s in samples if s.label and int(s.label) < boundary],
        "modern": [s for s in samples if s.label and int(s.label) >= boundary],
    }


def _one_per_year(samples: list[Sample]) -> list[np.ndarray]:
    """Cleanest composite for each distinct year, best first.

    Most inliers means most noise averaged away.  Choosing these arbitrarily
    instead makes the whole bootstrap a coin flip on dictionary order.
    """
    ranked = sorted((s for s in samples if s.label), key=lambda s: -s.instances)
    by_label: dict[str, np.ndarray] = {}
    for sample in ranked:
        by_label.setdefault(sample.label, sample.composite)
    return sorted(by_label.values(), key=lambda c: -float(np.ptp(c)))


def locate_wordmark(samples: list[Sample]) -> tuple[int, int, int, int]:
    """Find the wordmark window from composites spanning at least two years."""
    pool = _one_per_year(samples)
    if len(pool) < 2:
        raise ValueError("need panoramas from at least two years to locate the wordmark")
    # No alignment step: every composite was stacked at the same anchor offset,
    # so they already share a frame.
    return invariant_window(pool)


def derive_anchor(
    samples: list[Sample], window: tuple[int, int, int, int]
) -> tuple[np.ndarray, int, int]:
    """Cut the anchor for a group of samples from its sharpest composite.

    The window is passed in rather than re-derived.  Once composites are locked
    to an anchor they no longer look like the free-floating seed composites the
    window search expects, and re-running that search on them is unstable.
    """
    pool = [s.composite for s in sorted(samples, key=lambda s: -s.instances)]
    if not pool:
        raise ValueError("no composites to cut an anchor from")
    return cut_anchor(pool, window)


def canonical_samples(
    pano_ids: list[str],
    anchors: list[Template],
    fetcher: TileFetcher,
    labels: dict[str, str] | None = None,
    zoom: int = DEFAULT_ZOOM,
    rows: str = "top",
    threshold: float = DETECT_THRESHOLD,
    workers: int = 8,
) -> tuple[list[Sample], list[tuple[str, str]]]:
    """Re-analyse panoramas with real anchors, producing canonical composites."""
    samples: list[Sample] = []
    failures: list[tuple[str, str]] = []
    labels = labels or {}

    def one(pano_id: str):
        try:
            field_ = _fetch_field(fetcher, pano_id, zoom, rows)
        except PanoramaUnavailable as exc:
            return (pano_id, str(exc))
        result = build_composite(field_, anchors, threshold=threshold)
        if result.instances == 0:
            return (pano_id, "no watermark instances detected")
        return Sample(
            pano_id=pano_id,
            composite=result.composite,
            instances=result.instances,
            label=labels.get(pano_id),
            style=result.style,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(one, pano_ids):
            if isinstance(outcome, Sample):
                samples.append(outcome)
            else:
                failures.append(outcome)

    return samples, failures


def assemble(
    styled: dict[str, list[Sample]],
    anchors: dict[str, np.ndarray],
    year_region: slice,
    pitch: int,
) -> Bank:
    """Average style-grouped samples into per-(style, year) templates."""
    bank = Bank(anchors=anchors)

    # Templates are stored decluttered: they are only ever *compared* against a
    # runtime composite, never used as a detection filter, and classify() cleans
    # its input the same way.
    grouped: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for style, members in styled.items():
        for sample in members:
            if sample.label and sample.label.isdigit():
                grouped[(style, int(sample.label))].append(declutter(sample.composite))

    for key, composites in sorted(grouped.items()):
        bank.templates[key] = average_composites(composites)

    for style in sorted(styled):
        templates = [t for (s, _), t in bank.templates.items() if s == style]
        if len(templates) >= 2:
            bank.slots[style] = derive_digit_slots(templates, year_region, pitch)
        elif templates:
            # A style with a single year cannot show where its digits vary; fall
            # back to an even split of the year block.
            span = year_region.stop - year_region.start
            step = span // 4
            bank.slots[style] = [
                (year_region.start + i * step, year_region.start + (i + 1) * step)
                for i in range(4)
            ]

    return bank


def build_seed_bank(
    labelled: dict[str, str],
    fetcher: TileFetcher,
    zoom: int = DEFAULT_ZOOM,
    rows: str = "top",
    workers: int = 8,
    seed_font=None,
) -> tuple[Bank, list[tuple[str, str]]]:
    """Build a bank from ``{pano_id: "2024"}``, bootstrapping templates from scratch.

    Panoramas labelled ``None`` are excluded: they carry no watermark, so a seed
    pass over them would only average noise into the anchor.
    """
    years_only = {p: y for p, y in labelled.items() if y.isdigit()}
    if not years_only:
        raise ValueError("no panoramas with a year label; cannot bootstrap a bank")

    font = find_seed_font(seed_font)
    log.info("seeding from %s", font)

    ids = sorted(years_only)
    log.info("running synthetic seed pass over %d panoramas", len(ids))
    samples, failures = discover_samples(ids, fetcher, font, zoom, rows, workers)
    for sample in samples:
        sample.label = years_only.get(sample.pano_id)

    if len(samples) < 2:
        raise BootstrapFailed(
            f"only {len(samples)} panoramas yielded a seed composite; "
            "cannot bootstrap a bank"
        )

    # The wordmark's position is measured once, from the free-floating seed
    # composites, and reused from here on.  Every later composite is locked to an
    # anchor and no longer has the structure this search expects.
    window = locate_wordmark(samples)
    top, bottom, left, right = window
    log.info("wordmark window: rows %d-%d, columns %d-%d", top, bottom, left, right)

    # One anchor to begin with.  Both renderings share the same wordmark, so a
    # single anchor finds instances in both; style is separated afterwards, once
    # the composites are clean.
    primary, _, _ = derive_anchor(samples, window)
    log.info("bootstrapped anchor: %dx%d px", primary.shape[1], primary.shape[0])

    # The year sits immediately left of the wordmark in both renderings.
    year_region = slice(max(0, left - YEAR_SPAN), max(1, left - 2))
    # "Google" is six glyphs, so its width divided by six is this font's average
    # advance -- and the year is set in the same font at the same size.  Measured
    # this way the pitch stays right even when only one digit varies across the
    # bank's years, which is where measuring it from the digits themselves fails.
    pitch = int(np.clip(round((right - left) / len(SEED_WORD)), 5, 12))
    log.info("year block at columns %d-%d, digit pitch %d px",
             year_region.start, year_region.stop, pitch)

    log.info("re-analysing %d panoramas with the bootstrapped anchor", len(ids))
    first_pass, more_failures = canonical_samples(
        ids,
        [Template(array=primary, style="primary", cx=ANCHOR_CX, cy=ANCHOR_CY)],
        fetcher, labels=years_only, zoom=zoom, rows=rows, workers=workers,
    )
    if len(first_pass) < 2:
        raise BootstrapFailed("the bootstrapped anchor detected nothing; cannot build a bank")
    log.info(
        "first pass: %d panoramas, median %d instances",
        len(first_pass), int(np.median([s.instances for s in first_pass])),
    )

    styles = group_by_style(first_pass, year_region)
    log.info(
        "render styles: %s",
        ", ".join(f"{name} ({len(members)} panoramas)" for name, members in styles.items()),
    )

    # A style-specific anchor is a sharper filter than one shared across
    # renderings, so cut one per style now that the composites are clean.
    anchors: dict[str, np.ndarray] = {}
    for style, members in styles.items():
        try:
            anchor, _, _ = derive_anchor(members, window)
        except ValueError:
            anchor = primary
        anchors[style] = anchor
        log.info("anchor for %s: %dx%d px", style, anchor.shape[1], anchor.shape[0])

    log.info("final pass with %d style anchor(s)", len(anchors))
    templates = [
        Template(array=a, style=s, cx=ANCHOR_CX, cy=ANCHOR_CY) for s, a in anchors.items()
    ]
    final, final_failures = canonical_samples(
        ids, templates, fetcher, labels=years_only, zoom=zoom, rows=rows, workers=workers
    )

    # Group by era, the same split the anchors were cut from.  Grouping by which
    # anchor happened to win instead would let a single panorama detected by the
    # other style's anchor land its template in the wrong group.
    era = {
        int(s.label): name
        for name, members in styles.items()
        for s in members
        if s.label and s.label.isdigit()
    }
    by_style: dict[str, list[Sample]] = defaultdict(list)
    for sample in final:
        if sample.label and sample.label.isdigit():
            by_style[era.get(int(sample.label), "modern")].append(sample)

    return (
        assemble(dict(by_style), anchors, year_region, pitch),
        failures + more_failures + final_failures,
    )


def harvest(
    pano_ids: list[str],
    bank: Bank,
    fetcher: TileFetcher,
    zoom: int = DEFAULT_ZOOM,
    rows: str = "top",
    workers: int = 8,
) -> HarvestReport:
    """Analyse unlabelled panoramas and cluster them into (style, year) groups.

    Everything is clustered together and the style is decided *per cluster*,
    from its centroid.  Grouping by the per-panorama style first splits a single
    year across two groups whenever the wrong anchor happens to win on a few
    panoramas -- observed producing two "2026" clusters, one of them labelled
    legacy.  A centroid is an average of the whole cluster, so deciding from it
    is both more reliable and decided once instead of per panorama.
    """
    anchors = [Template(array=a, style=s) for s, a in sorted(bank.anchors.items())]
    samples, failures = canonical_samples(
        pano_ids, anchors, fetcher, zoom=zoom, rows=rows, workers=workers
    )

    report = HarvestReport(failures=failures)
    if not samples:
        return report

    # The digit block sits at essentially the same columns in every style, so a
    # single window serves for clustering across all of them.
    spans = [s for slots in bank.slots.values() for s in slots]
    columns = (
        slice(min(a for a, _ in spans), max(b for _, b in spans))
        if spans
        else slice(0, PATCH_W)
    )

    cleaned = [declutter(s.composite) for s in samples]
    for cluster in cluster_composites(cleaned, columns=columns, threshold=YEAR_CUT):
        members = [samples[i] for i in cluster]
        centroid = average_composites([cleaned[i] for i in cluster])
        style = _closest_style(bank, centroid)
        for member in members:
            member.style = style
        report.clusters.append(members)

    report.clusters.sort(key=len, reverse=True)
    return report


def _closest_style(bank: Bank, centroid: np.ndarray) -> str:
    """Which render style a cluster centroid matches best."""
    best: tuple[float, str] | None = None
    for style in bank.styles:
        templates = [t for (s, _), t in bank.templates.items() if s == style]
        if not templates:
            continue
        reference = np.mean(np.stack([normalise(t) for t in templates]), axis=0)
        score, _, _ = best_alignment(reference, centroid)
        if best is None or score > best[0]:
            best = (score, style)
    return best[1] if best else "modern"
