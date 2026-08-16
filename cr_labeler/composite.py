"""Turn a panorama into one clean, averaged image of its watermark.

A single watermark instance sits only a few grey levels above the noise floor.
Averaging N aligned instances suppresses that noise by roughly sqrt(N) while the
watermark, being identical in every instance, adds coherently.  A typical
panorama yields 20-40 instances, which is more than enough to render the year
unambiguously legible.

Detection runs at a permissive threshold so that the faint legacy rendering is
not missed, and the resulting false positives are removed by consensus rather
than by threshold tuning -- see :func:`cr_labeler.signal.consensus`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import (
    ANCHOR_CX,
    ANCHOR_CY,
    DETECT_THRESHOLD,
    PATCH_H,
    PATCH_W,
)
from .signal import Peak, consensus, correlator_for, find_peaks, shift_subpixel

# Instances considered for consensus, highest scoring first.  Real watermarks
# rank near the top; a longer tail only costs time.
CONSENSUS_CANDIDATES = 150


@dataclass(frozen=True)
class Template:
    """A correlation template plus where its centre sits in patch coordinates."""

    array: np.ndarray
    style: str
    cx: int = ANCHOR_CX
    cy: int = ANCHOR_CY

    @classmethod
    def from_patch(cls, patch: np.ndarray, style: str) -> Template:
        """Wrap a full-size patch (e.g. a composite) for use as a template."""
        return cls(array=patch, style=style, cx=PATCH_W // 2, cy=PATCH_H // 2)


@dataclass
class CompositeResult:
    """The averaged watermark image and the evidence behind it."""

    composite: np.ndarray
    instances: int
    style: str
    scores: list[float]
    detections: int = 0

    @property
    def median_score(self) -> float:
        return float(np.median(self.scores)) if self.scores else 0.0


def _patches(field: np.ndarray, peaks: list[Peak], template: Template) -> list[np.ndarray]:
    """Cut a patch around each peak, on a common sub-pixel grid."""
    height, width = field.shape
    out: list[np.ndarray] = []
    for peak in peaks:
        top = peak.y - template.cy
        left = peak.x - template.cx
        if top < 0 or left < 0 or top + PATCH_H > height or left + PATCH_W > width:
            continue
        patch = field[top : top + PATCH_H, left : left + PATCH_W]
        # The true peak sits at (+dy, +dx) from the integer maximum, so shifting
        # back by that amount puts every instance on the same grid.  Without it,
        # averaging blurs exactly the glyph edges that carry the year.
        out.append(shift_subpixel(patch, -peak.dy, -peak.dx))
    return out


def _empty(style: str) -> CompositeResult:
    return CompositeResult(
        composite=np.zeros((PATCH_H, PATCH_W), dtype=np.float32),
        instances=0,
        style=style,
        scores=[],
        detections=0,
    )


def build_composites(
    field: np.ndarray,
    anchors: list[Template],
    threshold: float = DETECT_THRESHOLD,
    candidates: int = CONSENSUS_CANDIDATES,
) -> list[CompositeResult]:
    """Build one composite per anchor.

    Every anchor gets its own composite rather than the anchors competing on raw
    correlation mass.  A wrong-style anchor can easily out-detect the right one
    -- it fires on texture, so it accumulates plenty of weak matches -- and the
    only reliable way to tell which composite is the real watermark is to look
    at the composites themselves, which the caller does by classifying each.
    """
    if not anchors:
        raise ValueError("at least one anchor is required")

    correlator = correlator_for(field)
    results: list[CompositeResult] = []

    for anchor in anchors:
        peaks = find_peaks(correlator.match(anchor.array), threshold, limit=candidates)
        if not peaks:
            results.append(_empty(anchor.style))
            continue
        composite, inliers = consensus(_patches(field, peaks, anchor))
        if composite is None:
            results.append(_empty(anchor.style))
            continue
        results.append(
            CompositeResult(
                composite=composite.astype(np.float32),
                instances=len(inliers),
                style=anchor.style,
                scores=[peaks[i].score for i in inliers if i < len(peaks)],
                detections=len(peaks),
            )
        )

    return results


def build_composite(
    field: np.ndarray,
    anchors: list[Template],
    threshold: float = DETECT_THRESHOLD,
    candidates: int = CONSENSUS_CANDIDATES,
) -> CompositeResult:
    """Best single composite, judged by how many instances survived consensus."""
    results = build_composites(field, anchors, threshold, candidates)
    return max(results, key=lambda r: r.instances)
