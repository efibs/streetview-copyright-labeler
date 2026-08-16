"""Bootstrapping a template bank from nothing.

The classifier needs an *empirical* anchor -- a picture of the real ``Google``
wordmark as Google's tile server renders it.  Synthetic text is too poor a match
to classify with (measured 10/20 against ground truth), but it is easily good
enough to get a foothold: a plain Roboto ``Google`` scores ~0.6 against real
instances, which finds 30-70 of them per panorama.  Averaging those yields a
real composite, and the real anchor is cut from that.

So: synthetic text opens the door, then the data takes over.  Nothing here runs
during normal labelling -- it runs once, when a bank is built.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .geometry import ANCHOR_CX, ANCHOR_CY, PATCH_H, PATCH_W
from .signal import Correlator, consensus, declutter, find_peaks, normalise

log = logging.getLogger(__name__)

# Roboto is Google's own UI font and the closest match to the watermark, but any
# clean grotesque gets the bootstrap far enough to be taken over by real data.
SEED_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/roboto/hinted/Roboto-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "C:/Windows/Fonts/arial.ttf",
)
SEED_WORD = "Google"
SEED_SIZES = (14, 15, 16)
SEED_THRESHOLD = 0.45
MIN_SEED_HITS = 6
# How many seed hits to consider before consensus filtering.  Real instances
# rank near the top, so there is nothing to gain from a longer tail.
SEED_CANDIDATES = 120


class BootstrapFailed(RuntimeError):
    """Raised when no seed font is available or no panorama yields a composite."""


def find_seed_font(explicit: Path | None = None) -> Path:
    """Locate a usable TrueType font for the synthetic seed."""
    if explicit:
        if Path(explicit).exists():
            return Path(explicit)
        raise BootstrapFailed(f"seed font not found: {explicit}")
    for candidate in SEED_FONT_CANDIDATES:
        if Path(candidate).exists():
            return Path(candidate)
    raise BootstrapFailed(
        "no seed font found. Pass --seed-font /path/to/a/sans-serif.ttf "
        "(Roboto or Arial work best)."
    )


def render_seed(font_path: Path, size: int, word: str = SEED_WORD, pad: int = 3) -> np.ndarray:
    """Render ``word`` as a rough matched filter."""
    font = ImageFont.truetype(str(font_path), size)
    canvas = Image.new("L", (400, 120), 0)
    ImageDraw.Draw(canvas).text((200, 60), word, font=font, fill=255, anchor="mm")
    box = canvas.getbbox()
    if not box:
        raise BootstrapFailed("seed font rendered nothing")
    glyphs = canvas.crop(box)
    padded = Image.new("L", (glyphs.width + 2 * pad, glyphs.height + 2 * pad), 0)
    padded.paste(glyphs, (pad, pad))
    # Slight blur matches the watermark's anti-aliased, alpha-blended edges.
    return np.asarray(padded.filter(ImageFilter.GaussianBlur(0.7)), np.float32)


def _patches_at(field: np.ndarray, peaks, cx: int, cy: int) -> list[np.ndarray]:
    height, width = field.shape
    out = []
    for peak in peaks:
        top, left = peak.y - cy, peak.x - cx
        if top < 0 or left < 0 or top + PATCH_H > height or left + PATCH_W > width:
            continue
        out.append(field[top : top + PATCH_H, left : left + PATCH_W])
    return out


def seed_composite(
    field: np.ndarray, font_path: Path, threshold: float = SEED_THRESHOLD
) -> tuple[np.ndarray, int] | None:
    """Average watermark instances located by a synthetic ``Google``.

    Tries a few point sizes and keeps whichever finds the most instances.
    Returns ``(composite, instances)`` in canonical patch geometry.
    """
    correlator = Correlator(field)
    best: tuple[int, np.ndarray] | None = None

    for size in SEED_SIZES:
        seed = render_seed(font_path, size)
        peaks = find_peaks(correlator.match(seed), threshold, limit=SEED_CANDIDATES)
        if len(peaks) < MIN_SEED_HITS:
            continue
        composite, inliers = consensus(_patches_at(field, peaks, ANCHOR_CX, ANCHOR_CY))
        if composite is not None and len(inliers) >= MIN_SEED_HITS:
            if best is None or len(inliers) > best[0]:
                best = (len(inliers), composite.astype(np.float32))

    if best is None:
        return None
    used, composite = best
    # Returned raw, in the same high-pass domain the detector correlates against.
    # Cleaning happens only where the *location* of the wordmark is being found;
    # an anchor cut from a cleaned composite is spectrally mismatched and matches
    # far too much.
    return composite, used


def align_to(reference: np.ndarray, other: np.ndarray, radius: int = 30) -> np.ndarray:
    """Shift ``other`` onto ``reference``'s frame by maximum correlation."""
    best = (-2.0, 0, 0)
    for dy in range(-4, 5):
        for dx in range(-radius, radius + 1):
            shifted = np.roll(other, (dy, dx), axis=(0, 1))
            score = float((normalise(reference) * normalise(shifted)).sum())
            if score > best[0]:
                best = (score, dy, dx)
    _, dy, dx = best
    return np.roll(other, (dy, dx), axis=(0, 1))


def _runs(mask: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """Contiguous True spans of ``mask`` as half-open ``(start, stop)`` pairs.

    ``bridge`` joins spans separated by that many False columns or fewer, so a
    single dip between two letters does not split one wordmark into two.
    """
    spans: list[tuple[int, int]] = []
    start = None
    for index, flag in enumerate(np.append(mask, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None

    if bridge <= 0 or not spans:
        return spans

    merged = [spans[0]]
    for left, right in spans[1:]:
        if left - merged[-1][1] <= bridge:
            merged[-1] = (merged[-1][0], right)
        else:
            merged.append((left, right))
    return merged


def invariant_window(
    composites: list[np.ndarray],
    min_width: int = 28,
    max_width: int = 60,
    keep: float = 0.40,
) -> tuple[int, int, int, int]:
    """Locate the year-invariant block in composites spanning several years.

    Averaging composites from different years *is* the invariance test: glyphs
    that every year shares reinforce, while the year digits -- different in each
    -- average towards zero.  The widest surviving block of stroke energy is the
    ``Google`` wordmark.

    Returns ``(top, bottom, left, right)``.  Only the *window* comes from the
    mean; the anchor pixels themselves must be cut from a single sharp
    composite, because a blurred average makes a hopelessly unselective filter.

    Composites must already share a frame -- they do, being stacked at a common
    anchor offset.  Do not pre-align them: across different years the best
    alignment is ambiguous and will happily register a digit onto a letter.
    """
    if len(composites) < 2:
        raise ValueError("need composites from at least two different years")

    # Strip each composite's scene ghost first: it is low-frequency, differs per
    # panorama, and would otherwise swamp the glyph energy this searches for.
    mean = np.stack([normalise(declutter(c)) for c in composites]).mean(axis=0)

    band = slice(2, PATCH_H - 2)
    column_energy = np.abs(mean[band]).sum(axis=0)

    # Sweep the cut level rather than fixing it.  On noisy composites a low cut
    # is needed to see the wordmark at all; on clean ones the same cut swallows
    # the entire string and returns a full-width "anchor" that matches
    # everything.  The wordmark is a bounded object, so accept the first level
    # whose widest block is plausibly one.
    chosen: tuple[int, int] | None = None
    widest_seen = 0
    for level in np.arange(keep, 0.95, 0.05):
        columns = _runs(column_energy > level * column_energy.max(), bridge=2)
        spans = [s for s in columns if min_width <= s[1] - s[0] <= max_width]
        widest_seen = max(widest_seen, max((s[1] - s[0] for s in columns), default=0))
        if spans:
            chosen = max(spans, key=lambda span: span[1] - span[0])
            break

    if chosen is None:
        raise ValueError(
            f"no invariant block between {min_width} and {max_width}px wide "
            f"(widest seen was {widest_seen}px); the seed composites are too "
            "noisy to bootstrap from"
        )
    left, right = chosen

    # Tighten vertically to the text band -- a snug anchor is a sharper filter.
    row_energy = np.abs(mean[:, left:right]).sum(axis=1)
    rows = _runs(row_energy > 0.35 * row_energy.max())
    top, bottom = (
        max(rows, key=lambda span: span[1] - span[0]) if rows else (band.start, band.stop)
    )
    return max(0, top - 1), min(PATCH_H, bottom + 1), left, right


def sharpness(composite: np.ndarray, window: tuple[int, int, int, int]) -> float:
    """How cleanly a composite's watermark stands out from its background.

    Ratio of stroke contrast inside the wordmark window to the residual scene
    texture outside it.  Picks the composite worth cutting a template from.
    """
    top, bottom, left, right = window
    cleaned = declutter(composite)
    inside = cleaned[top:bottom, left:right]
    outside = np.concatenate([cleaned[:top].ravel(), cleaned[bottom:].ravel()])
    if outside.size == 0 or float(np.std(outside)) < 1e-9:
        return 0.0
    return float(np.ptp(inside) / np.std(outside))


# Rows kept above and below the glyph band.  The watermark is drawn with a dark
# halo, and that halo is what makes the template selective -- cropped to the
# bright strokes alone it matches any small bright blob, which measurably
# multiplies false detections on panoramas that carry no watermark at all.
ANCHOR_ROW_PAD = 4


def cut_anchor(
    composites: list[np.ndarray],
    window: tuple[int, int, int, int],
    row_pad: int = ANCHOR_ROW_PAD,
) -> tuple[np.ndarray, int, int]:
    """Cut the anchor from whichever composite shows the wordmark most sharply."""
    top, bottom, left, right = window
    best = max(composites, key=lambda c: sharpness(c, window))
    top = max(0, top - row_pad)
    bottom = min(PATCH_H, bottom + row_pad)
    anchor = best[top:bottom, left:right].astype(np.float32)
    return anchor, (left + right) // 2, (top + bottom) // 2
