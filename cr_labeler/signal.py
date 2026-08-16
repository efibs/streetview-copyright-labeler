"""Matched-filter machinery: high-pass, FFT cross-correlation, peak picking.

The watermark is a faint alpha-blend on top of arbitrary scene content.  Two
steps make it recoverable:

1. A high-pass removes everything the scene contributes at low spatial
   frequency (sky gradients, buildings, foliage) and leaves the thin watermark
   strokes standing in near-zero background.
2. Normalised cross-correlation against a template of the invariant ``Google``
   wordmark locates every instance regardless of what is behind it.

Correlation is done in the Fourier domain.  The panorama's spectrum is computed
once and reused for every template, which is what makes multi-style detection
cost roughly the same as single-style.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from .geometry import HIGHPASS_SIGMA, MAX_INSTANCES, NMS_DX, NMS_DY


def highpass(image: Image.Image, sigma: float = HIGHPASS_SIGMA) -> np.ndarray:
    """Return the high-frequency residual of ``image`` as float32.

    ``grey - blur(grey)``.  Pillow's Gaussian is separable and C-implemented,
    which beats a numpy convolution comfortably at panorama sizes.
    """
    grey = image.convert("L")
    flat = np.asarray(grey, dtype=np.float32)
    blurred = np.asarray(grey.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)
    return flat - blurred


def normalise(patch: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-norm copy of ``patch``; correlating these gives NCC."""
    centred = patch - patch.mean()
    norm = float(np.linalg.norm(centred))
    if norm < 1e-9:
        return np.zeros_like(centred)
    return centred / norm


# Averaging instances leaves a smooth ghost of the scene behind the glyphs.  It
# differs from panorama to panorama, so it must go before two composites are
# compared -- but *not* before a composite is used as a detection template,
# which has to stay in the same spectral domain as the field it searches.
COMPOSITE_HIGHPASS_SIGMA = 3.0


def declutter(composite: np.ndarray, sigma: float = COMPOSITE_HIGHPASS_SIGMA) -> np.ndarray:
    """Strip the smooth scene ghost from a composite, leaving the glyphs."""
    spread = float(np.ptp(composite))
    if spread < 1e-9:
        return composite.astype(np.float32)
    scaled = (composite - composite.min()) / spread * 255.0
    image = Image.fromarray(scaled.astype(np.uint8))
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(sigma)), np.float32)
    return scaled.astype(np.float32) - blurred


class Correlator:
    """Cross-correlates one image against many templates.

    Holds the image spectrum and the running-sum spectra needed for the
    normalisation denominator, so each additional template costs one inverse
    FFT rather than a full re-analysis.
    """

    def __init__(self, field: np.ndarray):
        self.field = field
        self.shape = field.shape
        self._spec = np.fft.rfft2(field)
        self._spec_sq = np.fft.rfft2(field * field)

    def match(self, template: np.ndarray) -> np.ndarray:
        """Return the NCC surface, shifted so a peak sits at the template centre."""
        th, tw = template.shape
        tpl = normalise(template)

        # Correlation == convolution with the flipped kernel.
        kernel = np.fft.rfft2(tpl[::-1, ::-1], self.shape)
        numerator = np.fft.irfft2(self._spec * kernel, self.shape)

        # Local mean/variance of the image under the sliding window.
        window = np.fft.rfft2(np.ones_like(template), self.shape)
        total = np.fft.irfft2(self._spec * window, self.shape)
        total_sq = np.fft.irfft2(self._spec_sq * window, self.shape)
        count = th * tw
        variance = np.maximum(total_sq - total * total / count, 1e-6)

        surface = numerator / np.sqrt(variance)
        # irfft2 places the correlation origin at the template's top-left; roll
        # it so index (y, x) means "template centred here".
        return np.roll(surface, (-(th // 2), -(tw // 2)), axis=(0, 1))


@dataclass(frozen=True)
class Peak:
    """One detected watermark instance."""

    x: int
    y: int
    score: float
    # Sub-pixel offset from the integer peak, in pixels.
    dx: float = 0.0
    dy: float = 0.0


def _parabolic_offset(low: float, mid: float, high: float) -> float:
    """Vertex of the parabola through three samples, clamped to +/- 0.5 px.

    Neighbours can be -inf where suppression has already blanked them, so
    non-finite input falls back to no sub-pixel correction.
    """
    if not (np.isfinite(low) and np.isfinite(high)):
        return 0.0
    denom = low - 2.0 * mid + high
    if abs(denom) < 1e-9:
        return 0.0
    return float(np.clip(0.5 * (low - high) / denom, -0.5, 0.5))


def find_peaks(
    surface: np.ndarray,
    threshold: float,
    limit: int = MAX_INSTANCES,
    nms_dy: int = NMS_DY,
    nms_dx: int = NMS_DX,
) -> list[Peak]:
    """Greedy peak picking with rectangular non-maximum suppression.

    Returns peaks in descending score order, each carrying a sub-pixel offset
    estimated by parabolic interpolation of its neighbourhood.
    """
    work = surface.copy()
    height, width = work.shape
    peaks: list[Peak] = []

    for _ in range(limit):
        index = int(np.argmax(work))
        y, x = divmod(index, width)
        score = float(work[y, x])
        if score < threshold:
            break

        dx = dy = 0.0
        if 0 < x < width - 1:
            dx = _parabolic_offset(work[y, x - 1], score, work[y, x + 1])
        if 0 < y < height - 1:
            dy = _parabolic_offset(work[y - 1, x], score, work[y + 1, x])
        peaks.append(Peak(x=x, y=y, score=score, dx=dx, dy=dy))

        work[
            max(0, y - nms_dy) : y + nms_dy,
            max(0, x - nms_dx) : x + nms_dx,
        ] = -np.inf

    return peaks


def shift_subpixel(patch: np.ndarray, dy: float, dx: float) -> np.ndarray:
    """Translate ``patch`` by a fractional offset using bilinear weights.

    Used to bring every instance onto a common sub-pixel grid before averaging;
    without it, averaging blurs the glyph edges that carry the year.
    """
    if abs(dy) < 1e-3 and abs(dx) < 1e-3:
        return patch

    iy, ix = int(np.floor(dy)), int(np.floor(dx))
    fy, fx = dy - iy, dx - ix
    base = np.roll(patch, (iy, ix), axis=(0, 1))
    right = np.roll(base, (0, 1), axis=(0, 1))
    down = np.roll(base, (1, 0), axis=(0, 1))
    both = np.roll(base, (1, 1), axis=(0, 1))
    return (
        base * (1 - fy) * (1 - fx)
        + right * (1 - fy) * fx
        + down * fy * (1 - fx)
        + both * fy * fx
    )


def consensus(
    patches: list[np.ndarray], agreement: float = 0.5
) -> tuple[np.ndarray | None, list[int]]:
    """Keep only the patches that agree with one another.

    Matched filtering trades precision for recall: a permissive threshold finds
    every real instance but also fires on JPEG texture, and averaging those
    false positives is what smears a composite into an unreadable blur.

    Real instances are literally the same bitmap, so they form one tight
    mutually-similar group while false positives agree with nothing.  Taking the
    largest agreeing group removes the contamination without needing a
    finely-tuned detection threshold -- and on a panorama with no watermark at
    all, no such group exists, which is exactly the answer wanted there.

    Agreement is judged on decluttered patches, because the watermark is faint
    and whatever scene happens to lie behind it otherwise dominates the
    comparison -- two genuine instances over different backgrounds would look
    nothing alike.  The *average* is taken over the original patches, keeping
    the composite in the same domain the detector works in.

    Returns ``(mean_of_inliers, inlier_indices)``.
    """
    if not patches:
        return None, []

    flat = np.stack([normalise(declutter(p)).ravel() for p in patches]).astype(np.float32)
    similarity = flat @ flat.T
    support = (similarity > agreement).sum(axis=1)
    centre = int(np.argmax(support))
    inliers = [i for i, score in enumerate(similarity[centre]) if score > agreement]

    if not inliers:
        return None, []
    return np.mean(np.stack([patches[i] for i in inliers]), axis=0), inliers


def best_alignment(
    a: np.ndarray,
    b: np.ndarray,
    radius_y: int = 1,
    radius_x: int = 2,
    columns: slice | None = None,
) -> tuple[float, int, int]:
    """Best NCC of ``a`` against ``b`` over small integer shifts of ``b``.

    Returns ``(score, dy, dx)``.  ``columns`` restricts scoring to a column
    slice, which is how per-digit comparison isolates the varying glyphs.
    """
    sl = columns if columns is not None else slice(None)
    reference = normalise(a[:, sl])
    best = (-2.0, 0, 0)
    for dy in range(-radius_y, radius_y + 1):
        for dx in range(-radius_x, radius_x + 1):
            shifted = np.roll(b, (dy, dx), axis=(0, 1))
            score = float((reference * normalise(shifted[:, sl])).sum())
            if score > best[0]:
                best = (score, dy, dx)
    return best
