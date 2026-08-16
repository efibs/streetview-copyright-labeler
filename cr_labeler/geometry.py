"""Fixed pixel geometry of the Street View copyright watermark.

Every constant here was measured against real panorama tiles at zoom 3.  The
watermark is composited by Google's tile server at a *constant pixel size*
regardless of zoom level, so these numbers hold at any zoom we fetch.

All extracted patches are anchored on the ``Google`` wordmark, which is the one
part of the string that never changes.  That makes patches from different
panoramas directly comparable without any per-image alignment search.
"""

from __future__ import annotations

# --- Patch geometry -------------------------------------------------------
# Size of the patch cut around every detected watermark instance.  Wide enough
# to hold the full "(c) YYYY Google" string plus margin in both render styles.
PATCH_H = 32
PATCH_W = 132

# Where the anchor ("Google" wordmark) centre lands inside the patch.  Chosen so
# the year digits fall comfortably inside the patch for both styles.
ANCHOR_CX = 98
ANCHOR_CY = 15

# --- Detection ------------------------------------------------------------
# Sigma of the Gaussian used for the high-pass.  The watermark strokes are
# 1-2 px wide; sigma=4 removes scene structure and sky gradients while leaving
# the strokes intact.
HIGHPASS_SIGMA = 4.0

# Non-maximum suppression half-window, in pixels.  Roughly the footprint of one
# watermark so two peaks cannot come from the same instance.
NMS_DY = 18
NMS_DX = 70

# Permissive by design: the legacy render style only scores ~0.44 against a
# modern anchor.  False positives are rejected later by the composite quality
# gate in classify.py, not here.
DETECT_THRESHOLD = 0.36

# Hard cap on instances per panorama, so a pathological image cannot stall the
# non-maximum suppression loop.
MAX_INSTANCES = 400

# --- Tiles ----------------------------------------------------------------
TILE_PX = 512
DEFAULT_ZOOM = 3


def pano_grid(zoom: int) -> tuple[int, int]:
    """Return ``(cols, rows)`` of the 512 px tile grid at ``zoom``."""
    return 2**zoom, 2 ** (zoom - 1)
