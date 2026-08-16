"""Synthetic panoramas for testing without touching the network.

The watermark is reproduced the way Google composites it -- faint light text
with a dark halo, stamped at scattered positions over scene-like content -- so
the detection and averaging path can be exercised end to end offline.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from cr_labeler.discover import find_seed_font

PANO_W, PANO_H = 2048, 512


@pytest.fixture(scope="session")
def font_path():
    try:
        return find_seed_font(None)
    except Exception:
        pytest.skip("no TrueType font available for synthetic rendering")


def render_watermark(font_path, text: str, size: int = 15) -> Image.Image:
    """Light text with a dark halo, matching how the real overlay is drawn."""
    font = ImageFont.truetype(str(font_path), size)
    canvas = Image.new("L", (200, 40), 0)
    draw = ImageDraw.Draw(canvas)
    draw.text((100, 20), text, font=font, fill=255, anchor="mm")
    box = canvas.getbbox()
    glyphs = canvas.crop(box) if box else canvas

    padded = Image.new("L", (glyphs.width + 8, glyphs.height + 8), 0)
    padded.paste(glyphs, (4, 4))
    halo = padded.filter(ImageFilter.GaussianBlur(1.5))
    combined = np.asarray(padded, np.float32) - 0.6 * np.asarray(halo, np.float32)
    return Image.fromarray(np.clip(combined + 128, 0, 255).astype(np.uint8))


def make_panorama(
    font_path,
    text: str = "© 2024 Google",
    count: int = 24,
    strength: float = 0.16,
    seed: int = 7,
) -> tuple[Image.Image, list[tuple[int, int]]]:
    """A scene-like panorama with ``count`` watermark stamps blended in.

    Returns the image and the stamp centres, so tests can check that detection
    finds them where they actually are.
    """
    rng = np.random.default_rng(seed)

    # Smooth vertical gradient plus low-frequency blobs: sky over ground, with
    # enough structure that a detector keying on "any edge" would go wrong.
    gradient = np.linspace(210, 90, PANO_H, dtype=np.float32)[:, None]
    scene = np.repeat(gradient, PANO_W, axis=1)
    blobs = rng.normal(0, 40, (PANO_H // 16, PANO_W // 16)).astype(np.float32)
    blobs = np.asarray(
        Image.fromarray(blobs).resize((PANO_W, PANO_H), Image.BICUBIC), np.float32
    )
    scene = scene + blobs + rng.normal(0, 2.0, (PANO_H, PANO_W)).astype(np.float32)

    stamp = np.asarray(render_watermark(font_path, text), np.float32) - 128.0
    sh, sw = stamp.shape
    centres: list[tuple[int, int]] = []

    for _ in range(count):
        y = int(rng.integers(sh, PANO_H - sh))
        x = int(rng.integers(sw, PANO_W - sw))
        scene[y : y + sh, x : x + sw] += strength * stamp
        centres.append((x + sw // 2, y + sh // 2))

    image = Image.fromarray(np.clip(scene, 0, 255).astype(np.uint8)).convert("RGB")
    return image, centres


@pytest.fixture
def watermarked(font_path):
    return make_panorama(font_path)


@pytest.fixture
def blank(font_path):
    """Same scene generator, no watermark stamped in."""
    image, _ = make_panorama(font_path, count=0)
    return image
