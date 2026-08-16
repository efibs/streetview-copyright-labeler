"""Regenerate the illustrations used in the README.

Everything here is produced from a real panorama by the same code paths the
labeller uses, so the pictures cannot drift away from what the program does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from cr_labeler.bank import Bank  # noqa: E402
from cr_labeler.composite import Template, _patches, build_composites  # noqa: E402
from cr_labeler.fetch import TileCache, TileFetcher  # noqa: E402
from cr_labeler.geometry import DETECT_THRESHOLD  # noqa: E402
from cr_labeler.signal import (  # noqa: E402
    Correlator,
    declutter,
    find_peaks,
    highpass,
)

PANO = "wuZLx2SV9tkH5qFQkBdGvQ"  # ground truth 2024
OUT = REPO / "docs"


def grey(array: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    spread = float(np.ptp(array))
    scaled = (array - array.min()) / (spread + 1e-9) * 255.0
    image = Image.fromarray(scaled.astype(np.uint8))
    return image.resize(size, Image.LANCZOS) if size else image


def captioned(panes: list[tuple[str, Image.Image]], width: int, pane_h: int) -> Image.Image:
    sheet = Image.new("L", (width, (pane_h + 22) * len(panes)), 20)
    draw = ImageDraw.Draw(sheet)
    for i, (caption, image) in enumerate(panes):
        sheet.paste(image.resize((width, pane_h), Image.LANCZOS), (0, i * (pane_h + 22) + 20))
        draw.text((5, i * (pane_h + 22) + 6), caption, fill=255)
    return sheet


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fetcher = TileFetcher(TileCache(REPO / "cache"))
    bank = Bank.load()
    anchor = Template(array=bank.anchors[sorted(bank.anchors)[-1]], style="modern")

    panorama = fetcher.fetch(PANO, rows="top").image
    field = highpass(panorama)
    height, width = field.shape
    peaks = find_peaks(Correlator(field).match(anchor.array), DETECT_THRESHOLD, limit=150)

    # 1. one stamp close up, before and after the high-pass ------------------
    # Shown at the panorama's own brightness on the left, so the comparison is
    # honest: the stamp really is a few grey levels on a smooth sky.
    peak = peaks[0]
    crop = (slice(peak.y - 26, peak.y + 26), slice(peak.x - 130, peak.x + 130))
    raw = np.asarray(panorama.convert("L"), np.float32)[crop]
    captioned(
        [("1. as the panorama shows it -- a few grey levels on a smooth sky",
          Image.fromarray(np.clip(raw, 0, 255).astype(np.uint8))),
         ("2. after the high-pass -- the sky's gradient subtracted away",
          grey(field[crop]))],
        1040, 208,
    ).save(OUT / "01-highpass.png")

    # 2. one instance against the average of many ---------------------------
    patches = _patches(field, peaks, anchor)
    result = build_composites(field, [anchor])[0]
    single = declutter(patches[len(patches) // 2])
    captioned(
        [("3. a single instance, straight out of the panorama -- barely legible", grey(single)),
         (f"4. {result.instances} of them averaged -- the year is unambiguous",
          grey(declutter(result.composite)))],
        700, 170,
    ).save(OUT / "02-averaging.png")

    # 3. where the stamps are ------------------------------------------------
    shot = panorama.convert("L").resize((1200, int(1200 * height / width)))
    shot = shot.point(lambda v: 60 + v // 3)
    marked = shot.convert("RGB")
    draw = ImageDraw.Draw(marked)
    sx, sy = 1200 / width, shot.height / height
    for peak in peaks[:40]:
        x, y = peak.x * sx, peak.y * sy
        draw.rectangle([x - 34, y - 9, x + 34, y + 9], outline=(255, 90, 90), width=2)
    marked.save(OUT / "03-instances.png")

    print(f"wrote images to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
