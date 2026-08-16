"""Audit a labelled Vali set without needing hand-made ground truth.

Vali gives no copyright year -- that is the thing being predicted -- so accuracy
cannot be read off directly. Three independent checks get most of the way there:

1. **Capture-year consistency.** Imagery cannot be stamped with a copyright year
   older than the year it was photographed, so a prediction below the recorded
   capture year is suspicious. (The converse is not a check: re-processing
   legitimately pushes the stamp years forward.)

   Treat a violation as a prompt to look, not a proven error. It fires two ways,
   both seen in practice. Genuine misread: four panoramas reading
   "(c) 2012 Google" were reported as 2011 while 2012 was missing from the bank.
   False alarm: four panoramas reported as 2009 read "(c) 2009 Google"
   correctly, but Vali had recorded 2010 -- its `Year` describes the *default*
   panorama at that location, and `Oldest`/`Random` verification deliberately
   selects a different one. So the strata using those strategies can violate the
   check while being right.
2. **Trekker.** Most trekker panoramas carry no watermark -- verified by eye
   against a control -- so the stratum should come back overwhelmingly `None`.
   Not universally: measured at ~75%, and the remainder genuinely are stamped,
   confirmed by reading their composites. A sharp drop in that rate would mean
   detection has regressed.
3. **Eye check.** A random sample of composites is written to a contact sheet.
   Composites are legible, so predictions can simply be read against them.

Coverage and abstention rates come out alongside, per stratum.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from cr_labeler.bank import Bank  # noqa: E402
from cr_labeler.classify import classify  # noqa: E402
from cr_labeler.composite import Template, build_composites  # noqa: E402
from cr_labeler.fetch import TileCache, TileFetcher  # noqa: E402
from cr_labeler.signal import declutter, highpass  # noqa: E402


def load_locations(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows = document["customCoordinates"] if isinstance(document, dict) else document
    return {r["panoId"]: r for r in rows if r.get("panoId")}


def tags_of(row: dict) -> list[str]:
    return row.get("extra", {}).get("tags", [])


def capture_year(row: dict) -> int | None:
    for tag in tags_of(row):
        if tag.isdigit() and 2000 <= int(tag) <= 2100:
            return int(tag)
    return None


def stratum_of(row: dict) -> str:
    for tag in tags_of(row):
        if tag.startswith("stratum_"):
            return tag[len("stratum_") :]
    return "?"


def contact_sheet(pano_ids: list[str], predictions: dict[str, str], out: Path) -> None:
    """Render composites next to their predicted year, for reading by eye."""
    bank = Bank.load()
    fetcher = TileFetcher(TileCache(REPO / "cache"))
    anchors = [Template(array=a, style=s) for s, a in sorted(bank.anchors.items())]

    panes = []
    for pano in pano_ids:
        try:
            field = highpass(fetcher.fetch(pano, rows="top").image)
        except Exception as exc:
            print(f"  {pano}: {exc}")
            continue
        best = None
        for result in build_composites(field, anchors):
            verdict = classify(result.composite, result.instances, bank)
            if best is None or verdict.quality > best[0].quality:
                best = (verdict, result)
        if best is None:
            continue
        panes.append((f"predicted {predictions.get(pano, '?')}   {pano[:16]}", best[1].composite))

    if not panes:
        return
    width, height = 132 * 5, 32 * 5
    sheet = Image.new("L", (width, (height + 17) * len(panes)), 30)
    draw = ImageDraw.Draw(sheet)
    for i, (caption, composite) in enumerate(panes):
        cleaned = declutter(composite)
        scaled = (cleaned - cleaned.min()) / (np.ptp(cleaned) + 1e-9) * 255
        image = Image.fromarray(scaled.astype("uint8")).resize((width, height), Image.LANCZOS)
        sheet.paste(image, (0, i * (height + 17) + 15))
        draw.text((4, i * (height + 17) + 3), caption, fill=255)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"\ncontact sheet -> {out}  ({len(panes)} panoramas)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="CSV from `cr-label label`")
    parser.add_argument("--sample", type=int, default=0, help="composites to render for eye check")
    parser.add_argument("--sheet", type=Path, default=Path("audit-sheet.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    locations = load_locations(args.locations)
    rows = [r for r in csv.DictReader(args.report.open()) if r["panoId"]]
    print(f"{len(rows)} labelled panoramas\n")

    labels = Counter(r["label"] for r in rows)
    total = sum(labels.values())
    print("label distribution")
    for label, count in sorted(labels.items()):
        print(f"  {label:9s} {count:5d}  {100 * count / total:5.1f}%")

    read = sum(c for lbl, c in labels.items() if lbl.isdigit())
    print(f"\nread a year   {read}/{total}  ({100 * read / total:.1f}%)")
    print(f"None          {labels.get('None', 0)}")
    print(f"abstained     {labels.get('unknown', 0)}")

    # --- per stratum -------------------------------------------------------
    by_stratum: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_stratum[stratum_of(locations.get(row["panoId"], {}))][row["label"]] += 1
    print("\nper stratum (years read / None / abstained)")
    for name, counts in sorted(by_stratum.items()):
        years = sum(c for lbl, c in counts.items() if lbl.isdigit())
        print(f"  {name:11s} n={sum(counts.values()):5d}   "
              f"years={years:5d}  None={counts.get('None', 0):5d}  unknown={counts.get('unknown', 0):5d}")

    # --- check 1: copyright year cannot precede capture year ---------------
    violations = []
    checked = 0
    for row in rows:
        if not row["label"].isdigit():
            continue
        year = capture_year(locations.get(row["panoId"], {}))
        if year is None:
            continue
        checked += 1
        if int(row["label"]) < year:
            violations.append((row["panoId"], year, row["label"], row["confidence"]))
    print(f"\ncapture-year consistency: {len(violations)} violations in {checked} checked")
    for pano, cap, pred, conf in violations[:15]:
        print(f"  {pano[:20]} captured {cap} but read {pred} (confidence {conf})")

    # --- check 2: trekker carries no watermark -----------------------------
    trekker = [r for r in rows if stratum_of(locations.get(r["panoId"], {})) == "trekker"]
    if trekker:
        none_count = sum(1 for r in trekker if r["label"] == "None")
        print(f"\ntrekker: {none_count}/{len(trekker)} returned None "
              f"({100 * none_count / len(trekker):.1f}%)")

    # --- copyright vs capture year spread ----------------------------------
    pairs = [
        (capture_year(locations.get(r["panoId"], {})), int(r["label"]))
        for r in rows if r["label"].isdigit()
    ]
    pairs = [(c, p) for c, p in pairs if c]
    if pairs:
        lag = [p - c for c, p in pairs]
        print(f"\ncopyright minus capture year: median {int(np.median(lag))}, "
              f"range {min(lag)} to {max(lag)}")

    if args.sample:
        random.seed(args.seed)
        pool = [r["panoId"] for r in rows if r["label"].isdigit()]
        picked = random.sample(pool, min(args.sample, len(pool)))
        contact_sheet(picked, {r["panoId"]: r["label"] for r in rows}, args.sheet)

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
