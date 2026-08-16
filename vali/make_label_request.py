"""Pick the panoramas worth labelling by hand, and write them in CR_GT.json format.

Labelling at random is mostly wasted effort: ~87% of the test set is read
correctly and easily, so a random sample spends almost all of its budget
confirming what is already known.  It is still needed for one thing -- an
*unbiased* accuracy figure -- so a random block comes first and the targeted
blocks follow.

The targeted blocks aim at where the remaining errors must be:

* ``unknown``  -- abstentions.  Is the gate too tight, or are these genuinely
  unreadable?  Only a human can say.
* ``None``     -- the class with no independent validation.  Measured on a small
  sample, some of these are missed watermarks rather than absent ones.
* confusable   -- years whose templates are >96% similar (2011/2012, 2016/2018,
  2018/2019).  This is the only place a *wrong* year can survive the gates.
* low-confidence -- the tail where errors concentrate, if they concentrate.

No prediction is written into the file.  The point is an independent reading;
predictions are joined back afterwards on panoId from the report CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

# Year pairs whose bank templates are near-identical over the digit slots, so a
# misread can survive the per-digit gate.  Measured on the shipped bank:
# 2021/2022 = 0.967, 2016/2018 = 0.963, 2011/2012 = 0.968.
CONFUSABLE = [(2021, 2022), (2016, 2018), (2011, 2012), (2012, 2013), (2025, 2026), (2009, 2010)]

# Years only recently added, from few panoramas, so their templates are the
# least well established in the bank.
THIN = {2009, 2010, 2012, 2014, 2015}

# Looking up: the watermark is most legible against flat sky, which is also
# where ~96% of instances land.
LABEL_PITCH = 40.0


def country_of(row: dict) -> str:
    for tag in row.get("extra", {}).get("tags", []):
        if len(tag) == 2 and tag.isalpha() and tag.isupper():
            return tag
    return "??"


def stratum_of(row: dict) -> str:
    for tag in row.get("extra", {}).get("tags", []):
        if tag.startswith("stratum_"):
            return tag[len("stratum_") :]
    return "?"


def pano_date(row: dict) -> str | None:
    """Capture date as ``YYYY-MM``.

    map-making.app validates ``panoDate`` against ``\\d{4}-\\d{2}`` and rejects
    the whole import otherwise, so a bare year is not good enough.  Vali emits
    the year and month as separate tags (``["2008", "8", "IT", ...]``), which
    have to be recombined and the month zero-padded.
    """
    tags = row.get("extra", {}).get("tags", [])
    year = next((t for t in tags if t.isdigit() and 2000 <= int(t) <= 2100), None)
    month = next((t for t in tags if t.isdigit() and 1 <= int(t) <= 12 and len(t) <= 2), None)
    if not year:
        return None
    return f"{year}-{int(month):02d}" if month else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locations", type=Path, default=Path("datasets/test.json"))
    parser.add_argument("--report", type=Path, default=Path("datasets/test-report.csv"))
    parser.add_argument("--out", type=Path, default=Path("datasets/to_label.json"))
    parser.add_argument("--random", type=int, default=60, help="unbiased baseline block")
    parser.add_argument("--per-target", type=int, default=35, help="size of each targeted block")
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument(
        "--exclude", type=Path, nargs="*", default=[],
        help="files whose panoIds must not be re-issued (earlier label requests)",
    )
    parser.add_argument(
        "--per-country", type=int, default=0,
        help="if set, also take this many from every country present, so no "
             "country is left unrepresented by proportional sampling",
    )
    parser.add_argument("--name", default="CR label request")
    args = parser.parse_args()

    locations = {
        r["panoId"]: r
        for r in json.loads(args.locations.read_text(encoding="utf-8"))["customCoordinates"]
        if r.get("panoId")
    }
    report = [r for r in csv.DictReader(args.report.open()) if r["panoId"] in locations]

    already: set[str] = set()
    for path in args.exclude:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = doc["customCoordinates"] if isinstance(doc, dict) else doc
        already |= {r["panoId"] for r in rows if r.get("panoId")}
    if already:
        before = len(report)
        report = [r for r in report if r["panoId"] not in already]
        print(f"excluded {before - len(report)} panoramas already issued for labelling\n")

    rng = random.Random(args.seed)

    chosen: list[tuple[str, str]] = []  # (panoId, why)
    taken: set[str] = set()

    def take(pool: list[dict], count: int, why: str) -> None:
        pool = [r for r in pool if r["panoId"] not in taken]
        rng.shuffle(pool)
        for row in pool[:count]:
            taken.add(row["panoId"])
            chosen.append((row["panoId"], why))

    # 1. Unbiased baseline -- keep first so a partial pass is still usable.
    take(list(report), args.random, "random")

    # 2. Every country, so proportional sampling cannot leave one unrepresented.
    if args.per_country:
        by_country: dict[str, list[dict]] = defaultdict(list)
        for row in report:
            by_country[country_of(locations[row["panoId"]])].append(row)
        for code in sorted(by_country):
            take(by_country[code], args.per_country, f"country-{code}")

    # 3. Abstentions: every one, they are few and each is informative.
    take([r for r in report if r["label"] == "unknown"], args.per_target, "abstained")

    # 4. None, excluding trekker (trekker is mostly genuinely unstamped).
    take(
        [r for r in report if r["label"] == "None"
         and stratum_of(locations[r["panoId"]]) != "trekker"],
        args.per_target, "called-None",
    )

    # 5. Years that sit either side of a confusable pair.
    risky = {str(y) for pair in CONFUSABLE for y in pair}
    take([r for r in report if r["label"] in risky], args.per_target, "confusable-year")

    # 6. Years whose templates were built from very few panoramas.
    take([r for r in report if r["label"].isdigit() and int(r["label"]) in THIN],
         args.per_target, "thin-template")

    # 7. Gen 1 specifically, which the first request drew entirely from one country.
    take([r for r in report if stratum_of(locations[r["panoId"]]) == "gen1"],
         args.per_target, "gen1")

    # 8. The low-confidence tail of the year reads.
    reads = sorted(
        (r for r in report if r["label"].isdigit()),
        key=lambda r: float(r["confidence"] or 0),
    )
    take(reads, args.per_target, "low-confidence")

    entries = []
    for pano_id, _why in chosen:
        source = locations[pano_id]
        extra: dict[str, object] = {"tags": []}
        date = pano_date(source)
        if date:
            extra["panoDate"] = date
        entries.append({
            "lat": source["lat"],
            "lng": source["lng"],
            "heading": source.get("heading", 0),
            "pitch": LABEL_PITCH,
            "zoom": 0,
            "panoId": pano_id,
            "countryCode": None,
            "stateCode": None,
            "extra": extra,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(e, ensure_ascii=False, separators=(",", ":")) for e in entries]
    args.out.write_text(
        '{"name":"' + args.name + '","customCoordinates":[\n' + ",\n".join(lines) + "\n]}",
        encoding="utf-8",
    )

    counts: dict[str, int] = defaultdict(int)
    for _, why in chosen:
        counts[why] += 1
    print(f"wrote {len(entries)} locations to {args.out}\n")
    order = [
        "random",
        *sorted(k for k in counts if k.startswith("country-")),
        "gen1", "abstained", "called-None", "confusable-year",
        "thin-template", "low-confidence",
    ]
    start = 1
    for why in order:
        n = counts.get(why, 0)
        if n:
            print(f"  rows {start:>4}-{start + n - 1:<4} {n:>4}  {why}")
            start += n
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
