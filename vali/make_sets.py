"""Generate stratified train and test location sets with Vali.

Why strata: the label we care about is the *copyright* year, which Vali cannot
filter on -- it is not in any metadata, it is in the pixels. What Vali *can*
filter on is the capture year, and capture age is the strongest available proxy
for copyright age: imagery that was never re-processed keeps its original stamp.
Sampling evenly across capture-year bands is therefore how we reach the rare old
copyright years instead of drawing 60% `2026` as a naive sample does.

The bands double as generation proxies (Vali has no `Generation` property), and
a separate trekker stratum supplies the no-watermark class.

Train and test come from one generated pool per stratum, split by a hash of the
panorama id. That guarantees the two sets are disjoint and identically
distributed -- two independent Vali runs would give neither.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = Path.home() / "vali-data" / "Vali"

# Include every kind of coverage. Vali otherwise applies a default
# DescriptionLength filter that drops trekker.
ALL_COVERAGE = "DescriptionLength gt -1"

# Capture-year bands. These are *coverage-age* bands, not camera generations.
# Vali has no generation property (confirmed against v1.0.0: "Unknown property
# 'Generation'"), and capture year does not stand in for one -- screening the
# `oldest` stratum by true panorama resolution found 0 of 396 to be first
# generation. Vali cannot reach Gen 1 at all: asked for the oldest panorama
# within 20 km of a known Gen 1 location it returns nothing older than 2010.
# Gen 1 is collected separately, by resolution, in cr_labeler/panometa.py.
# Ordered rarest first. Strata are de-duplicated against each other, and with
# an "Oldest" verification strategy two different capture-year bands can resolve
# to the *same* panorama -- so whichever stratum runs first claims it. Running
# the plentiful modern coverage first would starve the scarce old coverage that
# is the whole reason for stratifying.
STRATA = [
    # name        expression                                     countries
    ("oldest",      "Year lte 2009",                               "GEN1"),
    ("old",      "Year gte 2010 and Year lte 2014",             "ALL"),
    ("mid","Year gte 2015 and Year lte 2018",             "ALL"),
    ("recent", "Year gte 2019 and Year lte 2021",             "ALL"),
    ("trekker",   "IsScout eq true",                             "ALL"),
    ("newest",      "Year gte 2022",                               "ALL"),
]

# The earliest Street View countries, where the oldest capture dates survive.
GEN1_COUNTRIES = ["AU", "US", "NZ", "JP", "MX", "ES", "FR", "IT"]

# Old imagery is over-sampled on purpose: it is the only source of the old
# copyright years, and it is where a naive sample is thinnest.
# Vali is asked for more than we need because pano verification and the
# cross-stratum de-duplication both shrink the result.
OVERSHOOT = 2.5

WEIGHTS = {
    "oldest": 1.0,
    "old": 2.0,
    "mid": 2.0,
    "recent": 1.5,
    "newest": 1.0,
    "trekker": 1.0,
}

# Older imagery is sparser, so locations have to sit closer together to reach
# the requested count at all.
# Kept low deliberately. These filters are restrictive enough that spacing, not
# the goal, becomes the binding constraint; measured yield was identical at
# 1000 m and 200 m because availability -- not distance -- was the ceiling.
MIN_DISTANCE = {
    "oldest": 100,
    "old": 150,
    "mid": 200,
    "recent": 200,
    "newest": 500,
    "trekker": 100,
}

# Which panorama the verifier picks at each location. Old strata ask for the
# oldest surviving panorama, which is the one still carrying an old stamp.
STRATEGY = {
    "oldest": "Oldest",
    "old": "Oldest",
    "mid": "Oldest",
    "recent": "Random",
    "newest": "Newest",
    "trekker": "Newest",
}


def available_countries() -> list[str]:
    if not DATA_DIR.exists():
        sys.exit(f"no Vali data at {DATA_DIR}; run 'vali download --country XX' first")
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir() and len(p.name) == 2)


def write_config(name: str, countries: list[str], expression: str, goal: int) -> Path:
    config = {
        "countryCodes": countries,
        "distributionStrategy": {
            "key": "FixedCountByMaxMinDistance",
            "locationCountGoal": goal,
            "minMinDistance": MIN_DISTANCE[name],
        },
        "globalLocationFilter": f"{ALL_COVERAGE} and ({expression})",
        "output": {
            "panoVerificationStrategy": STRATEGY[name],
            # Year and IsScout are carried through so the generated set can be
            # audited afterwards without re-querying Google.
            "locationTags": ["Year", "Month", "CountryCode", "IsScout"],
        },
    }
    path = HERE / f"stratum-{name}.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def run_vali(config: Path) -> list[dict]:
    """Run `vali generate`, through a pty because Vali polls the console."""
    out = config.with_name(config.stem + "-locations.json")
    out.unlink(missing_ok=True)
    subprocess.run(
        ["script", "-qec", f"vali generate --file {config.name}", "/dev/null"],
        cwd=HERE, capture_output=True, text=True, timeout=7200,
    )
    if not out.exists():
        return []
    return json.loads(out.read_text(encoding="utf-8"))


def assign(pano_id: str, test_fraction: float) -> str:
    """Deterministic train/test assignment from the panorama id."""
    digest = hashlib.sha256(pano_id.encode()).digest()
    return "test" if int.from_bytes(digest[:4], "big") / 2**32 < test_fraction else "train"


def save(path: Path, name: str, rows: list[dict]) -> None:
    lines = [json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows]
    header = json.dumps({"name": name}, ensure_ascii=False, separators=(",", ":"))[1:-1]
    path.write_text(
        "{" + header + ',"customCoordinates":[\n' + ",\n".join(lines) + "\n]}",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total", type=int, default=3600, help="locations across both sets")
    parser.add_argument("--test-fraction", type=float, default=0.35)
    parser.add_argument("--out-dir", type=Path, default=HERE.parent / "datasets" / "generated")
    parser.add_argument(
        "--countries",
        default="",
        help="comma-separated country codes; default is every downloaded country. "
             "Pass an explicit list to keep a run reproducible, or to skip a "
             "country whose download is still in flight.",
    )
    args = parser.parse_args()

    if not shutil.which("vali"):
        sys.exit("vali not on PATH (try: export PATH=$PATH:$HOME/.dotnet/tools)")

    countries = (
        [c.strip().upper() for c in args.countries.split(",") if c.strip()]
        if args.countries
        else available_countries()
    )
    missing = [c for c in countries if not (DATA_DIR / c).is_dir()]
    if missing:
        sys.exit(f"no downloaded data for: {' '.join(missing)}")
    print(f"using {len(countries)} countries: {' '.join(countries)}\n")

    weight_total = sum(WEIGHTS.values())
    pool: dict[str, dict] = {}
    per_stratum: dict[str, int] = {}

    for name, expression, scope in STRATA:
        selected = (
            [c for c in GEN1_COUNTRIES if c in countries] if scope == "GEN1" else countries
        )
        if not selected:
            print(f"{name:11s} skipped (no downloaded country qualifies)")
            continue

        # Over-request: verification drops locations, and the cross-stratum
        # de-duplication above removes more. Vali returns what it can find.
        goal = max(150, round(OVERSHOOT * args.total * WEIGHTS[name] / weight_total))
        config = write_config(name, selected, expression, goal)
        print(f"{name:11s} goal={goal:5d}  {STRATEGY[name]:7s}  {len(selected)} countries ... ", end="", flush=True)

        rows = run_vali(config)
        added = 0
        for row in rows:
            pano = row.get("panoId")
            if not pano or pano in pool:  # keep strata disjoint as well
                continue
            row.setdefault("extra", {}).setdefault("tags", [])
            row["extra"]["tags"].append(f"stratum_{name}")
            pool[pano] = row
            added += 1
        per_stratum[name] = added
        print(f"{added} new locations")

    if not pool:
        sys.exit("Vali produced no locations; check the configs and downloaded data")

    train = [r for p, r in pool.items() if assign(p, args.test_fraction) == "train"]
    test = [r for p, r in pool.items() if assign(p, args.test_fraction) == "test"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save(args.out_dir / "train.json", "CR train", train)
    save(args.out_dir / "test.json", "CR test", test)

    print(f"\ntrain {len(train)}   test {len(test)}   (unique panoIds {len(pool)})")
    assert not ({r["panoId"] for r in train} & {r["panoId"] for r in test}), "sets overlap"
    print("disjoint: yes")
    for name, count in per_stratum.items():
        in_train = sum(1 for r in train if f"stratum_{name}" in r["extra"]["tags"])
        print(f"  {name:11s} {count:5d}  ->  train {in_train:5d}  test {count - in_train:5d}")
    print(f"\nwrote {args.out_dir}/train.json and {args.out_dir}/test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
