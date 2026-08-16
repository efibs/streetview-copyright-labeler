"""Accuracy report against hand-labelled ground truth.

The labelled set is deliberately *not* a random sample -- most of it was chosen
to stress the places errors were expected to hide -- so a single headline number
would understate real performance badly.  The report therefore separates:

* the **random blocks**, which are an unbiased sample and the only rows that
  support a real-world accuracy figure;
* the **targeted blocks**, which are adversarial by construction and answer a
  different question: where does it actually break?

Abstentions (`unknown`) are reported apart from errors throughout.  Declining to
answer and answering wrongly are not the same failure, and the design trades the
first to avoid the second.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

UNKNOWN = "unknown"
NONE = "None"

# Row ranges of each request file, in the order make_label_request.py emitted
# them. The random blocks come first in each file by design.
BLOCKS: dict[str, list[tuple[int, int, str]]] = {
    "to_label.json": [
        (0, 60, "random"), (60, 89, "abstained"), (89, 124, "called-None"),
        (124, 159, "confusable-year"), (159, 194, "low-confidence"),
    ],
    "to_label_2.json": [
        (0, 50, "random"), (50, 143, "per-country"), (143, 173, "gen1"),
        (173, 203, "abstained"), (203, 233, "called-None"),
        (233, 263, "confusable-year"), (263, 293, "thin-template"),
        (293, 323, "low-confidence"),
    ],
    "to_label_gen1.json": [(0, 26, "gen1")],
}

# per-country was drawn without regard to difficulty, so it is unbiased too.
UNBIASED = {"random", "per-country"}


def block_of(pano_id: str, requests_dir: Path) -> str:
    for filename, blocks in BLOCKS.items():
        path = requests_dir / filename
        if not path.exists():
            continue
        ids = [
            r["panoId"]
            for r in json.loads(path.read_text(encoding="utf-8"))["customCoordinates"]
        ]
        if pano_id not in ids:
            continue
        index = ids.index(pano_id)
        for start, stop, name in blocks:
            if start <= index < stop:
                return name
    return "?"


def truth_of(row: dict) -> str | None:
    tags = row.get("extra", {}).get("tags", [])
    for tag in tags:
        value = tag[3:] if tag.startswith("GT_") else tag
        if value.isdigit() or value == NONE:
            return value
    return None


def bar(value: float, width: int = 28) -> str:
    filled = round(value * width)
    return "#" * filled + "." * (width - filled)


def section(title: str) -> None:
    print(f"\n{title}\n{'=' * len(title)}")


def score(pairs: list[tuple[str, str]]) -> dict[str, float | int]:
    """pairs of (truth, predicted)."""
    total = len(pairs)
    answered = [(t, p) for t, p in pairs if p != UNKNOWN]
    correct = sum(1 for t, p in answered if t == p)
    wrong = len(answered) - correct
    return {
        "total": total,
        "answered": len(answered),
        "abstained": total - len(answered),
        "correct": correct,
        "wrong": wrong,
        "precision": correct / len(answered) if answered else 0.0,
        "coverage": len(answered) / total if total else 0.0,
        "accuracy": correct / total if total else 0.0,
    }


def show(name: str, s: dict) -> None:
    print(f"  {name:18s} n={s['total']:4d}  answered={s['answered']:4d}  "
          f"correct={s['correct']:4d}  wrong={s['wrong']:3d}  abstained={s['abstained']:3d}  "
          f"precision={s['precision']:6.1%}  coverage={s['coverage']:6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labelled", type=Path, default=Path("datasets/labeled.json"))
    parser.add_argument("--report", type=Path, default=Path("/tmp/eval.csv"))
    parser.add_argument("--requests-dir", type=Path, default=Path("datasets"))
    args = parser.parse_args()

    truth = {}
    for row in json.loads(args.labelled.read_text(encoding="utf-8"))["customCoordinates"]:
        value = truth_of(row)
        if value:
            truth[row["panoId"]] = value

    predicted = {r["panoId"]: r["label"] for r in csv.DictReader(args.report.open())}
    confidence = {r["panoId"]: float(r["confidence"] or 0) for r in csv.DictReader(args.report.open())}

    common = [p for p in truth if p in predicted]
    pairs = [(truth[p], predicted[p]) for p in common]

    section("Overall")
    overall = score(pairs)
    print(f"  {len(common)} panoramas with both a hand label and a prediction")
    show("all rows", overall)
    print("\n  NOTE: the set is mostly adversarial by design; see the split below.")

    # --- unbiased vs targeted ---------------------------------------------
    blocks = {p: block_of(p, args.requests_dir) for p in common}
    section("Unbiased sample (the real-world estimate)")
    unbiased = [(truth[p], predicted[p]) for p in common if blocks[p] in UNBIASED]
    show("random+country", score(unbiased))
    by_block = defaultdict(list)
    for p in common:
        by_block[blocks[p]].append((truth[p], predicted[p]))
    for name in ("random", "per-country"):
        if by_block.get(name):
            show(f"  {name}", score(by_block[name]))

    section("Targeted blocks (chosen to be hard -- not representative)")
    for name in ("gen1", "abstained", "called-None", "confusable-year",
                 "thin-template", "low-confidence", "?"):
        if by_block.get(name):
            show(name, score(by_block[name]))

    # --- None vs year ------------------------------------------------------
    section("No-watermark class")
    real_none = [(t, p) for t, p in pairs if t == NONE]
    said_none = [(t, p) for t, p in pairs if p == NONE]
    hit = sum(1 for t, p in real_none if p == NONE)
    print(f"  truly None      : {len(real_none):4d}   detected {hit:4d}  "
          f"recall {hit / max(len(real_none), 1):.1%}")
    print(f"  predicted None  : {len(said_none):4d}   correct  "
          f"{sum(1 for t, _ in said_none if t == NONE):4d}  "
          f"precision {sum(1 for t, _ in said_none if t == NONE) / max(len(said_none), 1):.1%}")
    missed = Counter(p for t, p in real_none if p != NONE)
    if missed:
        print(f"  None called something else: {dict(missed)}")
    false_none = Counter(t for t, p in said_none if t != NONE)
    if false_none:
        print(f"  watermarked but called None: {dict(false_none)}")

    # --- year errors -------------------------------------------------------
    section("Year reads")
    years = [(t, p) for t, p in pairs if t.isdigit()]
    answered = [(t, p) for t, p in years if p not in (UNKNOWN,)]
    numeric = [(t, p) for t, p in answered if p.isdigit()]
    exact = sum(1 for t, p in numeric if t == p)
    off_by = Counter(int(p) - int(t) for t, p in numeric if t != p)
    print(f"  panoramas with a real year : {len(years)}")
    print(f"  answered with a year       : {len(numeric)}  ({len(numeric) / max(len(years), 1):.1%})")
    print(f"  exact                      : {exact}  ({exact / max(len(numeric), 1):.1%} of answered)")
    if off_by:
        print(f"  wrong by                   : {dict(sorted(off_by.items()))}")
        adjacent = sum(c for d, c in off_by.items() if abs(d) == 1)
        print(f"  of the wrong ones, {adjacent}/{sum(off_by.values())} are off by exactly one year")

    section("Per-year detail (truth -> what was predicted)")
    per = defaultdict(Counter)
    for t, p in pairs:
        per[t][p] += 1
    print(f"  {'truth':7s} {'n':>4s} {'exact':>6s} {'wrong':>6s} {'abst':>5s}  accuracy")
    for t in sorted(per, key=lambda x: (x == NONE, x)):
        c = per[t]
        n = sum(c.values())
        ex = c.get(t, 0)
        ab = c.get(UNKNOWN, 0)
        print(f"  {t:7s} {n:4d} {ex:6d} {n - ex - ab:6d} {ab:5d}  {bar(ex / n)} {ex / n:5.1%}")

    # --- confusion matrix --------------------------------------------------
    section("Confusion matrix (rows = truth, columns = predicted)")
    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs if p != UNKNOWN},
                    key=lambda x: (x == NONE, x))
    header = "".join(f"{name[-2:] if name.isdigit() else 'No':>4s}" for name in labels)
    print(f"  {'':7s}{header}{'  unk':>5s}")
    for t in labels:
        if t not in per:
            continue
        cells = "".join(f"{per[t].get(name, 0) or '.':>4}" for name in labels)
        print(f"  {t:7s}{cells}{per[t].get(UNKNOWN, 0) or '.':>5}")

    # --- abstentions -------------------------------------------------------
    section("Abstentions")
    abst = [(t, p) for t, p in pairs if p == UNKNOWN]
    print(f"  {len(abst)} of {len(pairs)} ({len(abst) / len(pairs):.1%})")
    print(f"  their true labels: {dict(sorted(Counter(t for t, _ in abst).items()))}")
    print("  (an abstention is a refusal to guess, not an error)")

    # --- confidence --------------------------------------------------------
    section("Does confidence track correctness?")
    buckets = defaultdict(lambda: [0, 0])
    for p in common:
        if predicted[p] == UNKNOWN:
            continue
        b = min(int(confidence[p] * 10) / 10, 0.9)
        buckets[b][0] += 1
        buckets[b][1] += truth[p] == predicted[p]
    for b in sorted(buckets):
        n, ok = buckets[b]
        print(f"  confidence {b:.1f}-{b + 0.1:.1f}: n={n:4d}  correct={ok:4d}  {bar(ok / n)} {ok / n:5.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
