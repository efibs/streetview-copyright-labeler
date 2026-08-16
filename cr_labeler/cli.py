"""Command line interface: ``cr-label label | build-bank | harvest | evaluate``."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from .bank import DEFAULT_BANK, Bank
from .build import build_seed_bank, harvest
from .classify import UNKNOWN
from .config import MissingApiKey, resolve_api_key
from .discover import BootstrapFailed
from .fetch import TileCache, TileFetcher
from .geoguessr_io import GeoGuessrMap
from .geometry import DEFAULT_ZOOM, DETECT_THRESHOLD, LABEL_ZOOM
from .labeler import Labeler, LabelResult, save_composite

log = logging.getLogger("cr_labeler")


# --- shared plumbing ------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, zoom: int = DEFAULT_ZOOM) -> None:
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK, help="template bank .npz")
    parser.add_argument("--zoom", type=int, default=zoom, help="tile zoom level")
    parser.add_argument(
        "--rows",
        choices=("top", "all"),
        default="top",
        help="'top' downloads the upper hemisphere (~96%% of watermarks, half the bytes)",
    )
    parser.add_argument(
        "--workers", type=int, default=default_workers(),
        help=f"panoramas processed in parallel (default {default_workers()} here)",
    )
    parser.add_argument("--cache", type=Path, default=None, help="on-disk tile cache directory")
    parser.add_argument(
        "--threshold", type=float, default=DETECT_THRESHOLD, help="detection NCC threshold"
    )


def default_workers() -> int:
    """Threads to run panoramas on.

    Two different curves, so the answer depends on where the work is happening.

    On the CPU the transforms contend for memory bandwidth, but most of a
    panorama is now waiting on the network, so mild oversubscription wins:
    8.11 panoramas/second at 24 threads against 7.32 at 16. That used to cost
    throughput -- the tile pool is shared now, so the number of requests in
    flight no longer scales with this and Google stops throttling us for it.

    With a GPU the transforms leave the CPU entirely and serialise on one
    device, so extra threads only queue behind each other: 11.54 at 8 threads
    against 9.38 at 16 and 10.00 at 6.
    """
    from .accel import device

    cores = os.cpu_count() or 4
    if device() is not None:
        return 8
    return max(8, min(24, cores))


class _PlainProgress:
    """Stand-in for the tqdm bar, used only if tqdm is not installed.

    Keeps a long run readable on an environment built before tqdm was a
    dependency, rather than failing on the import.
    """

    def __init__(self, total: int, disable: bool):
        self.total = total
        self.disable = disable
        self.n = 0
        self.started = time.time()
        self.postfix = ""

    def set_postfix_str(self, text: str, refresh: bool = True) -> None:
        self.postfix = text

    def update(self, n: int = 1) -> None:
        self.n += n
        if self.disable:
            return
        elapsed = time.time() - self.started
        rate = self.n / max(elapsed, 1e-6)
        remaining = (self.total - self.n) / rate if rate else 0.0
        print(
            f"\r  {self.n}/{self.total}  {rate:5.2f} pano/s  "
            f"eta {remaining / 60:5.1f} min  {self.postfix}",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if not self.disable:
            print(file=sys.stderr)

    def __enter__(self) -> _PlainProgress:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _progress_bar(total: int, disable: bool):
    """A progress bar over panoramas, with an estimate of the time left.

    ``smoothing`` is turned well down from tqdm's default.  The default weights
    the last few panoramas heavily, and per-panorama cost here is genuinely
    spiky -- a reading that escalates to zoom 4 does eight times the work of one
    that settles at zoom 2 -- so a responsive estimate would swing wildly over a
    multi-hour run.  Near zero it averages over the whole run instead, which is
    what makes the estimate worth reading.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        return _PlainProgress(total, disable)

    return tqdm(
        total=total,
        disable=disable,
        unit="pano",
        smoothing=0.02,
        bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _fetcher(args) -> TileFetcher:
    return TileFetcher(TileCache(args.cache))


def _read_pano_ids(path: Path) -> list[str]:
    """Read panorama ids from a text file, a JSON array, or a GeoGuessr map."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith(("{", "[")):
        document = json.loads(text)
        entries = (
            document.get("customCoordinates", [])
            if isinstance(document, dict)
            else document
        )
        ids = []
        for entry in entries:
            if isinstance(entry, str):
                ids.append(entry)
            elif isinstance(entry, dict) and entry.get("panoId"):
                ids.append(entry["panoId"])
        return ids
    return [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]


# --- label ----------------------------------------------------------------


def cmd_label(args) -> int:
    bank = Bank.load(args.bank)
    log.info("bank: %s", bank.summary())

    document = GeoGuessrMap.load(args.input)
    log.info("loaded %d locations from %s", len(document.locations), args.input)

    try:
        api_key = resolve_api_key(args.api_key_file)
    except MissingApiKey as exc:
        log.error("%s", exc)
        return 2
    if api_key:
        log.info("API key loaded from %s (coordinate lookup enabled)", api_key.source)

    labeler = Labeler(
        bank=bank,
        fetcher=_fetcher(args),
        api_key=api_key,
        zoom=args.zoom,
        rows=args.rows,
        threshold=args.threshold,
        escalate=not args.no_escalate,
    )

    started = time.time()
    seen: Counter[str] = Counter()

    with _progress_bar(len(document.locations), disable=args.quiet) as bar:

        def progress(result: LabelResult) -> None:
            # Only the two counts worth abandoning a long run over; the full
            # breakdown is printed at the end either way.
            seen[result.label] += 1
            if not result.ok:
                seen["_error"] += 1
            bar.set_postfix_str(
                f"unknown {seen[UNKNOWN]}, errors {seen['_error']}", refresh=False
            )
            bar.update(1)

        results = labeler.label_all(
            document.locations, workers=args.workers, progress=progress
        )
    elapsed = time.time() - started

    by_index = {r.index: r for r in results}
    for location in document.locations:
        result = by_index.get(location.index)
        if result:
            location.apply_tag(result.label)

    output = args.output or args.input.with_name(f"{args.input.stem}_tagged.json")
    document.save(output)

    if args.save_composites:
        for result in results:
            if result.composite is not None and result.pano_id:
                save_composite(
                    result.composite,
                    Path(args.save_composites) / f"{result.label}_{result.pano_id}.png",
                )

    if args.report:
        _write_report(Path(args.report), results)

    counts = Counter(r.label for r in results)
    errors = [r for r in results if not r.ok]
    print(f"\ntagged {len(results)} locations in {elapsed:.1f}s "
          f"({len(results) / max(elapsed, 1e-6):.2f} pano/s)")
    for label, count in sorted(counts.items()):
        print(f"  CR_{label:<9s} {count}")
    if errors:
        print(f"\n{len(errors)} could not be processed:")
        for result in errors[:10]:
            print(f"  row {result.index}: {result.error.splitlines()[0]}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    print(f"\nwrote {output}")
    return 0


def _write_report(path: Path, results: list[LabelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "panoId", "label", "confidence", "instances",
             "style", "quality", "digit_score", "margin", "ground_truth", "error"]
        )
        for result in results:
            verdict = result.verdict
            writer.writerow([
                result.index,
                result.pano_id or "",
                result.label,
                f"{verdict.confidence:.3f}" if verdict else "",
                verdict.instances if verdict else "",
                verdict.style if verdict else "",
                f"{verdict.quality:.3f}" if verdict else "",
                f"{verdict.digit_score:.3f}" if verdict else "",
                f"{verdict.margin:.3f}" if verdict else "",
                result.ground_truth or "",
                (result.error or "").splitlines()[0] if result.error else "",
            ])


# --- build-bank -----------------------------------------------------------


def cmd_build_bank(args) -> int:
    document = GeoGuessrMap.load(args.seed)
    labelled = {
        location.pano_id: location.ground_truth()
        for location in document.locations
        if location.pano_id and location.ground_truth()
    }
    if not labelled:
        log.error("%s carries no GT_* tags to learn from", args.seed)
        return 2
    log.info("seed set: %d labelled panoramas", len(labelled))

    try:
        bank, failures = build_seed_bank(
            labelled,
            _fetcher(args),
            zoom=args.zoom,
            rows=args.rows,
            workers=args.workers,
            seed_font=args.seed_font,
        )
    except BootstrapFailed as exc:
        log.error("%s", exc)
        return 1

    bank.save(args.bank)
    print(f"\nbank written to {args.bank}")
    print(f"  {bank.summary()}")
    for style in bank.styles:
        anchor = bank.anchors[style]
        print(f"  {style}: anchor {anchor.shape[1]}x{anchor.shape[0]} px, "
              f"years {bank.years(style)}, digit slots {bank.slots.get(style)}")
    print("\nNext: cr-label refine-styles   (splits any style holding more than one rendering)")
    if failures:
        print(f"\n{len(failures)} panoramas contributed nothing:")
        for pano_id, reason in failures[:10]:
            print(f"  {pano_id}: {reason}")
    return 0


# --- harvest --------------------------------------------------------------


def cmd_harvest(args) -> int:
    bank = Bank.load(args.bank)
    pano_ids = _read_pano_ids(args.panos)
    log.info("harvesting %d panoramas", len(pano_ids))

    report = harvest(
        pano_ids, bank, _fetcher(args), zoom=args.zoom, rows=args.rows, workers=args.workers
    )

    import numpy as np

    from .bank import average_composites
    from .signal import declutter

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stub: dict[str, str] = {}
    centroids: dict[str, np.ndarray] = {}
    for number, cluster in enumerate(report.clusters):
        if len(cluster) < args.min_cluster:
            continue
        name = f"cluster_{number:03d}_{cluster[0].style}_n{len(cluster)}"
        # Decluttered, so the centroid is already in the domain bank templates
        # are stored and compared in.
        centroid = average_composites([declutter(s.composite) for s in cluster])
        save_composite(centroid, outdir / f"{name}.png", scale=6)
        centroids[name] = centroid
        stub[name] = ""
        (outdir / f"{name}.members.txt").write_text(
            "\n".join(s.pano_id for s in cluster), encoding="utf-8"
        )

    # adopt reads these back, so the operator never re-downloads to apply labels.
    np.savez_compressed(outdir / "composites.npz", **centroids)
    (outdir / "labels.json").write_text(json.dumps(stub, indent=2), encoding="utf-8")
    print(f"\n{len(stub)} clusters written to {outdir}")
    print("Open each cluster_*.png, read the year, and fill it into labels.json.")
    print(f"Then:  cr-label adopt --clusters {outdir} --bank {args.bank}")
    if report.failures:
        print(f"\n{len(report.failures)} panoramas yielded no composite")
    return 0


def cmd_adopt(args) -> int:
    """Fold hand-labelled harvest clusters into the bank."""
    import numpy as np

    from .bank import average_composites

    bank = Bank.load(args.bank)
    outdir = Path(args.clusters)
    labels = json.loads((outdir / "labels.json").read_text(encoding="utf-8"))

    cache_path = outdir / "composites.npz"
    if not cache_path.exists():
        log.error(
            "%s is missing -- rerun harvest with the same --out directory", cache_path
        )
        return 2
    cache = np.load(cache_path)

    added = skipped = 0
    for name, value in labels.items():
        value = str(value).strip()
        if not value or not value.isdigit():
            skipped += 1
            continue
        if name not in cache.files:
            log.warning("%s has no saved centroid; skipping", name)
            continue
        style = name.split("_")[2]
        key = (style, int(value))
        existing = bank.templates.get(key)
        pool = [cache[name]] if existing is None else [existing, cache[name]]
        bank.templates[key] = average_composites(pool)
        added += 1

    # Digit slots are geometry -- fixed by the anchor and the font's pitch -- so
    # adding years does not move them. Which slots *vary* is recomputed at
    # classification time from whatever the bank holds.

    bank.save(args.bank)
    print(f"adopted {added} labelled clusters; bank now holds {bank.summary()}")
    return 0


# --- evaluate -------------------------------------------------------------


def cmd_refine_styles(args) -> int:
    """Split each style so every template in it shares one rendering."""
    from .refine import refine, slot_fit

    bank = Bank.load(args.bank)
    before = slot_fit(bank)
    refined, notes = refine(bank)
    after = slot_fit(refined)

    print(f"styles: {bank.styles}  ->  {refined.styles}\n")
    for note in notes:
        print(f"  {note}")
    print("\nfraction of cross-year disagreement captured by each style's digit slots")
    for style, value in before.items():
        print(f"  before  {style:10s} {value:.3f}")
    for style, value in after.items():
        print(f"  after   {style:10s} {value:.3f}")

    if args.dry_run:
        print("\ndry run; bank not written")
        return 0
    refined.save(args.bank)
    print(f"\nwrote {args.bank}")
    return 0


def cmd_evaluate(args) -> int:
    bank = Bank.load(args.bank)
    document = GeoGuessrMap.load(args.gt)
    truth = {
        location.index: location.ground_truth()
        for location in document.locations
        if location.ground_truth()
    }
    if not truth:
        log.error("%s carries no GT_* tags", args.gt)
        return 2

    labeler = Labeler(
        bank=bank,
        fetcher=_fetcher(args),
        api_key=None,
        zoom=args.zoom,
        rows=args.rows,
        threshold=args.threshold,
        escalate=not args.no_escalate,
    )
    started = time.time()
    results = labeler.label_all(document.locations, workers=args.workers)
    elapsed = time.time() - started

    correct = wrong = abstained = 0
    print(f"\n{'panoId':<24s} {'truth':>6s} {'pred':>8s} {'conf':>6s} {'n':>4s} {'style':>7s}  verdict")
    print("-" * 78)
    for result in results:
        expected = truth.get(result.index)
        if expected is None:
            continue
        verdict = result.verdict
        if result.label == expected:
            status, symbol = "correct", "ok"
            correct += 1
        elif result.label == UNKNOWN:
            status, symbol = "abstained", "--"
            abstained += 1
        else:
            status, symbol = f"WRONG (truth {expected})", "XX"
            wrong += 1
        print(
            f"{(result.pano_id or '?'):<24s} {expected:>6s} {result.label:>8s} "
            f"{(verdict.confidence if verdict else 0):>6.2f} "
            f"{(verdict.instances if verdict else 0):>4d} "
            f"{(verdict.style or '-' if verdict else '-'):>7s}  {symbol} {status}"
        )

    total = correct + wrong + abstained
    print("-" * 78)
    print(f"exact      {correct}/{total}")
    print(f"wrong      {wrong}/{total}")
    print(f"abstained  {abstained}/{total}")
    print(f"time       {elapsed:.1f}s ({total / max(elapsed, 1e-6):.2f} pano/s)")

    if args.save_composites:
        for result in results:
            if result.composite is not None and result.pano_id:
                expected = truth.get(result.index, "?")
                save_composite(
                    result.composite,
                    Path(args.save_composites) / f"gt{expected}_pred{result.label}_{result.pano_id}.png",
                )
    return 1 if wrong else 0


# --- entry point ----------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cr-label",
        description="Tag Street View panoramas with the copyright year visible in the imagery.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser("label", help="tag a GeoGuessr map JSON")
    label.add_argument("input", type=Path, help="input map JSON")
    label.add_argument("-o", "--output", type=Path, default=None, help="output JSON")
    label.add_argument("--report", type=Path, default=None, help="also write a CSV report")
    label.add_argument(
        "--api-key-file", type=Path, default=None,
        help="file holding a Google Maps API key (only needed for entries without a panoId)",
    )
    label.add_argument("--save-composites", type=Path, default=None,
                       help="directory to write one composite PNG per panorama")
    label.add_argument("--no-escalate", action="store_true",
                       help="do not retry unreadable panoramas over the full sphere")
    label.add_argument("--quiet", action="store_true", help="no progress output")
    _add_common(label, zoom=LABEL_ZOOM)
    label.set_defaults(func=cmd_label)

    build = subparsers.add_parser("build-bank", help="build a template bank from labelled panoramas")
    build.add_argument("--seed", type=Path, required=True, help="map JSON carrying GT_* tags")
    build.add_argument("--seed-font", type=Path, default=None,
                       help="TrueType font used to bootstrap the first template")
    _add_common(build)
    # Deliberately shares the labelling default for --rows.  A bank built over
    # the full sphere and used against the upper hemisphere reads measurably
    # worse: the templates and the composites they are matched against have to
    # be averaged over the same part of the panorama.
    build.set_defaults(func=cmd_build_bank)

    harvest_cmd = subparsers.add_parser(
        "harvest", help="cluster unlabelled panoramas into (style, year) groups to label"
    )
    harvest_cmd.add_argument("--panos", type=Path, required=True,
                             help="text file of panoIds, or a map JSON")
    harvest_cmd.add_argument("--out", type=Path, default=Path("clusters"),
                             help="directory for cluster centroids and labels.json")
    harvest_cmd.add_argument("--min-cluster", type=int, default=3,
                             help="ignore clusters smaller than this")
    _add_common(harvest_cmd)
    harvest_cmd.set_defaults(func=cmd_harvest)

    adopt = subparsers.add_parser("adopt", help="fold hand-labelled harvest clusters into the bank")
    adopt.add_argument("--clusters", type=Path, required=True, help="directory from harvest")
    _add_common(adopt)
    adopt.set_defaults(func=cmd_adopt)

    refine_cmd = subparsers.add_parser(
        "refine-styles",
        help="split bank styles so each holds a single rendering (sharper digit slots)",
    )
    refine_cmd.add_argument("--dry-run", action="store_true", help="report without writing")
    _add_common(refine_cmd)
    refine_cmd.set_defaults(func=cmd_refine_styles)

    evaluate = subparsers.add_parser("evaluate", help="score the labeller against GT_* tags")
    evaluate.add_argument("--gt", type=Path, required=True, help="map JSON carrying GT_* tags")
    evaluate.add_argument("--save-composites", type=Path, default=None)
    evaluate.add_argument("--no-escalate", action="store_true")
    _add_common(evaluate, zoom=LABEL_ZOOM)
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(message)s",
    )
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.warning("interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
