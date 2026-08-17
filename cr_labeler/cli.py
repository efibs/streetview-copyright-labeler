"""Command line interface: ``cr-label label | build-bank | harvest | evaluate``."""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import json
import logging
import os
import signal
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

from .bank import DEFAULT_BANK, Bank
from .build import build_seed_bank, harvest
from .checkpoint import Checkpoint, CheckpointMismatch
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
    Both were re-measured after BLAS stopped competing for the same cores --
    see :func:`cr_labeler._limit_math_library_threads` -- which moved both
    optima, because the constraint they were originally fitted to was an
    artefact.

    With a GPU, 12: measured 23.53 panoramas/second against 19.05 at 8, 21.6 at
    16 and 22.2 at 24. Eight was right only while every worker was fighting
    BLAS for a core.

    On the CPU, 16: 16.67 against 15.48 at 24 and 13.33 at 32. The transforms
    contend for memory bandwidth, so past the core count it goes backwards.
    """
    from .accel import device

    cores = os.cpu_count() or 4
    if device() is not None:
        return 12
    return max(8, min(16, cores))


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

    output = args.output or args.input.with_name(f"{args.input.stem}_tagged.json")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else Path(f"{output}.progress")
    checkpoint = Checkpoint(
        checkpoint_path,
        header={
            "input": str(args.input.resolve()),
            "rows": len(document.locations),
            "zoom": args.zoom,
            "rows_mode": args.rows,
            "escalate": not args.no_escalate,
        },
    )

    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("--restart: discarded %s", checkpoint_path)

    try:
        already = checkpoint.load()
    except CheckpointMismatch as exc:
        log.error("%s", exc)
        log.error("pass --restart to discard it, or --checkpoint PATH to keep them apart")
        return 2

    counts: Counter[str] = Counter()
    # Only the first few are ever printed, so only the first few are kept: if
    # something systematic goes wrong -- the network drops, Google starts
    # refusing -- every one of 900k rows fails, and holding all those messages
    # to report ten of them is exactly the wrong thing to do while memory is
    # already the concern.
    failures = 0
    examples: list[tuple[int, str]] = []
    KEEP_EXAMPLES = 20

    def note_failure(index: int, message: str) -> None:
        nonlocal failures
        failures += 1
        if len(examples) < KEEP_EXAMPLES:
            examples.append((index, message))

    for record in already.values():
        counts[record.label] += 1
        if record.error:
            note_failure(record.index, record.error)
    if already:
        print(f"resuming: {len(already)} of {len(document.locations)} already done "
              f"in {checkpoint_path}")

    by_index = {loc.index: loc for loc in document.locations}
    for record in already.values():
        location = by_index.get(record.index)
        if location:
            location.apply_tag(record.label)

    todo = [loc for loc in document.locations if loc.index not in already]
    if not todo:
        print("nothing left to label")

    stopping = _install_stop_handler()
    disk = _DiskGuard(
        [checkpoint_path.parent, Path(args.cache) if args.cache else None],
        floor_gb=args.min_free_gb,
    )
    disk.warn_if_tight(len(todo), cache_on=bool(args.cache))
    disk.check(force=True)
    if disk.exhausted():
        log.error("not starting: %s", disk.reason)
        log.error("free some space, or lower --min-free-gb")
        return 1

    started = time.time()
    done_now = 0
    report = _ReportWriter(Path(args.report)) if args.report else None

    with checkpoint, (report or nullcontext()):
        with _progress_bar(len(document.locations), disable=args.quiet) as bar:
            bar.update(len(already))

            def should_stop() -> bool:
                return stopping() or disk.exhausted()

            for result in labeler.label_stream(
                todo, workers=args.workers, stop=should_stop
            ):
                # Everything durable happens before the composite is dropped,
                # and the composite is dropped before the next result arrives --
                # which is what keeps memory flat over 900k rows.
                checkpoint.record(result.index, result.label, result.pano_id, result.error)
                location = by_index.get(result.index)
                if location:
                    location.apply_tag(result.label)
                if report:
                    report.write(result)
                if args.save_composites and result.composite is not None and result.pano_id:
                    save_composite(
                        result.composite,
                        Path(args.save_composites) / f"{result.label}_{result.pano_id}.png",
                    )
                result.composite = None

                counts[result.label] += 1
                if not result.ok:
                    note_failure(result.index, result.error or "")
                done_now += 1
                if done_now % 2000 == 0:
                    _reclaim()
                disk.check()
                bar.set_postfix_str(
                    f"unknown {counts[UNKNOWN]}, errors {failures}", refresh=False
                )
                bar.update(1)

    elapsed = time.time() - started
    labelled = len(already) + done_now
    interrupted = labelled < len(document.locations)

    document.save(output)

    print(f"\nlabelled {done_now} locations in {elapsed:.1f}s "
          f"({done_now / max(elapsed, 1e-6):.2f} pano/s)")
    for label, count in sorted(counts.items()):
        print(f"  CR_{label:<9s} {count}")
    if failures:
        print(f"\n{failures} could not be processed:")
        for index, message in examples[:10]:
            print(f"  row {index}: {message.splitlines()[0] if message else ''}")
        if failures > 10:
            print(f"  ... and {failures - 10} more")

    print(f"\nwrote {output}")
    if interrupted:
        remaining = len(document.locations) - labelled
        print(f"\n{remaining} locations still unlabelled -- {disk.reason or 'run was interrupted'}.")
        print("The rest are saved. Run the same command again to carry on where it stopped:")
        print(f"  progress is in {checkpoint_path}")
        return 1

    print(f"complete; {checkpoint_path.name} can be deleted")
    return 0


class _ReportWriter:
    """Streams the per-panorama CSV instead of holding every row to the end.

    Appends, so a resumed run adds to the report its earlier attempt started
    rather than truncating it.
    """

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size == 0
        self._handle = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._handle)
        if fresh:
            self._writer.writerow(REPORT_COLUMNS)

    def write(self, result: LabelResult) -> None:
        self._writer.writerow(_report_row(result))

    def __enter__(self) -> _ReportWriter:
        return self

    def __exit__(self, *exc) -> None:
        self._handle.close()


def _reclaim() -> None:
    """Collect cycles and hand freed arenas back to the OS.

    Resident memory here is dominated by the panoramas in flight, which is set
    by ``--workers`` and not by how many rows have been processed -- measured
    flat at 4.58, 4.54 and 4.91 GB across the three thirds of a run.  What can
    still creep is glibc holding freed arenas: this work allocates and frees
    multi-megabyte buffers on many threads, which is the pattern that fragments
    them.  Twice an hour at full speed, and a few milliseconds each time.

    ``malloc_trim`` is glibc's, so its absence is not an error anywhere else.
    """
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _install_stop_handler():
    """Turn the first Ctrl-C into a clean stop, and the second into a quit.

    A run this long will sometimes need stopping on purpose.  The default
    KeyboardInterrupt would tear down mid-write; this lets the work in flight
    finish and be recorded, so stopping costs seconds rather than the batch.
    """
    asked = False

    def stopping() -> bool:
        return asked

    def handler(signum, frame):
        nonlocal asked
        if asked:  # they mean it
            raise KeyboardInterrupt
        asked = True
        print(
            "\nstopping -- finishing what is in flight, then saving. "
            "Ctrl-C again to quit now.",
            file=sys.stderr,
        )

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except ValueError:  # not on the main thread; leave the default
            pass
    return stopping


class _DiskGuard:
    """Stops the run cleanly before the disk fills, rather than after.

    Running out of space mid-write is the failure that corrupts things: the
    checkpoint, the cache and the output all want the same volume.  Stopping
    while there is still room leaves every one of them intact and resumable.
    """

    # Seconds between checks.  Rate-limiting by row count was wrong: a disk can
    # fill in seconds, and a short run could finish having never looked.
    INTERVAL = 5.0

    def __init__(self, paths, floor_gb: float):
        seen = {}
        for path in paths:
            if path is None:
                continue
            try:
                resolved = Path(path).resolve()
                resolved.mkdir(parents=True, exist_ok=True)
                seen[os.stat(resolved).st_dev] = resolved
            except OSError:
                continue
        self.paths = list(seen.values())
        self.floor = floor_gb * 1e9
        self.reason: str | None = None
        self._last_check = 0.0

    def free(self) -> float:
        lowest = float("inf")
        for path in self.paths:
            try:
                stats = os.statvfs(path)
            except OSError:
                continue
            lowest = min(lowest, stats.f_bavail * stats.f_frsize)
        return 0.0 if lowest == float("inf") else lowest

    def warn_if_tight(self, todo: int, cache_on: bool) -> None:
        free = self.free()
        if not cache_on or not todo:
            return
        # Measured over the tiles this machine has cached: ~37 KB each, and
        # ~9.7 tiles per panorama once escalation is counted.
        projected = todo * 9.7 * 37_000
        if projected > free * 0.8:
            print(
                f"\nwarning: caching tiles for {todo} panoramas needs roughly "
                f"{projected / 1e9:.0f} GB and only {free / 1e9:.0f} GB is free.\n"
                "         The cache only helps runs that revisit the same panoramas, "
                "and resuming\n"
                "         does not revisit them -- for a single large pass, drop "
                "--cache.\n"
                f"         The run will stop cleanly and stay resumable at "
                f"{self.floor / 1e9:.0f} GB free.",
                file=sys.stderr,
            )

    def check(self, force: bool = False) -> None:
        if self.reason:
            return
        now = time.time()
        if not force and now - self._last_check < self.INTERVAL:
            return
        self._last_check = now
        free = self.free()
        if free < self.floor:
            self.reason = f"stopped with only {free / 1e9:.1f} GB of disk free"
            log.error("%s -- saving and stopping while everything is still intact", self.reason)

    def exhausted(self) -> bool:
        return self.reason is not None


REPORT_COLUMNS = [
    "index", "panoId", "label", "confidence", "instances",
    "style", "quality", "digit_score", "margin", "ground_truth", "error",
]


def _report_row(result: LabelResult) -> list:
    verdict = result.verdict
    return [
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
    ]


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
    label.add_argument(
        "--checkpoint", type=Path, default=None,
        help="progress file for resuming (default: alongside the output, .progress)",
    )
    label.add_argument(
        "--restart", action="store_true",
        help="discard any existing progress and label every location again",
    )
    label.add_argument(
        "--min-free-gb", type=float, default=5.0,
        help="stop cleanly, still resumable, when disk free space falls below this",
    )
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
