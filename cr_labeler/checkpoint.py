"""Crash-resumable record of what a run has already labelled.

A 900k-location run takes most of a day, which is long enough that finishing is
not the only thing worth designing for.  Machines run out of memory, disks fill,
sessions drop, laptops sleep.  Losing twenty hours to any of those is the real
failure, not the crash itself.

So every result is appended to a sidecar file the moment it arrives, one JSON
object per line.  A run that dies at 600k has 600k rows on disk; starting the
same command again skips them and carries on.  Nothing needs to be recovered by
hand, and the file is plain text, so a half-written last line from a hard kill
costs one panorama rather than the file.

The format is deliberately dull -- append-only lines, no index, no database --
because the failure it has to survive is the process disappearing mid-write.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Bumped only if the line format changes incompatibly; an older file is then
# refused rather than misread.
FORMAT = 1


class CheckpointMismatch(RuntimeError):
    """The checkpoint on disk belongs to a different run."""


@dataclass(slots=True)
class Completed:
    """What a finished row contributes to the final document.

    ``slots`` because a resumed 900k run holds one of these per finished row,
    and the per-instance ``__dict__`` it removes is most of their weight.
    """

    index: int
    label: str
    pano_id: str | None
    error: str | None


class Checkpoint:
    """Append-only log of completed rows, with matching resume.

    Opened for one run and closed at the end.  ``load`` is a separate step so
    the caller can report how much is being skipped before any work starts.
    """

    def __init__(self, path: Path, header: dict[str, Any]):
        self.path = Path(path)
        self.header = header
        self.done: dict[int, Completed] = {}
        self._handle = None
        self._since_sync = 0

    # ---- resume -----------------------------------------------------------

    def load(self) -> dict[int, Completed]:
        """Read an existing checkpoint, if it belongs to this run.

        A mismatched header means the file describes different work -- another
        input, a different zoom -- and silently continuing it would produce a
        document that is part one run and part another.  That is refused.
        """
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
            if not first.strip():
                return {}
            try:
                stored = json.loads(first)
            except json.JSONDecodeError as exc:
                raise CheckpointMismatch(
                    f"{self.path} does not start with a checkpoint header"
                ) from exc
            self._check_header(stored)

            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    self.done[row["i"]] = Completed(
                        index=row["i"],
                        label=row["l"],
                        pano_id=row.get("p"),
                        error=row.get("e"),
                    )
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Only ever the last line, and only after a hard kill:
                    # the process died between write and newline.  That row is
                    # simply relabelled.
                    log.debug("ignoring incomplete checkpoint line")
        return self.done

    def _check_header(self, stored: dict[str, Any]) -> None:
        if stored.get("format") != FORMAT:
            raise CheckpointMismatch(
                f"{self.path} was written by a different version "
                f"(format {stored.get('format')}, expected {FORMAT})"
            )
        for key, expected in self.header.items():
            if stored.get(key) != expected:
                raise CheckpointMismatch(
                    f"{self.path} belongs to a different run: {key} is "
                    f"{stored.get(key)!r} there, {expected!r} here"
                )

    # ---- recording --------------------------------------------------------

    def open(self) -> None:
        """Open for appending, writing the header if the file is new."""
        fresh = not self.path.exists() or self.path.stat().st_size == 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", encoding="utf-8")
        if fresh:
            self._handle.write(json.dumps({"format": FORMAT, **self.header}) + "\n")
            self._handle.flush()

    def record(self, index: int, label: str, pano_id: str | None, error: str | None) -> None:
        """Append one finished row.

        Flushed every time so a kill -9 loses nothing already reported, but
        fsynced only periodically: fsync costs milliseconds and the run only
        produces ~11 rows a second, so this trades a few seconds of work in a
        power cut against not paying for a disk barrier on every panorama.
        """
        if self._handle is None:
            return
        row: dict[str, Any] = {"i": index, "l": label}
        if pano_id:
            row["p"] = pano_id
        if error:
            row["e"] = error[:500]
        self._handle.write(json.dumps(row) + "\n")
        self._handle.flush()
        self._since_sync += 1
        if self._since_sync >= 200:
            os.fsync(self._handle.fileno())
            self._since_sync = 0

    def close(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                os.fsync(self._handle.fileno())
            except OSError:  # a full disk must not mask the real error
                pass
            self._handle.close()
            self._handle = None

    def __enter__(self) -> Checkpoint:
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
