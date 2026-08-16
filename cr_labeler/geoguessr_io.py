"""Reading and writing the GeoGuessr custom-map JSON.

The output is the input with one tag added.  Every other field -- coordinates,
heading, pitch, existing tags, ``panoDate``, keys we have never seen -- is
carried through untouched, so a tagged map stays a valid map.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TAG_PREFIX = "CR_"


@dataclass
class Location:
    """One entry of ``customCoordinates``."""

    index: int
    raw: dict[str, Any]

    @property
    def pano_id(self) -> str | None:
        value = self.raw.get("panoId")
        return value if isinstance(value, str) and value else None

    @property
    def lat(self) -> float | None:
        value = self.raw.get("lat")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def lng(self) -> float | None:
        value = self.raw.get("lng")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def tags(self) -> list[str]:
        extra = self.raw.get("extra")
        if not isinstance(extra, dict):
            return []
        tags = extra.get("tags")
        return [t for t in tags if isinstance(t, str)] if isinstance(tags, list) else []

    def ground_truth(self) -> str | None:
        """The ``GT_*`` tag, if this entry carries one."""
        for tag in self.tags:
            if tag.startswith("GT_"):
                return tag[len("GT_") :]
        return None

    def describe(self) -> str:
        if self.pano_id:
            return f"row {self.index} (panoId {self.pano_id})"
        if self.lat is not None and self.lng is not None:
            return f"row {self.index} ({self.lat:.6f}, {self.lng:.6f})"
        return f"row {self.index}"

    def apply_tag(self, label: str) -> None:
        """Add ``CR_<label>``, replacing any previous ``CR_`` tag."""
        extra = self.raw.get("extra")
        if not isinstance(extra, dict):
            extra = {}
            self.raw["extra"] = extra
        tags = extra.get("tags")
        if not isinstance(tags, list):
            tags = []
        kept = [t for t in tags if not (isinstance(t, str) and t.startswith(TAG_PREFIX))]
        kept.append(f"{TAG_PREFIX}{label}")
        extra["tags"] = kept


@dataclass
class GeoGuessrMap:
    """A parsed map document plus its locations."""

    document: dict[str, Any]
    locations: list[Location]

    @classmethod
    def load(cls, path: Path) -> GeoGuessrMap:
        try:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc

        if isinstance(document, list):
            # Some exports are a bare array of locations.
            entries = document
            document = {"customCoordinates": entries}
        elif isinstance(document, dict):
            entries = document.get("customCoordinates")
        else:
            raise ValueError(f"{path}: expected a JSON object or array")

        if not isinstance(entries, list):
            raise ValueError(
                f"{path}: expected a 'customCoordinates' array, found "
                f"{type(entries).__name__}"
            )

        locations = [
            Location(index=i, raw=entry)
            for i, entry in enumerate(entries)
            if isinstance(entry, dict)
        ]
        return cls(document=document, locations=locations)

    def save(self, path: Path) -> None:
        """Write the document back, preserving the compact one-per-line layout.

        Streamed a row at a time and written to a temporary file that is renamed
        into place at the end.  Both matter at size: serialising 900k rows into
        one string wanted about half a gigabyte all at once, at the very end of
        a run that had already taken most of a day, and writing in place meant a
        failure part-way through left a truncated document where the previous
        one had been.  A rename is atomic, so the output file is either the old
        one or the complete new one, never half of either.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        entries = self.document.get("customCoordinates", [])
        head = {k: v for k, v in self.document.items() if k != "customCoordinates"}
        prefix = json.dumps(head, ensure_ascii=False, separators=(",", ":"))[1:-1]

        temporary = path.with_name(path.name + ".part")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("{" + (prefix + "," if prefix else "") + '"customCoordinates":[\n')
            for position, entry in enumerate(entries):
                if position:
                    handle.write(",\n")
                handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n]}")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
