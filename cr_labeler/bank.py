"""The reference template bank.

The bank holds everything the classifier needs and nothing it can learn at
runtime:

* ``anchors``     - one year-invariant ``Google`` wordmark per render style
* ``templates``   - one averaged composite per (style, year)
* ``slots``       - per-style column ranges of the four year digits

It is built once by :mod:`cr_labeler.cli` ``build-bank`` and shipped as a single
``.npz``.  Nothing here is trained; the templates are real composites, which is
precisely why they match real panoramas so much better than synthetic text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import PATCH_H, PATCH_W
from .signal import normalise

DEFAULT_BANK = Path(__file__).resolve().parent.parent / "bank" / "templates.npz"

STYLE_MODERN = "modern"
STYLE_LEGACY = "legacy"


@dataclass
class Bank:
    """Reference templates for every known render style and year."""

    anchors: dict[str, np.ndarray] = field(default_factory=dict)
    templates: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    slots: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # ---- introspection ---------------------------------------------------

    @property
    def styles(self) -> list[str]:
        return sorted(self.anchors)

    def years(self, style: str) -> list[int]:
        return sorted(year for (this_style, year) in self.templates if this_style == style)

    def summary(self) -> str:
        parts = []
        for style in self.styles:
            years = self.years(style)
            span = f"{years[0]}-{years[-1]}" if years else "none"
            parts.append(f"{style}: {len(years)} years ({span})")
        return "; ".join(parts) if parts else "empty bank"

    # ---- persistence -----------------------------------------------------

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        meta: dict[str, object] = {
            "styles": self.styles,
            "slots": {k: [list(v) for v in vs] for k, vs in self.slots.items()},
            "entries": [],
        }
        for style, anchor in self.anchors.items():
            arrays[f"anchor::{style}"] = anchor.astype(np.float32)
        for (style, year), template in sorted(self.templates.items()):
            arrays[f"tpl::{style}::{year}"] = template.astype(np.float32)
            meta["entries"].append([style, year])
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path | None = None) -> Bank:
        path = Path(path) if path else DEFAULT_BANK
        if not path.exists():
            raise FileNotFoundError(
                f"template bank not found at {path}.\n"
                "Build one with:  cr-label build-bank --seed CR_GT.json"
            )
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["__meta__"]).decode("utf-8"))
            bank = cls(
                slots={
                    k: [tuple(v) for v in vs] for k, vs in meta.get("slots", {}).items()
                }
            )
            for key in data.files:
                if key.startswith("anchor::"):
                    bank.anchors[key.split("::", 1)[1]] = data[key]
                elif key.startswith("tpl::"):
                    _, style, year = key.split("::")
                    bank.templates[(style, int(year))] = data[key]
        return bank


# --- construction helpers -------------------------------------------------


DIGITS_PER_YEAR = 4


def derive_digit_slots(
    templates: list[np.ndarray], year_region: slice, pitch: int
) -> list[tuple[int, int]]:
    """Locate the four year digits as column ranges.

    Glyph segmentation proved unreliable -- strokes merge and drop out -- so the
    year's right edge is found by disagreement instead: the units digit is the
    rightmost place where templates of different years differ, because every
    year in range ends in a different digit.  The four slots then step left from
    there at the font's fixed pitch.

    The pitch is supplied rather than measured from the disagreement, whose
    width is not trustworthy: where only the units digit varies across the
    bank's years, the single run smears to three digits wide and every slot ends
    up covering the whole year.

    Slots matter rather than one pooled weighting because each digit must count
    equally.  Pooled, a digit that varies widely across the bank drowns out one
    that varies subtly, and 2016/2018/2019 stop being separable even though the
    templates are perfectly legible.
    """
    if len(templates) < 2:
        raise ValueError("need at least two year templates to locate the digits")

    spread = np.stack([normalise(t) for t in templates]).std(axis=0).sum(axis=0)
    inside = np.zeros_like(spread)
    inside[year_region] = spread[year_region]
    if float(inside.max()) <= 0:
        raise ValueError("year templates do not differ anywhere in the year block")

    active = np.nonzero(inside > 0.35 * float(inside.max()))[0]
    if not active.size:
        raise ValueError("could not isolate a varying digit")

    right = int(active[-1]) + 1
    return [
        (max(0, right - pitch * (step + 1)), min(PATCH_W, right - pitch * step))
        for step in reversed(range(DIGITS_PER_YEAR))
    ]


def text_box(template: np.ndarray, keep: float = 0.35) -> tuple[int, int, int, int]:
    """Bounding box of the glyphs: ``(left, right, top, bottom)``."""
    energy = np.abs(template)
    threshold = keep * float(energy.max())
    cols = np.nonzero(energy.max(axis=0) > threshold)[0]
    rows = np.nonzero(energy.max(axis=1) > threshold)[0]
    if not cols.size or not rows.size:
        raise ValueError("template carries no glyphs")
    return int(cols[0]), int(cols[-1]), int(rows[0]), int(rows[-1])


def layout_groups(templates: dict[int, np.ndarray]) -> dict[tuple, list[int]]:
    """Split a style's years by the geometry of the rendering.

    Google changed the overlay more often than the two obvious styles suggest.
    Within what looks like one "modern" style there are three layouts -- measured
    text spans of columns 26-120, 28-121 and 20-120 -- and the digit slots of one
    do not land on the digits of another.  Sharing slots across them is what
    collapses the margin between adjacent years and forces an abstention.

    Grouping on the bounding box separates them without having to know when
    Google made each change.
    """
    groups: dict[tuple, list[int]] = {}
    for year, template in sorted(templates.items()):
        groups.setdefault(text_box(template), []).append(year)
    return groups


def varying_slots(
    templates: list[np.ndarray], slots: list[tuple[int, int]], floor: float = 0.25
) -> list[tuple[int, int]]:
    """Keep the digit slots that actually differ between the bank's years.

    Every year in range starts "20", so scoring those two slots would add the
    same near-perfect match to every candidate and shrink the margin that
    decides the answer.
    """
    if len(templates) < 2:
        return slots
    stack = np.stack([normalise(t) for t in templates])
    spreads = [float(stack.std(axis=0)[:, a:b].sum()) for a, b in slots]
    peak = max(spreads) or 1.0
    kept = [slot for slot, spread in zip(slots, spreads, strict=True) if spread >= floor * peak]
    return kept or slots


CLUSTER_REFINEMENTS = 4


def cluster_composites(
    composites: list[np.ndarray],
    columns: slice,
    threshold: float = 0.80,
) -> list[list[int]]:
    """Group composites by similarity over ``columns``; largest cluster first.

    Leader clustering followed by a few reassignment passes, rather than
    agglomerative merging.  Agglomerative is the textbook choice here and was
    the first implementation, but it re-scans every pair of clusters on every
    merge -- fine for the tens of composites a seed bank sees, hopeless for the
    thousands a harvest produces.

    The shortcut is sound because of what is being clustered: composites of the
    same year are near-identical (measured ~0.87 similarity) while different
    years sit far below the cut (~0.61), so the groups are compact and
    well-separated.  The refinement passes remove the order-dependence that
    plain leader clustering would otherwise leave behind.
    """
    if not composites:
        return []

    matrix = np.stack(
        [normalise(c[:, columns]).ravel() for c in composites]
    ).astype(np.float32)

    # --- leader pass: first composite of each group defines it ---------------
    centroids: list[np.ndarray] = []
    assignment = np.empty(len(matrix), dtype=np.int32)
    for index, vector in enumerate(matrix):
        if centroids:
            scores = np.stack(centroids) @ vector
            best = int(np.argmax(scores))
            if scores[best] >= threshold:
                assignment[index] = best
                continue
        centroids.append(vector.copy())
        assignment[index] = len(centroids) - 1

    # --- refine: reassign to the nearest centroid, then recompute ------------
    for _ in range(CLUSTER_REFINEMENTS):
        stack = np.stack(centroids)
        scores = matrix @ stack.T
        best = scores.argmax(axis=1)
        # A composite matching nothing well enough keeps its own group rather
        # than being forced into the nearest one.
        keep = scores[np.arange(len(matrix)), best] >= threshold
        updated = np.where(keep, best, assignment)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        centroids = []
        for label in range(int(assignment.max()) + 1):
            members = matrix[assignment == label]
            if len(members):
                mean = members.mean(axis=0)
                norm = float(np.linalg.norm(mean))
                centroids.append(mean / norm if norm > 1e-9 else mean)
        if not centroids:
            break
        # Labels may have collapsed; renumber so indices stay contiguous.
        used = sorted({int(a) for a in assignment})
        remap = {old: new for new, old in enumerate(used)}
        assignment = np.array([remap[int(a)] for a in assignment], dtype=np.int32)

    groups: dict[int, list[int]] = {}
    for index, label in enumerate(assignment):
        groups.setdefault(int(label), []).append(index)
    return sorted(groups.values(), key=len, reverse=True)


def average_composites(composites: list[np.ndarray]) -> np.ndarray:
    """Mean of normalised composites -- the canonical form stored in the bank."""
    if not composites:
        return np.zeros((PATCH_H, PATCH_W), dtype=np.float32)
    stack = np.stack([normalise(c) for c in composites])
    return stack.mean(axis=0).astype(np.float32)
