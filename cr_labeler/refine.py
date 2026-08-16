"""Split bank styles by rendering layout.

A "style" only has to satisfy one property: every template in it must share a
set of digit slots.  The styles that come out of a seed build are eras -- the
tight ``(c)2019`` form against the spaced ``(c) 2024`` one -- and that is too
coarse.  Measured on the shipped bank, what was one ``modern`` style holds three
different renderings, with the year sitting in different columns in each.  The
slots derived for the group land off the digits for at least one member, which
compresses the gap between adjacent years until the classifier declines to
choose: every one of the abstentions left on the hand-labelled set was 2026.

Splitting on the glyph bounding box separates them without needing to know when
Google changed the overlay.  The anchor is shared with the parent style -- it
matches the ``Google`` wordmark, which is common to all of them -- so only the
slots have to be re-derived.
"""

from __future__ import annotations

import logging

import numpy as np

from .bank import Bank, derive_digit_slots, layout_groups, text_box

log = logging.getLogger(__name__)

# A group needs two years before disagreement can show where the digits are.
MIN_FOR_SLOTS = 2


def _captured(templates: dict[int, np.ndarray], slots: list[tuple[int, int]]) -> float:
    """Fraction of the cross-year disagreement that falls inside ``slots``."""
    spread = np.stack(list(templates.values())).std(axis=0).sum(axis=0)
    total = float(spread.sum())
    if total <= 0:
        return 0.0
    return float(sum(spread[a:b].sum() for a, b in slots)) / total


def _year_region(slots: list[tuple[int, int]], pad: int = 12) -> slice:
    return slice(max(0, slots[0][0] - pad), min(132, slots[-1][1] + pad))


def refine(bank: Bank) -> tuple[Bank, list[str]]:
    """Return a bank whose styles each contain exactly one rendering."""
    refined = Bank()
    notes: list[str] = []

    for style in bank.styles:
        templates = {year: bank.templates[(style, year)] for year in bank.years(style)}
        if not templates:
            continue
        anchor = bank.anchors[style]
        groups = layout_groups(templates)

        if len(groups) <= 1:
            refined.anchors[style] = anchor
            refined.slots[style] = bank.slots[style]
            for year, template in templates.items():
                refined.templates[(style, year)] = template
            notes.append(f"{style}: single layout, unchanged ({len(templates)} years)")
            continue

        first = bank.slots[style][0]
        pitch = max(1, first[1] - first[0])
        for index, (box, years) in enumerate(sorted(groups.items(), key=lambda kv: min(kv[1]))):
            name = f"{style}_{index}" if len(groups) > 1 else style
            members = {y: templates[y] for y in years}
            refined.anchors[name] = anchor
            for year, template in members.items():
                refined.templates[(name, year)] = template

            if len(members) >= MIN_FOR_SLOTS:
                region = _year_region(bank.slots[style])
                candidate = derive_digit_slots(list(members.values()), region, pitch)
                # Two years differing in a single digit give the derivation very
                # little to work with, and it can settle on the wrong columns.
                # Keep the new slots only if they actually sit on more of the
                # disagreement than the parent's did.
                if _captured(members, candidate) >= _captured(members, bank.slots[style]):
                    refined.slots[name] = candidate
                    how = "derived"
                else:
                    refined.slots[name] = bank.slots[style]
                    how = "kept parent's (derivation fitted worse)"
            else:
                # One year alone cannot show where its digits vary; shift the
                # parent's slots by how far this layout sits from the parent's.
                parent_left = min(text_box(t)[0] for t in templates.values())
                offset = box[0] - parent_left
                refined.slots[name] = [
                    (a + offset, b + offset) for a, b in bank.slots[style]
                ]
                how = f"shifted {offset:+d}px from parent"
            notes.append(
                f"{name}: years {years}, text cols {box[0]}-{box[1]}, "
                f"slots {how} -> {refined.slots[name]}"
            )

    return refined, notes


def slot_fit(bank: Bank) -> dict[str, float]:
    """How well each style's slots sit on its own digits.

    The digits are where the templates of a style disagree, so a good slot set
    captures most of that disagreement.  A low score means the slots are landing
    somewhere else.
    """
    out: dict[str, float] = {}
    for style in bank.styles:
        years = bank.years(style)
        if len(years) < 2:
            continue
        stack = np.stack([bank.templates[(style, y)] for y in years])
        spread = stack.std(axis=0).sum(axis=0)
        total = float(spread.sum())
        inside = float(sum(spread[a:b].sum() for a, b in bank.slots[style]))
        out[style] = inside / total if total else 0.0
    return out
