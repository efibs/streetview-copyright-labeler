"""Splitting a style whose templates do not share one rendering."""

from __future__ import annotations

import numpy as np
import pytest

from cr_labeler.bank import Bank, layout_groups, text_box
from cr_labeler.geometry import PATCH_H, PATCH_W
from cr_labeler.refine import refine, slot_fit


def _template(left: int, width: int = 90, seed: int = 0) -> np.ndarray:
    """A synthetic template whose glyph block starts at ``left``.

    The block has a fixed extent so its bounding box depends only on ``left``,
    as a real rendering's does; only the digit region varies between years.
    """
    rng = np.random.default_rng(seed)
    patch = np.zeros((PATCH_H, PATCH_W), np.float32)
    # Block level sits well clear of the bounding-box threshold so the box
    # depends only on `left`; the digit region varies between years.
    patch[9:22, left : left + width] = 10.0
    digits = slice(left + 40, left + 68)
    patch[9:22, digits] += rng.normal(0, 2, (13, 28)).astype(np.float32)
    return patch


def _bank(layouts: dict[int, int]) -> Bank:
    """`layouts` maps year -> the column its glyphs start at."""
    bank = Bank(anchors={"modern": np.zeros((20, 44), np.float32)})
    for i, (year, left) in enumerate(sorted(layouts.items())):
        bank.templates[("modern", year)] = _template(left, seed=i)
    bank.slots["modern"] = [(43, 50), (50, 57), (57, 64), (64, 71)]
    return bank


def test_text_box_finds_the_glyph_block():
    left, right, top, bottom = text_box(_template(26))
    assert left == 26
    assert right == 26 + 90 - 1
    assert (top, bottom) == (9, 21)


def test_layout_groups_separate_different_renderings():
    templates = {2011: _template(26, seed=1), 2013: _template(26, seed=2),
                 2024: _template(20, seed=3), 2025: _template(20, seed=4)}
    groups = layout_groups(templates)
    assert len(groups) == 2
    assert sorted(sorted(v) for v in groups.values()) == [[2011, 2013], [2024, 2025]]


def test_refine_splits_a_mixed_style():
    bank = _bank({2011: 26, 2013: 26, 2024: 20, 2025: 20, 2026: 20})
    refined, notes = refine(bank)
    assert len(refined.styles) == 2, notes
    years = {s: refined.years(s) for s in refined.styles}
    assert sorted(years.values()) == [[2011, 2013], [2024, 2025, 2026]]
    # every split style keeps a usable anchor and its own slots
    for style in refined.styles:
        assert refined.anchors[style].size
        assert len(refined.slots[style]) == 4


def test_refine_leaves_a_consistent_style_alone():
    bank = _bank({2011: 26, 2013: 26, 2014: 26})
    refined, notes = refine(bank)
    assert refined.styles == ["modern"], notes
    assert refined.slots["modern"] == bank.slots["modern"]
    assert refined.years("modern") == [2011, 2013, 2014]


def test_refine_is_idempotent():
    bank = _bank({2011: 26, 2013: 26, 2024: 20, 2025: 20})
    once, _ = refine(bank)
    twice, notes = refine(once)
    assert twice.styles == once.styles, notes
    for style in once.styles:
        assert twice.slots[style] == once.slots[style]
        assert twice.years(style) == once.years(style)


def test_refine_survives_a_single_year_style():
    """A lone year cannot show where its digits vary; it must still get slots."""
    bank = _bank({2009: 24, 2016: 27, 2017: 27})
    refined, notes = refine(bank)
    assert len(refined.styles) == 2, notes
    for style in refined.styles:
        assert len(refined.slots[style]) == 4


def test_slot_fit_is_a_fraction():
    bank = _bank({2011: 26, 2013: 26, 2014: 26})
    for value in slot_fit(bank).values():
        assert 0.0 <= value <= 1.0


def test_text_box_rejects_an_empty_template():
    with pytest.raises(ValueError, match="no glyphs"):
        text_box(np.zeros((PATCH_H, PATCH_W), np.float32))
