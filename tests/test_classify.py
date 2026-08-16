"""Classification decisions, driven by a small hand-made bank."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cr_labeler.bank import Bank, derive_digit_slots, varying_slots
from cr_labeler.classify import NO_WATERMARK, UNKNOWN, classify
from cr_labeler.geometry import PATCH_H, PATCH_W
from tests.conftest import render_watermark


def _template(font_path, text: str) -> np.ndarray:
    """A year template placed in canonical patch geometry."""
    stamp = np.asarray(render_watermark(font_path, text), np.float32) - 128.0
    patch = np.zeros((PATCH_H, PATCH_W), np.float32)
    height = min(PATCH_H, stamp.shape[0])
    width = min(PATCH_W, stamp.shape[1])
    patch[2 : 2 + height, 6 : 6 + width] = stamp[:height, :width]
    return patch


@pytest.fixture
def bank(font_path):
    years = [2021, 2024, 2025, 2026]
    templates = {("modern", y): _template(font_path, f"© {y} Google") for y in years}
    built = Bank(
        anchors={"modern": np.zeros((20, 44), np.float32)},
        templates=templates,
    )
    region = slice(10, 80)
    built.slots["modern"] = derive_digit_slots(list(templates.values()), region, pitch=7)
    return built


def test_reads_each_year_it_was_built_from(bank, font_path):
    for year in bank.years("modern"):
        composite = _template(font_path, f"© {year} Google")
        verdict = classify(composite, instances=20, bank=bank)
        assert verdict.label == str(year), f"misread {year} as {verdict.label}"
        assert verdict.confidence > 0.5


def test_no_instances_means_no_watermark(bank):
    verdict = classify(np.zeros((PATCH_H, PATCH_W), np.float32), instances=0, bank=bank)
    assert verdict.label == NO_WATERMARK
    assert verdict.confidence == 1.0


def test_noise_is_rejected_not_guessed(bank):
    rng = np.random.default_rng(2)
    noise = rng.normal(0, 10, (PATCH_H, PATCH_W)).astype(np.float32)
    verdict = classify(noise, instances=25, bank=bank)
    assert verdict.label in (NO_WATERMARK, UNKNOWN)
    assert verdict.year is None


@pytest.mark.parametrize("absent", ["2017", "2022", "2023", "2027"])
def test_a_year_absent_from_the_bank_abstains(bank, font_path, absent):
    """The dangerous case, and the reason for the per-slot gate.

    Ranking only compares the years the bank holds, so an absent year picks its
    nearest neighbour -- and wins by a wide margin, because three of its four
    digits genuinely do match. Observed on real panoramas reading "(c) 2022
    Google" being reported as 2021 with a margin of 0.8.
    """
    verdict = classify(_template(font_path, f"© {absent} Google"), instances=20, bank=bank)
    assert verdict.label == UNKNOWN, (
        f"{absent} is not in the bank but was reported as {verdict.label}"
    )


def test_digit_slots_land_on_the_year(bank):
    slots = bank.slots["modern"]
    assert len(slots) == 4
    widths = {b - a for a, b in slots}
    assert widths == {7}, "digits share one pitch"
    for (_, end), (start, _) in itertools.pairwise(slots):
        assert end == start, "slots must be contiguous"


def test_varying_slots_drops_the_constant_prefix(bank):
    templates = [bank.templates[("modern", y)] for y in bank.years("modern")]
    kept = varying_slots(templates, bank.slots["modern"])
    # Every year here is 202x, so only the units digit carries information.
    assert len(kept) < 4
    assert kept[-1] == bank.slots["modern"][-1]


def test_verdict_reports_the_runner_up(bank, font_path):
    verdict = classify(_template(font_path, "© 2025 Google"), instances=20, bank=bank)
    assert verdict.runner_up in bank.years("modern")
    assert verdict.runner_up != verdict.year
