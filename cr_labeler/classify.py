"""Read the year out of an averaged watermark composite.

Two stages, deliberately separated:

1. *Alignment and quality.*  The composite is aligned against each style's mean
   reference.  The resulting score answers "is this a watermark at all?", which
   is what rejects the noise composites that permissive detection produces on
   panoramas that carry no watermark.
2. *Digit reading.*  With alignment fixed, each year digit is scored in its own
   slot.  Only the slots that actually differ across the bank's years decide the
   ranking -- every year in range begins with "20", so scoring those constant
   glyphs would only dilute the margin -- but *all four* slots are checked
   before the answer is accepted, which is what catches a year the bank does not
   contain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bank import Bank, varying_slots
from .signal import best_alignment, declutter, normalise

# Below this alignment score the composite does not look like a watermark, so
# whatever the digit stage says about it is meaningless.  Measured on ground
# truth, correct reads score 0.53 and up while panoramas carrying no watermark
# land at 0.25, leaving a wide gap to sit in.
QUALITY_GATE = 0.45
# Further down still, the composite is not a degraded watermark but no watermark
# at all -- unofficial coverage, photospheres, some first-generation imagery.
# That is a real answer ("None"), not an abstention.
ABSENT_GATE = 0.35
DIGIT_GATE = 0.50
# Only a guard against a literal tie.  Margin was originally the main defence
# against a doubtful read, but that does not survive a dense bank: with every
# year from 2011 to 2026 present, 2021 and 2022 differ by a single digit and the
# gap between them is small even when the read is certain.  Measured against
# ground truth with the full bank, correct reads run down to a margin of 0.010
# while the year that ranks first is right 17 times out of 17 -- so a meaningful
# gate here rejects good answers and catches nothing.  Verification is
# SLOT_GATE's job; this only refuses to break an exact tie.
MARGIN_GATE = 0.005
# Every digit of the winning year must match, not just the average of them.
# Measured on real panoramas: correct reads never drop below 0.854 at their
# weakest digit, while panoramas whose year is missing from the bank peak at
# 0.678. This sits in that gap. See the note at its use site.
SLOT_GATE = 0.75

NO_WATERMARK = "None"
UNKNOWN = "unknown"


@dataclass
class Verdict:
    """The classifier's answer, with the numbers behind it."""

    label: str  # "2024", "None" or "unknown"
    year: int | None
    style: str | None
    quality: float
    digit_score: float
    margin: float
    instances: int
    runner_up: int | None = None

    @property
    def confidence(self) -> float:
        """A single 0-1 number for reporting and sorting.

        Blends how watermark-like the composite is with how decisively the
        winning year beat the alternatives.
        """
        if self.label == NO_WATERMARK:
            return 1.0 if self.instances == 0 else 0.5
        if self.year is None:
            return 0.0
        decisiveness = min(self.margin / 0.20, 1.0)
        return float(np.clip(0.5 * self.quality + 0.3 * self.digit_score + 0.2 * decisiveness, 0.0, 1.0))


def _style_reference(bank: Bank, style: str) -> np.ndarray:
    """Mean of every year template in a style: the year-agnostic prototype."""
    templates = [t for (s, _), t in bank.templates.items() if s == style]
    return np.mean(np.stack([normalise(t) for t in templates]), axis=0)


def classify(composite: np.ndarray, instances: int, bank: Bank) -> Verdict:
    """Classify one composite against the bank."""
    if instances == 0:
        return Verdict(
            label=NO_WATERMARK,
            year=None,
            style=None,
            quality=0.0,
            digit_score=0.0,
            margin=0.0,
            instances=0,
        )

    # Bank templates are stored decluttered, so the composite has to be brought
    # into the same domain before anything is compared.
    cleaned = declutter(composite)

    # --- stage 1: which style, and how watermark-like is this? -------------
    best_style: tuple[float, str, int, int] | None = None
    for style in bank.styles:
        if not bank.years(style):
            continue
        score, dy, dx = best_alignment(_style_reference(bank, style), cleaned)
        if best_style is None or score > best_style[0]:
            best_style = (score, style, dy, dx)

    if best_style is None:
        raise ValueError("template bank contains no year templates")

    quality, style, dy, dx = best_style
    aligned = np.roll(cleaned, (dy, dx), axis=(0, 1))

    if quality < QUALITY_GATE:
        # Whatever consensus latched onto here, it is not the watermark: either
        # there is none, or it is too degraded to read.
        return Verdict(
            label=NO_WATERMARK if quality < ABSENT_GATE else UNKNOWN,
            year=None,
            style=style,
            quality=quality,
            digit_score=0.0,
            margin=0.0,
            instances=instances,
        )

    # --- stage 2: read the digits that actually vary -----------------------
    # Each digit is scored in its own slot and the slots are averaged, so every
    # digit position counts the same.  Scoring the year block as one window
    # instead lets whichever digit varies most across the bank dominate, and
    # adjacent years stop being separable at all.
    templates = {y: t for (s, y), t in bank.templates.items() if s == style}
    all_slots = bank.slots[style]
    slots = varying_slots(list(templates.values()), all_slots)
    probes = [normalise(aligned[:, a:b]) for a, b in slots]

    scores = {
        year: float(
            np.mean(
                [
                    (probe * normalise(template[:, a:b])).sum()
                    for probe, (a, b) in zip(probes, slots, strict=True)
                ]
            )
        )
        for year, template in templates.items()
    }

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top_year, top_score = ranked[0]
    runner_up, runner_score = ranked[1] if len(ranked) > 1 else (None, -1.0)
    margin = top_score - runner_score if runner_up is not None else 1.0

    # A year the bank has never seen is the dangerous case: ranking only compares
    # the candidates present, so 2022 against a bank holding 2021/2024/2025/2026
    # picks 2021 and wins by a wide margin -- three of its four digits do match.
    # Guard by checking the winner digit by digit: a year that is really there
    # matches at *every* position, so the weakest position is what exposes an
    # impostor. Verified on real panoramas reading "(c) 2022 Google".
    weakest = min(
        float((normalise(aligned[:, a:b]) * normalise(templates[top_year][:, a:b])).sum())
        for a, b in all_slots
    )

    if top_score < DIGIT_GATE or margin < MARGIN_GATE or weakest < SLOT_GATE:
        return Verdict(
            label=UNKNOWN,
            year=None,
            style=style,
            quality=quality,
            digit_score=top_score,
            margin=margin,
            instances=instances,
            runner_up=runner_up,
        )

    return Verdict(
        label=str(top_year),
        year=top_year,
        style=style,
        quality=quality,
        digit_score=top_score,
        margin=margin,
        instances=instances,
        runner_up=runner_up,
    )
