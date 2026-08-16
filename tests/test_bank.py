"""Bank construction helpers: clustering, digit slots, persistence."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pytest

from cr_labeler.bank import Bank, cluster_composites, derive_digit_slots, varying_slots
from cr_labeler.geometry import PATCH_H, PATCH_W


def _groups(count: int, per_group: int, noise: float, seed: int = 0):
    """`count` distinct prototypes with `per_group` noisy copies of each."""
    rng = np.random.default_rng(seed)
    prototypes = [rng.normal(0, 1, (PATCH_H, PATCH_W)).astype(np.float32) for _ in range(count)]
    composites, truth = [], []
    for label, prototype in enumerate(prototypes):
        for _ in range(per_group):
            composites.append(prototype + rng.normal(0, noise, (PATCH_H, PATCH_W)).astype(np.float32))
            truth.append(label)
    return composites, truth


def test_clusters_recover_the_true_groups():
    composites, truth = _groups(6, 40, noise=0.25)
    clusters = cluster_composites(composites, slice(0, PATCH_W), threshold=0.80)
    assert len(clusters) == 6
    for cluster in clusters:
        assert len({truth[i] for i in cluster}) == 1, "cluster mixes two groups"


def test_clustering_is_order_independent():
    """Leader clustering alone depends on input order; the refinement passes
    are what remove that, so the property is worth pinning down."""
    composites, truth = _groups(4, 25, noise=0.25, seed=3)
    forward = cluster_composites(composites, slice(0, PATCH_W), 0.80)
    order = list(reversed(range(len(composites))))
    backward = cluster_composites([composites[i] for i in order], slice(0, PATCH_W), 0.80)

    def signature(clusters, index):
        return sorted(sorted({truth[index[i]] for i in c}) for c in clusters)

    assert signature(forward, list(range(len(composites)))) == signature(backward, order)


def test_clustering_scales():
    """A harvest clusters thousands of composites; the original agglomerative
    implementation re-scanned every cluster pair per merge and could not."""
    composites, _ = _groups(6, 400, noise=0.25, seed=1)
    started = time.time()
    clusters = cluster_composites(composites, slice(0, PATCH_W), 0.80)
    elapsed = time.time() - started
    assert len(clusters) == 6
    assert elapsed < 10.0, f"clustering 2400 composites took {elapsed:.1f}s"


def test_unrelated_composites_do_not_merge():
    composites, _ = _groups(12, 1, noise=0.0, seed=7)
    clusters = cluster_composites(composites, slice(0, PATCH_W), 0.80)
    assert len(clusters) == 12


def test_empty_input():
    assert cluster_composites([], slice(0, PATCH_W)) == []


# --- digit slots ----------------------------------------------------------


def _year_templates(pitch: int = 7, right: int = 70):
    """Templates differing only in their last two digit slots."""
    rng = np.random.default_rng(5)
    base = rng.normal(0, 1, (PATCH_H, PATCH_W)).astype(np.float32)
    templates = []
    for _ in range(4):
        template = base.copy()
        template[:, right - 2 * pitch : right] = rng.normal(
            0, 1, (PATCH_H, 2 * pitch)
        ).astype(np.float32)
        templates.append(template)
    return templates


def test_digit_slots_are_four_contiguous_boxes_of_one_pitch():
    slots = derive_digit_slots(_year_templates(), slice(10, 80), pitch=7)
    assert len(slots) == 4
    assert {b - a for a, b in slots} == {7}
    for (_, end), (start, _) in itertools.pairwise(slots):
        assert end == start


def test_digit_slots_need_two_templates():
    with pytest.raises(ValueError, match="at least two"):
        derive_digit_slots(_year_templates()[:1], slice(10, 80), pitch=7)


def test_varying_slots_keeps_only_what_differs():
    templates = _year_templates()
    slots = derive_digit_slots(templates, slice(10, 80), pitch=7)
    kept = varying_slots(templates, slots)
    assert 0 < len(kept) <= 4
    assert kept[-1] == slots[-1], "the units digit always varies"


# --- persistence ----------------------------------------------------------


def test_bank_round_trips(tmp_path):
    rng = np.random.default_rng(2)
    bank = Bank(
        anchors={"modern": rng.normal(0, 1, (20, 44)).astype(np.float32)},
        templates={("modern", y): rng.normal(0, 1, (PATCH_H, PATCH_W)).astype(np.float32)
                   for y in (2024, 2025)},
        slots={"modern": [(43, 50), (50, 57), (57, 64), (64, 71)]},
    )
    path = tmp_path / "bank.npz"
    bank.save(path)
    loaded = Bank.load(path)

    assert loaded.styles == ["modern"]
    assert loaded.years("modern") == [2024, 2025]
    assert loaded.slots["modern"] == [(43, 50), (50, 57), (57, 64), (64, 71)]
    assert np.allclose(loaded.anchors["modern"], bank.anchors["modern"])
    assert np.allclose(loaded.templates[("modern", 2024)], bank.templates[("modern", 2024)])


def test_loading_a_missing_bank_explains_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-bank"):
        Bank.load(tmp_path / "nope.npz")
