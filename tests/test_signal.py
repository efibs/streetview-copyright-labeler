"""Detection, averaging and consensus, exercised on synthetic panoramas."""

from __future__ import annotations

import numpy as np

from cr_labeler.composite import Template, build_composites
from cr_labeler.geometry import ANCHOR_CX, ANCHOR_CY
from cr_labeler.signal import (
    Correlator,
    consensus,
    declutter,
    find_peaks,
    highpass,
    normalise,
    shift_subpixel,
)
from tests.conftest import render_watermark


def _anchor(font_path, text="© 2024 Google"):
    """A template covering the invariant tail of the string."""
    stamp = np.asarray(render_watermark(font_path, text), np.float32) - 128.0
    return stamp[:, -46:]


def test_highpass_removes_smooth_content(watermarked):
    image, _ = watermarked
    field = highpass(image)
    # A gradient-dominated image should leave almost nothing behind at low
    # frequency once the blur is subtracted.
    assert abs(float(field.mean())) < 1.0
    assert field.shape == (image.height, image.width)


def test_normalise_gives_unit_vectors():
    patch = np.arange(24, dtype=np.float32).reshape(4, 6)
    unit = normalise(patch)
    assert abs(float(np.linalg.norm(unit)) - 1.0) < 1e-6
    assert abs(float(unit.mean())) < 1e-6


def test_normalise_survives_a_flat_patch():
    assert not np.any(normalise(np.full((4, 4), 3.0, dtype=np.float32)))


def test_detection_finds_the_stamped_positions(font_path, watermarked):
    image, centres = watermarked
    field = highpass(image)
    peaks = find_peaks(Correlator(field).match(_anchor(font_path)), 0.45)

    assert len(peaks) >= len(centres) // 2, "should recover most stamps"
    # Every strong peak should sit near a stamp, allowing for the anchor
    # covering the tail of the string rather than its centre.
    for peak in peaks[:10]:
        assert min(abs(peak.y - cy) for _, cy in centres) <= 3


def test_no_detections_on_a_blank_panorama(font_path, blank):
    peaks = find_peaks(Correlator(highpass(blank)).match(_anchor(font_path)), 0.45)
    assert len(peaks) == 0


def test_averaging_recovers_the_watermark(font_path, watermarked):
    """The whole premise: N averaged instances reproduce the stamped glyphs."""
    image, _ = watermarked
    field = highpass(image)
    anchors = [Template(array=_anchor(font_path), style="modern")]
    result = build_composites(field, anchors)[0]
    assert result.instances >= 8

    # The composite should contain the same glyphs that were stamped in. Search
    # over offsets because the anchor covers the tail of the string, so the
    # composite's frame is not the renderer's.
    truth = normalise(np.asarray(render_watermark(font_path, "© 2024 Google"), np.float32) - 128.0)
    composite = declutter(result.composite)
    th, tw = truth.shape

    best = 0.0
    for dy in range(0, composite.shape[0] - th + 1):
        for dx in range(0, composite.shape[1] - tw + 1):
            window = normalise(composite[dy : dy + th, dx : dx + tw])
            best = max(best, abs(float((window * truth).sum())))

    assert best > 0.5, f"composite does not resemble the stamped watermark ({best:.2f})"


def test_averaging_beats_a_single_instance(font_path, watermarked):
    """Averaging must actually reduce noise, not merely preserve the signal."""
    image, _ = watermarked
    field = highpass(image)
    anchor = _anchor(font_path)
    peaks = find_peaks(Correlator(field).match(anchor), 0.45)
    assert len(peaks) >= 8

    template = Template(array=anchor, style="modern")
    composite = build_composites(field, [template])[0].composite

    from cr_labeler.composite import _patches

    patches = _patches(field, peaks, template)
    reference = normalise(declutter(composite))

    single = np.mean(
        [abs(float((normalise(declutter(p)) * reference).sum())) for p in patches]
    )
    assert single < 0.95, "a lone instance should be noisier than the average"


def test_consensus_discards_disagreeing_patches():
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 1, (32, 132)).astype(np.float32)
    real = [signal + rng.normal(0, 0.3, (32, 132)).astype(np.float32) for _ in range(6)]
    noise = [rng.normal(0, 1, (32, 132)).astype(np.float32) for _ in range(9)]

    composite, inliers = consensus(real + noise)
    assert composite is not None
    assert len(inliers) == 6
    assert all(index < 6 for index in inliers)


def test_consensus_finds_no_group_in_pure_noise():
    """Unrelated patches must not form a group; a patch always matches itself,
    so one survivor is the floor, and the quality gate rejects it downstream."""
    rng = np.random.default_rng(5)
    _, inliers = consensus(
        [rng.normal(0, 1, (32, 132)).astype(np.float32) for _ in range(10)]
    )
    assert len(inliers) <= 2


def test_consensus_handles_no_patches():
    composite, inliers = consensus([])
    assert composite is None and inliers == []


def test_subpixel_shift_moves_by_the_requested_amount():
    """On a smooth ramp, bilinear interpolation is exact, so the shift is checkable.
    (White noise is not a fair test: no interpolator can round-trip it.)"""
    ramp = np.tile(np.arange(132, dtype=np.float32), (32, 1))
    shifted = shift_subpixel(ramp, 0.0, 0.5)
    interior = (slice(None), slice(4, -4))
    assert np.allclose(shifted[interior], ramp[interior] - 0.5, atol=1e-4)


def test_subpixel_shift_is_a_noop_for_zero_offset():
    patch = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert np.array_equal(shift_subpixel(patch, 0.0, 0.0), patch)


def test_build_composites_returns_one_result_per_anchor(font_path, watermarked):
    image, _ = watermarked
    anchors = [
        Template(array=_anchor(font_path), style="modern", cx=ANCHOR_CX, cy=ANCHOR_CY),
        Template(array=_anchor(font_path, "© 2019 Google"), style="legacy"),
    ]
    results = build_composites(highpass(image), anchors)
    assert [r.style for r in results] == ["modern", "legacy"]


def test_blank_panorama_yields_no_instances(font_path, blank):
    anchors = [Template(array=_anchor(font_path), style="modern")]
    assert build_composites(highpass(blank), anchors)[0].instances == 0


def test_gpu_highpass_matches_pillow_closely():
    """The GPU filter must reproduce Pillow's, not merely approximate it.

    Pillow blurs with three extended box filters rather than a Gaussian; a real
    Gaussian of the same sigma disagrees by up to 15 grey levels, which is what
    this guards against regressing to.
    """
    import numpy as np
    import pytest
    from PIL import Image

    from cr_labeler.accel import device
    from cr_labeler.accel import highpass as gpu_highpass
    from cr_labeler.geometry import HIGHPASS_SIGMA
    from cr_labeler.signal import highpass

    if device() is None:
        pytest.skip("no GPU available")

    rng = np.random.default_rng(0)
    scene = rng.integers(0, 256, size=(256, 512), dtype=np.uint8)
    image = Image.fromarray(scene, mode="L").convert("RGB")

    gpu = gpu_highpass(image, HIGHPASS_SIGMA)
    assert gpu is not None
    reference = highpass(image)

    assert np.abs(gpu - reference).max() <= 1.0
    assert (gpu == reference).mean() > 0.90


def test_gpu_highpass_can_be_switched_off(monkeypatch):
    from PIL import Image

    from cr_labeler.accel import highpass as gpu_highpass
    from cr_labeler.geometry import HIGHPASS_SIGMA

    monkeypatch.setenv("CR_LABELER_GPU_HIGHPASS", "0")
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    assert gpu_highpass(image, HIGHPASS_SIGMA) is None
