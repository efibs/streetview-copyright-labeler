"""Optional GPU backend for the matched filter.

Correlation is about three quarters of the work, even with a cold tile cache,
and it is almost entirely FFT -- which is what a GPU is good at.  Measured on an
RTX A5000, a two-anchor match takes 16 ms against 255 ms in numpy.

Entirely optional.  PyTorch is not a dependency; if it is missing, or there is
no CUDA device, or ``CR_LABELER_DEVICE=cpu`` is set, everything runs on the
numpy path exactly as before.  Nothing in the program requires particular
hardware, and the answers do not depend on which path ran -- verified across all
543 hand-labelled panoramas.

Set ``CR_LABELER_DEVICE`` to ``cpu``, ``cuda`` or ``auto`` (the default).
"""

from __future__ import annotations

import logging
import math
import os
import threading

import numpy as np

log = logging.getLogger(__name__)

_PROBE_LOCK = threading.Lock()
_DEVICE: object | None = None
_PROBED = False


def device():
    """The CUDA device to use, or ``None`` to stay on numpy.

    Probed once.  Any failure -- torch absent, no driver, no device, a broken
    install -- is answered with ``None`` rather than an exception, because a
    missing accelerator is not an error.
    """
    global _DEVICE, _PROBED
    if _PROBED:
        return _DEVICE
    with _PROBE_LOCK:
        if _PROBED:
            return _DEVICE
        _PROBED = True
        preference = os.environ.get("CR_LABELER_DEVICE", "auto").lower()
        if preference == "cpu":
            _DEVICE = None
            return None
        try:
            import torch

            if not torch.cuda.is_available():
                if preference == "cuda":
                    log.warning("CR_LABELER_DEVICE=cuda but no CUDA device is available")
                _DEVICE = None
            else:
                _DEVICE = torch.device("cuda")
                log.info("using GPU for correlation: %s", torch.cuda.get_device_name(0))
        except Exception as exc:  # any failure at all means "no accelerator"
            if preference == "cuda":
                log.warning("CR_LABELER_DEVICE=cuda but torch is unusable: %s", exc)
            _DEVICE = None
        return _DEVICE


def describe() -> str:
    """One line naming the active backend, for logs."""
    dev = device()
    if dev is None:
        return "numpy (CPU)"
    import torch

    return f"torch/CUDA ({torch.cuda.get_device_name(0)})"


def _box_kernel(sigma: float, passes: int, torch, device):
    """Pillow's blur kernel, not a Gaussian one.

    ``ImageFilter.GaussianBlur`` does not convolve a Gaussian.  It runs three
    *extended box* filters whose radii approximate one (Gwosdek et al. 2011),
    and the difference is not academic: a true Gaussian of the same sigma
    disagrees with it by up to 15 grey levels on real panoramas, against 1 for
    this.  The bank was cut from Pillow's output, so this reproduces Pillow.
    """
    variance = sigma * sigma / passes
    length = math.sqrt(12.0 * variance + 1.0)
    whole = math.floor((length - 1.0) / 2.0)
    edge = (2 * whole + 1) * (whole * (whole + 1) - 3 * variance)
    edge /= 6 * (variance - (whole + 1) * (whole + 1))
    taps = [edge] + [1.0] * (2 * int(whole) + 1) + [edge]
    total = 2 * edge + 2 * whole + 1
    return torch.tensor([t / total for t in taps], dtype=torch.float32, device=device)


_KERNELS: dict[tuple[float, int], object] = {}
_BLUR_PASSES = 3


def highpass(image, sigma: float) -> np.ndarray | None:
    """``grey - blur(grey)`` on the GPU, or ``None`` to let Pillow do it.

    Reproduces Pillow's result rather than merely approximating it: same
    kernel, same pass order -- every horizontal pass before any vertical one --
    and the same rounding to whole grey levels in between.  Getting the order
    wrong costs a factor of seven in agreement (0.9979 against 0.999956), and
    what remains is one unit of rounding at worst: 99.6-99.9% of pixels come
    out bit-identical and no pixel differs by more than a single grey level.

    That is an empirical match, not an exact one, so it was checked where it
    matters rather than where it is convenient: identical labels on the ground
    truth, on all 543 hand-labelled panoramas, and on 1200 live Gen-4 ones.
    Worth 2.4x on this step at zoom 2 and 3.7x at zoom 4, ~11% of compute and
    ~9% end to end.

    ``CR_LABELER_GPU_HIGHPASS=0`` forces Pillow back for anyone who would
    rather have the byte-for-byte original.
    """
    if os.environ.get("CR_LABELER_GPU_HIGHPASS", "1") == "0":
        return None
    dev = device()
    if dev is None:
        return None
    import torch

    grey = image.convert("L")
    field = torch.from_numpy(np.array(grey, dtype=np.uint8)).to(dev).float()

    key = (sigma, _BLUR_PASSES)
    kernel = _KERNELS.get(key)
    if kernel is None:
        kernel = _box_kernel(sigma, _BLUR_PASSES, torch, dev)
        _KERNELS[key] = kernel
    radius = (len(kernel) - 1) // 2

    pad = torch.nn.functional.pad
    conv = torch.nn.functional.conv2d
    blurred = field[None, None]
    for weights, padding in (
        (kernel.view(1, 1, 1, -1), (radius, radius, 0, 0)),
        (kernel.view(1, 1, -1, 1), (0, 0, radius, radius)),
    ):
        for _ in range(_BLUR_PASSES):
            blurred = conv(pad(blurred, padding, mode="replicate"), weights)
            blurred = blurred.round().clamp_(0, 255)

    return (field - blurred[0, 0]).cpu().numpy()


class CudaCorrelator:
    """Same contract as :class:`cr_labeler.signal.Correlator`, run on the GPU.

    Intermediates stay on the device; only the finished surface comes back, so a
    panorama costs one upload and one download per anchor.
    """

    def __init__(self, field: np.ndarray):
        import torch

        self._torch = torch
        self._device = device()
        self.field = field
        self.shape = field.shape
        self._field = torch.from_numpy(np.ascontiguousarray(field)).to(self._device)
        self._spec = torch.fft.rfft2(self._field)
        self._spec_sq = None
        self._stats: dict[tuple[int, int], object] = {}

    def _variance(self, th: int, tw: int):
        cached = self._stats.get((th, tw))
        if cached is None:
            torch = self._torch
            if self._spec_sq is None:
                self._spec_sq = torch.fft.rfft2(self._field * self._field)
            window = torch.fft.rfft2(
                torch.ones((th, tw), dtype=self._field.dtype, device=self._device),
                s=self.shape,
            )
            total = torch.fft.irfft2(self._spec * window, s=self.shape)
            total_sq = torch.fft.irfft2(self._spec_sq * window, s=self.shape)
            cached = torch.clamp(total_sq - total * total / (th * tw), min=1e-6)
            self._stats[(th, tw)] = cached
        return cached

    def match(self, template: np.ndarray) -> np.ndarray:
        from .signal import normalise

        torch = self._torch
        th, tw = template.shape
        tpl = torch.from_numpy(np.ascontiguousarray(normalise(template))).to(self._device)
        kernel = torch.fft.rfft2(torch.flip(tpl, [0, 1]), s=self.shape)
        numerator = torch.fft.irfft2(self._spec * kernel, s=self.shape)
        surface = numerator / torch.sqrt(self._variance(th, tw))
        surface = torch.roll(surface, (-(th // 2), -(tw // 2)), dims=(0, 1))
        return surface.cpu().numpy()
