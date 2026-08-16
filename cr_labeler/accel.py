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
