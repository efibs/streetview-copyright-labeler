"""Automatic tagging of Google Street View panoramas with their copyright year."""

import os

__version__ = "1.0.0"


def _limit_math_library_threads() -> None:
    """Keep BLAS to one thread per caller, before numpy loads and fixes it.

    This program runs its own pool of workers, one panorama each.  Underneath,
    ``consensus`` compares every detected instance against every other with a
    single matrix multiply, and BLAS parallelises that across every core it can
    see.  Eight workers each fanning out to 32 threads is 256 threads competing
    for 32 cores, and the cost is not subtle: the process sat at 3111% CPU of a
    possible 3200% while the GPU idled at 25%, and pinning BLAS to one thread
    took the same work from 12.90 to 19.05 panoramas a second.

    The panorama-level pool is where the parallelism belongs; it is already
    sized by measurement, and it does not need BLAS competing with it.

    Set any of these variables yourself and that choice is left alone.  This
    must run before numpy is imported, which is why it lives here rather than
    somewhere more obvious -- the libraries read it once, at load.
    """
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")


_limit_math_library_threads()
