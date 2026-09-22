"""Time conversions for Gaia epoch astrometry.

The published timestamp is an int64 count of nanoseconds since the VOTable's
``TIMESYS`` time origin.  That integer is the canonical representation and is
never overwritten; the helpers here produce *additional* derived columns.

Precision note: a float64 Julian Date spans ~2.46e6 days, so its resolution near
the Gaia epoch is about 40 ns.  Keep the integer column for anything that cares.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

#: ``TIMESYS/@timeorigin`` of the prerelease VOTable, as a Julian Date.
TIME_ORIGIN_JD = 2455197.5

#: ``TIMESYS/@timescale``.
TIME_SCALE = "tcb"

_NS_PER_DAY = 86_400_000_000_000


def tcb_ns_to_jd(ns: np.ndarray, *, origin_jd: float = TIME_ORIGIN_JD) -> np.ndarray:
    """Convert nanoseconds since the time origin to Julian Date.

    Parameters
    ----------
    ns : ndarray
        Integer nanoseconds since *origin_jd*.
    origin_jd : float
        Time origin, as a Julian Date in the same timescale.

    Returns
    -------
    ndarray
        Julian Dates as float64.

    Notes
    -----
    The division is done in two parts -- whole days and the remainder -- so that
    the result keeps sub-microsecond accuracy instead of losing it to the
    magnitude of the Julian Date.
    """
    arr = np.asarray(ns)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.asarray(np.rint(np.asarray(arr, dtype="float64")), dtype="int64")
    days, rem = np.divmod(arr, _NS_PER_DAY)
    return origin_jd + days.astype("float64") + rem.astype("float64") / _NS_PER_DAY


def tcb_ns_to_time(ns: np.ndarray, *, origin_jd: float = TIME_ORIGIN_JD) -> Time:
    """Convert nanoseconds since the time origin to an Astropy ``Time``.

    The two-part ``val``/``val2`` form is used so that no precision is lost.
    """
    arr = np.asarray(ns)
    if not np.issubdtype(arr.dtype, np.integer):
        arr = np.asarray(np.rint(np.asarray(arr, dtype="float64")), dtype="int64")
    days, rem = np.divmod(arr, _NS_PER_DAY)
    return Time(
        origin_jd + days.astype("float64"),
        rem.astype("float64") / _NS_PER_DAY,
        format="jd",
        scale=TIME_SCALE,
    )


def tcb_ns_to_jyear(ns: np.ndarray, *, origin_jd: float = TIME_ORIGIN_JD) -> np.ndarray:
    """Convert nanoseconds since the time origin to Julian years (TCB)."""
    return tcb_ns_to_time(ns, origin_jd=origin_jd).jyear
