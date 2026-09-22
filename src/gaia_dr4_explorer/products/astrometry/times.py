"""Time conversions for Gaia epoch astrometry.

The published timestamp is an int64 count of nanoseconds since the VOTable's
``TIMESYS`` time origin.  That integer is the canonical representation and is
never overwritten; the helpers here produce *additional* derived columns.

Precision note: a float64 Julian Date spans ~2.46e6 days, so its resolution near
the Gaia epoch is about 40 ns.  Keep the integer column for anything that cares.

Time reference, settled from the draft Gaia DR4 data model (2026-06-26):

    obs_time_bary_corr : "Barycentric correction to the observation time, in
    the sense of TCB(barycentric) - TCB(at Gaia), calculated for the obsTime of
    the AF4 CCD, i.e. at the middle of the FoV transit."

So ``obs_time_tcb`` is **TCB at Gaia**, not barycentric, and

    t_barycentric = obs_time_tcb + obs_time_bary_corr

The prerelease VOTable declares ``TIMESYS/@refposition="BARYCENTER"`` for this
table, which contradicts that and should not be believed.  ``gaiasupdate``
agrees with the data model: ``set_relative_time()`` adds the correction.

Note the correction is computed at AF4, the middle of the transit, so it is
exact there and approximate for the other CCDs of the same transit.
"""

from __future__ import annotations

import numpy as np
from astropy.time import Time

#: ``TIMESYS/@timeorigin`` of the prerelease VOTable, as a Julian Date.
TIME_ORIGIN_JD = 2455197.5

#: ``TIMESYS/@timescale``.
TIME_SCALE = "tcb"

#: Gaia DR4 astrometric reference epoch, from gaiasupdate.constants
#: (``DR4_REFERENCE_EPOCH = Time('2017.5', format='jyear', scale='tcb')``).
#: DR3 used J2016.0; do not carry that assumption over.
DR4_REFERENCE_EPOCH_JYEAR = 2017.5

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


def barycentric_ns(obs_time_tcb: np.ndarray, obs_time_bary_corr: np.ndarray) -> np.ndarray:
    """Apply the barycentric correction to an at-Gaia TCB timestamp.

    Parameters
    ----------
    obs_time_tcb : ndarray
        Per-CCD observation time, nanoseconds since the origin, TCB at Gaia.
    obs_time_bary_corr : ndarray
        Per-transit correction in nanoseconds, in the sense
        TCB(barycentric) - TCB(at Gaia), broadcast to the CCD rows.

    Returns
    -------
    ndarray
        Barycentric TCB, nanoseconds since the origin, as float64 so that a
        missing correction can stay missing as NaN rather than becoming zero.
    """
    at_gaia = np.asarray(obs_time_tcb, dtype="float64")
    correction = np.asarray(obs_time_bary_corr, dtype="float64")
    return at_gaia + correction


def relative_time_year(
    obs_time_tcb: np.ndarray,
    obs_time_bary_corr: np.ndarray,
    *,
    reference_epoch_jyear: float = DR4_REFERENCE_EPOCH_JYEAR,
    origin_jd: float = TIME_ORIGIN_JD,
) -> np.ndarray:
    """Barycentric TCB time in years relative to the DR4 reference epoch.

    Matches ``gaiasupdate``'s ``relative_time_year``: zero corresponds to
    J2017.5 TCB, barycentrically corrected.
    """
    ns = barycentric_ns(obs_time_tcb, obs_time_bary_corr)
    jd = origin_jd + ns / (_NS_PER_DAY)
    return Time(jd, format="jd", scale=TIME_SCALE).jyear - reference_epoch_jyear
