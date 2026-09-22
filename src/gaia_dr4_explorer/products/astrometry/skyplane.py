"""Sky-plane reconstruction of Gaia epoch astrometry.

Gaia's epoch astrometry is **one-dimensional**: each CCD observation measures
only the along-scan coordinate, and the across-scan centroid is not even
published in this release. So there is no measured (RA, Dec) per epoch, and any
view that draws one is showing a model, not data. This module keeps the two
apart: the fitted track is a model, and each measurement is drawn as the 1-D
constraint it actually is.

Conventions, taken from the official ``gaiasupdate`` design matrix and
**validated** against the published ``parallax_factor_al`` (see
``tests/unit/test_skyplane.py``), not assumed:

    theta = scan_pos_angle in radians
    w     = dra * sin(theta) + ddec * cos(theta)
          + parallax * p_AL
          + pmra * t * sin(theta) + pmdec * t * cos(theta)

where ``dra`` is the offset in right ascension *already multiplied by
cos(dec)*, ``ddec`` the offset in declination, both in mas, and ``t`` is
barycentric TCB in years from the DR4 reference epoch J2017.5.

The parallactic displacement per unit parallax, for an observer at barycentric
position ``b`` in AU and a source at ``(alpha, delta)``:

    dra_plx  = b_x sin(alpha) - b_y cos(alpha)
    ddec_plx = b_x cos(alpha) sin(delta) + b_y sin(alpha) sin(delta)
               - b_z cos(delta)

Projected on the scan direction this reproduces the published
``parallax_factor_al`` with a correlation of 1.00000 and an rms of 0.004-0.006,
the residual being Gaia's ~0.01 AU offset from Earth at L2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

#: Body used for the barycentric observer position. Gaia orbits L2, about
#: 0.01 AU from Earth, which is the size of the residual quoted above. Passing
#: the spacecraft ephemeris would remove it; Earth is enough to draw a track.
OBSERVER_BODY = "earth"


@dataclass(frozen=True)
class SkyModel:
    """A five-parameter astrometric model in the tangent plane.

    All offsets are in mas relative to the transit reference point
    ``(ra0, dec0)``; proper motions are mas/yr; ``t`` is years from J2017.5.
    """

    delta_alpha_star: float
    delta_delta: float
    parallax: float
    pmra_star: float
    pmdec: float

    @classmethod
    def from_fit(cls, result) -> SkyModel:
        """Build from a :class:`SourceUpdateResult`."""
        p = result.as_dict()
        return cls(
            delta_alpha_star=p["delta_alpha_star"][0],
            delta_delta=p["delta_delta"][0],
            parallax=p["parallax"][0],
            pmra_star=p["pmra_star"][0],
            pmdec=p["pmdec"][0],
        )


def parallax_displacement(
    times: Time, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Parallactic displacement per unit parallax, in the tangent plane.

    Returns
    -------
    (dra, ddec) : ndarray, ndarray
        Dimensionless factors. Multiply by the parallax in mas to get an
        offset in mas. ``dra`` is already the great-circle offset, i.e. it
        includes the cos(dec) factor.
    """
    body = get_body_barycentric(OBSERVER_BODY, times)
    bx = body.x.to("AU").value
    by = body.y.to("AU").value
    bz = body.z.to("AU").value
    a = np.deg2rad(np.asarray(ra_deg, dtype="float64"))
    d = np.deg2rad(np.asarray(dec_deg, dtype="float64"))
    dra = bx * np.sin(a) - by * np.cos(a)
    ddec = bx * np.cos(a) * np.sin(d) + by * np.sin(a) * np.sin(d) - bz * np.cos(d)
    return dra, ddec


def model_offsets(
    model: SkyModel,
    t_year: np.ndarray,
    plx_dra: np.ndarray,
    plx_ddec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Modelled (dra, ddec) offsets in mas at each time.

    Parameters
    ----------
    t_year : ndarray
        Barycentric TCB years from the DR4 reference epoch.
    plx_dra, plx_ddec : ndarray
        Parallax factors from :func:`parallax_displacement` at those times.
    """
    t = np.asarray(t_year, dtype="float64")
    dra = model.delta_alpha_star + model.pmra_star * t + model.parallax * plx_dra
    ddec = model.delta_delta + model.pmdec * t + model.parallax * plx_ddec
    return dra, ddec


def along_scan(dra: np.ndarray, ddec: np.ndarray, scan_pos_angle_deg: np.ndarray) -> np.ndarray:
    """Project a tangent-plane offset onto the scan direction.

    This is the quantity Gaia measures: ``w = dra sin(theta) + ddec cos(theta)``.
    """
    theta = np.deg2rad(np.asarray(scan_pos_angle_deg, dtype="float64"))
    return np.asarray(dra) * np.sin(theta) + np.asarray(ddec) * np.cos(theta)


def scan_unit_vector(scan_pos_angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector along the scan, in (dra, ddec)."""
    theta = np.deg2rad(np.asarray(scan_pos_angle_deg, dtype="float64"))
    return np.sin(theta), np.cos(theta)


def measured_positions(
    model_dra: np.ndarray,
    model_ddec: np.ndarray,
    residual_mas: np.ndarray,
    scan_pos_angle_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Where each measurement places the source, along its own scan direction.

    A measurement constrains the position only along the scan, so the honest
    placement is the model position displaced by the along-scan residual. The
    perpendicular coordinate is unmeasured and is *taken from the model* -- it
    is not data.
    """
    ux, uy = scan_unit_vector(scan_pos_angle_deg)
    r = np.asarray(residual_mas, dtype="float64")
    return np.asarray(model_dra) + r * ux, np.asarray(model_ddec) + r * uy


def constraint_segments(
    position_dra: np.ndarray,
    position_ddec: np.ndarray,
    scan_pos_angle_deg: np.ndarray,
    half_length_mas: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Endpoints of the line each 1-D measurement actually constrains.

    The locus of sky positions consistent with one along-scan measurement is a
    line *perpendicular* to the scan direction. Drawing it keeps the
    one-dimensionality of the data visible.

    Returns
    -------
    (x0, y0, x1, y1) : ndarray
        Segment endpoints in (dra, ddec) mas.
    """
    ux, uy = scan_unit_vector(scan_pos_angle_deg)
    # Perpendicular to the scan direction.
    px, py = -uy, ux
    x = np.asarray(position_dra, dtype="float64")
    y = np.asarray(position_ddec, dtype="float64")
    return (
        x - px * half_length_mas, y - py * half_length_mas,
        x + px * half_length_mas, y + py * half_length_mas,
    )


def smooth_track(
    model: SkyModel, t_start: float, t_end: float, ra_deg: float, dec_deg: float,
    *, reference_epoch_jyear: float = 2017.5, n: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A densely sampled model track, for drawing the parallax loops.

    Returns
    -------
    (t_year, dra, ddec)
    """
    t = np.linspace(float(t_start), float(t_end), int(n))
    times = Time(reference_epoch_jyear + t, format="jyear", scale="tcb")
    plx_dra, plx_ddec = parallax_displacement(
        times, np.full(t.shape, ra_deg), np.full(t.shape, dec_deg)
    )
    dra, ddec = model_offsets(model, t, plx_dra, plx_ddec)
    return t, dra, ddec
