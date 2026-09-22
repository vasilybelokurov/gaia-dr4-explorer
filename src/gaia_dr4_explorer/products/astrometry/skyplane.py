"""Sky-plane reconstruction of Gaia epoch astrometry.

Gaia's epoch astrometry is **one-dimensional**: each CCD observation measures
only the along-scan local-plane coordinate ``w`` (``centroid_pos_al``); the
measured across-scan centroid is not published in this release. What *is*
published, for every CCD, is ``calculated_pos_ac`` -- the across-scan
coordinate ``z`` that the AGIS solution predicts. The pair (w, z) is therefore
Gaia's own placement of each observation on the sky: w measured, z from the
catalogue model. This module turns that pair into tangent-plane offsets, and
keeps it apart from our own refitted model, which is drawn over it.

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

The across-scan axis, **validated** against the published ``calculated_pos_ac``
(see ``tests/integration/test_skyplane.py``):

    z = -dra * cos(theta) + ddec * sin(theta)

so that, inverting the rotation,

    dra  = w sin(theta) - z cos(theta)
    ddec = w cos(theta) + z sin(theta)

With this sign the refitted model reproduces ``calculated_pos_ac`` to
0.000-0.23 mas rms on all twelve prerelease sources; the opposite sign misses
by up to 1473 mas (HD 114762, whose track spans 2865 mas).

``w`` here is the *published* ``centroid_pos_al``. The observation
``gaiasupdate`` fits also carries its colour correction, which differs from the
raw value by 0.005-0.21 mas rms across the sample; that correction stays inside
``gaiasupdate`` (CLAUDE.md invariant 12).
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


    @classmethod
    def from_reference(cls, values: dict[str, float]) -> SkyModel:
        """Build from a row of the precomputed reference table."""
        return cls(
            delta_alpha_star=values["fit_delta_alpha_star_mas"],
            delta_delta=values["fit_delta_delta_mas"],
            parallax=values["fit_parallax_mas"],
            pmra_star=values["fit_pmra_mas_yr"],
            pmdec=values["fit_pmdec_mas_yr"],
        )

    def without_proper_motion(self) -> SkyModel:
        """The same model with the linear motion set to zero."""
        return SkyModel(self.delta_alpha_star, self.delta_delta, self.parallax, 0.0, 0.0)


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


def across_scan(dra: np.ndarray, ddec: np.ndarray, scan_pos_angle_deg: np.ndarray) -> np.ndarray:
    """Project a tangent-plane offset onto the across-scan axis.

    ``z = -dra cos(theta) + ddec sin(theta)``: the sign that reproduces the
    published ``calculated_pos_ac``.
    """
    theta = np.deg2rad(np.asarray(scan_pos_angle_deg, dtype="float64"))
    return -np.asarray(dra) * np.cos(theta) + np.asarray(ddec) * np.sin(theta)


def local_plane_to_sky(
    w_mas: np.ndarray, z_mas: np.ndarray, scan_pos_angle_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Local-plane (w, z) to tangent-plane (dra, ddec) offsets, in mas.

    The inverse of :func:`along_scan` and :func:`across_scan`. Missing inputs
    stay missing: a NaN in ``w`` or ``z`` gives NaN offsets.
    """
    theta = np.deg2rad(np.asarray(scan_pos_angle_deg, dtype="float64"))
    w = np.asarray(w_mas, dtype="float64")
    z = np.asarray(z_mas, dtype="float64")
    s, c = np.sin(theta), np.cos(theta)
    return w * s - z * c, w * c + z * s


def epoch_sky_positions(frame, *, error_scale: float = 1.0):
    """Add each CCD observation's sky position to a flattened CCD frame.

    Parameters
    ----------
    frame : DataFrame
        Needs ``centroid_pos_al``, ``calculated_pos_ac``, ``scan_pos_angle``
        and ``centroid_pos_error_al``. Not modified.
    error_scale : float
        Multiplier for the along-scan error bar.

    Returns
    -------
    DataFrame
        A copy with ``dra``, ``ddec`` (mas) and the along-scan error-bar
        endpoints ``ex0, ey0, ex1, ey1``. The error bar lies along the scan,
        the only direction the observation measures.
    """
    out = frame.copy()
    theta = out["scan_pos_angle"].to_numpy(dtype="float64")
    dra, ddec = local_plane_to_sky(
        out["centroid_pos_al"].to_numpy(dtype="float64"),
        out["calculated_pos_ac"].to_numpy(dtype="float64"),
        theta,
    )
    ux, uy = scan_unit_vector(theta)
    sig = error_scale * out["centroid_pos_error_al"].to_numpy(dtype="float64")
    out["dra"], out["ddec"] = dra, ddec
    out["ex0"], out["ey0"] = dra - sig * ux, ddec - sig * uy
    out["ex1"], out["ey1"] = dra + sig * ux, ddec + sig * uy
    return out


def scan_unit_vector(scan_pos_angle_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vector along the scan, in (dra, ddec)."""
    theta = np.deg2rad(np.asarray(scan_pos_angle_deg, dtype="float64"))
    return np.sin(theta), np.cos(theta)


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
