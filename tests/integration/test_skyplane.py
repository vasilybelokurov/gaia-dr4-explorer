"""The sky-plane transformation, validated against published Gaia quantities.

PLAN.md forbids building a tangent-plane view by guessing the scan-angle sign
convention. These tests are the validation that permits it: the convention is
checked against the published parallax_factor_al and against the fit residuals,
both of which come from Gaia, not from us.
"""

import numpy as np
import pytest
from tests.conftest import PRERELEASE_ZIP

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.products.astrometry import normalize_epoch_astrometry, skyplane
from gaia_dr4_explorer.products.astrometry.fitting import fit_dr4_like_single_source
from gaia_dr4_explorer.products.astrometry.times import tcb_ns_to_time

pytestmark = pytest.mark.integration

HD183633 = 4181040337841125632
BH3 = 4318465066420528000
QSO = 2237987199365376


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    return PreReleaseProvider(
        AppConfig(cache_dir=tmp_path_factory.mktemp("cache"),
                  local_prerelease_zip=PRERELEASE_ZIP, allow_network=False)
    )


def _transits(provider, sid):
    """One row per transit: scan angle, published parallax factor, time, reference point."""
    ccd = normalize_epoch_astrometry(provider.raw_table_for(sid)).ccd
    f = lambda n: np.asarray(np.ma.filled(ccd[n], np.nan), dtype="float64")  # noqa: E731
    tid = f("transit_id")
    good = np.isfinite(f("parallax_factor_al")) & np.isfinite(f("scan_pos_angle"))
    _, first = np.unique(tid[good], return_index=True)
    sel = np.flatnonzero(good)[first]
    ns = np.asarray(np.ma.filled(ccd["obs_time_tcb"], 0), dtype="int64")[sel]
    return {
        "time": tcb_ns_to_time(ns),
        "theta": f("scan_pos_angle")[sel],
        "p_al": f("parallax_factor_al")[sel],
        "ra0": f("ra0")[sel],
        "dec0": f("dec0")[sel],
    }


@pytest.mark.parametrize("sid", [HD183633, BH3, QSO, 1663617687609809280])
def test_parallax_convention_reproduces_the_published_factor(provider, sid):
    """The decisive check: our tangent-plane formula must reproduce Gaia's own
    parallax_factor_al. Any sign error shows up immediately as anticorrelation."""
    d = _transits(provider, sid)
    dra, ddec = skyplane.parallax_displacement(d["time"], d["ra0"], d["dec0"])
    predicted = skyplane.along_scan(dra, ddec, d["theta"])
    correlation = float(np.corrcoef(predicted, d["p_al"])[0, 1])
    rms = float(np.sqrt(np.mean((predicted - d["p_al"]) ** 2)))
    assert correlation > 0.9999, f"{sid}: correlation {correlation:+.5f}"
    # The residual is Gaia's ~0.01 AU offset from Earth at L2, not a sign error.
    assert rms < 0.02, f"{sid}: rms {rms:.4f} against the published factor"


@pytest.mark.parametrize("sid", [HD183633, QSO])
def test_every_plausible_wrong_convention_is_rejected(provider, sid):
    """Guards the guard.

    Correlation alone does NOT discriminate: sin/cos swapped still correlates
    at 0.87-0.98 with the published factor, so validating on correlation could
    have shipped a wrong sign. The rms does discriminate, by a factor of 50-200.
    """
    d = _transits(provider, sid)
    dra, ddec = skyplane.parallax_displacement(d["time"], d["ra0"], d["dec0"])
    theta = np.deg2rad(d["theta"])
    right = skyplane.along_scan(dra, ddec, d["theta"])
    wrong = {
        "sin/cos swapped": dra * np.cos(theta) + ddec * np.sin(theta),
        "overall sign flipped": -(dra * np.sin(theta) + ddec * np.cos(theta)),
        "declination sign flipped": dra * np.sin(theta) - ddec * np.cos(theta),
    }
    rms = lambda v: float(np.sqrt(np.mean((v - d["p_al"]) ** 2)))  # noqa: E731
    assert rms(right) < 0.02
    for name, variant in wrong.items():
        assert rms(variant) > 0.1, f"{name} was not rejected (rms {rms(variant):.4f})"
        assert rms(variant) > 20 * rms(right), name


def test_along_scan_matches_the_gaiasupdate_design_matrix(provider):
    """w = dra sin(theta) + ddec cos(theta) is exactly the official design row."""
    theta = np.array([0.0, 90.0, 180.0, 270.0])
    assert skyplane.along_scan(np.ones(4), np.zeros(4), theta) == pytest.approx(
        [0.0, 1.0, 0.0, -1.0], abs=1e-12
    )
    assert skyplane.along_scan(np.zeros(4), np.ones(4), theta) == pytest.approx(
        [1.0, 0.0, -1.0, 0.0], abs=1e-12
    )


def test_model_reproduces_the_fit_residuals(provider):
    """Round trip: evaluating our sky model along the scan must return the same
    residuals gaiasupdate reported, to well within the measurement precision."""
    raw = provider.raw_table_for(HD183633)
    result = fit_dr4_like_single_source(raw, HD183633)
    ccd = normalize_epoch_astrometry(raw).ccd

    f = lambda n: np.asarray(np.ma.filled(ccd[n], np.nan), dtype="float64")  # noqa: E731
    used = np.asarray(np.ma.filled(ccd["used_by_agis_al"], False), dtype=bool)
    theta = f("scan_pos_angle")[used]
    w_obs = f("centroid_pos_al")[used]
    t_year = f("relative_time_year")[used]
    ns = np.asarray(np.ma.filled(ccd["obs_time_tcb"], 0), dtype="int64")[used]
    times = tcb_ns_to_time(ns)

    model = skyplane.SkyModel.from_fit(result)
    plx_dra, plx_ddec = skyplane.parallax_displacement(times, f("ra0")[used], f("dec0")[used])
    dra, ddec = skyplane.model_offsets(model, t_year, plx_dra, plx_ddec)
    w_model = skyplane.along_scan(dra, ddec, theta)

    ours = w_obs - w_model
    theirs = np.asarray(result.residuals, dtype="float64")
    assert ours.size == theirs.size
    # Our parallax factor comes from Earth, Gaia's from the spacecraft: the
    # difference is bounded by 0.01 AU x parallax, tiny for a 1 mas source.
    assert float(np.std(ours - theirs)) < 0.05, "sky model disagrees with the official fit"


def test_measurement_is_displaced_only_along_the_scan(provider):
    """The perpendicular coordinate is model, not data, and must not move."""
    theta = np.array([37.0, 120.0])
    mdra, mddec = np.array([1.0, -2.0]), np.array([0.5, 3.0])
    residual = np.array([0.3, -0.4])
    pdra, pddec = skyplane.measured_positions(mdra, mddec, residual, theta)
    ux, uy = skyplane.scan_unit_vector(theta)
    moved_along = (pdra - mdra) * ux + (pddec - mddec) * uy
    moved_perp = (pdra - mdra) * (-uy) + (pddec - mddec) * ux
    assert moved_along == pytest.approx(residual, abs=1e-12)
    assert moved_perp == pytest.approx([0.0, 0.0], abs=1e-12)


def test_constraint_segment_is_perpendicular_to_the_scan(provider):
    theta = np.array([0.0, 45.0, 90.0])
    x0, y0, x1, y1 = skyplane.constraint_segments(
        np.zeros(3), np.zeros(3), theta, half_length_mas=1.0
    )
    ux, uy = skyplane.scan_unit_vector(theta)
    along = (x1 - x0) * ux + (y1 - y0) * uy
    assert along == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
    assert np.hypot(x1 - x0, y1 - y0) == pytest.approx([2.0, 2.0, 2.0], abs=1e-12)


def test_track_shows_parallax_for_a_nearby_star_and_not_for_a_qso(provider):
    """A physical check: the 25.6 mas parallax star must loop, the QSO must not."""
    for sid, expect_loops in ((3937211745905473024, True), (QSO, False)):
        result = fit_dr4_like_single_source(provider.raw_table_for(sid), sid)
        model = skyplane.SkyModel.from_fit(result)
        ccd = normalize_epoch_astrometry(provider.raw_table_for(sid)).ccd
        ra0 = float(np.asarray(ccd["ra0"])[0])
        dec0 = float(np.asarray(ccd["dec0"])[0])
        _, dra, ddec = skyplane.smooth_track(model, -0.5, 0.5, ra0, dec0, n=200)
        # Wobble about the straight proper-motion line, in mas.
        line_dra = np.linspace(dra[0], dra[-1], dra.size)
        line_ddec = np.linspace(ddec[0], ddec[-1], ddec.size)
        wobble = float(np.max(np.hypot(dra - line_dra, ddec - line_ddec)))
        if expect_loops:
            assert wobble > 5.0, f"{sid}: expected parallax loops, wobble {wobble:.2f} mas"
        else:
            assert wobble < 1.0, f"{sid}: QSO should not loop, wobble {wobble:.2f} mas"
