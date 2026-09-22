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


# ------------------------------------------------ the across-scan axis

HD114762 = 3937211745905473024
#: Sources whose tracks span hundreds of mas, where a wrong across-scan sign
#: cannot hide: HD 114762 (2865 mas), Gaia BH3 (759), and two more.
LARGE_TRACK = (HD114762, BH3, 1663617687609809280, 1457486023639239296)


def _used_with_model(provider, sid):
    raw = provider.raw_table_for(sid)
    result = fit_dr4_like_single_source(raw, sid)
    ccd = normalize_epoch_astrometry(raw).ccd
    f = lambda n: np.asarray(np.ma.filled(ccd[n], np.nan), dtype="float64")  # noqa: E731
    used = np.asarray(np.ma.filled(ccd["used_by_agis_al"], False), dtype=bool)
    ns = np.asarray(np.ma.filled(ccd["obs_time_tcb"], 0), dtype="int64")[used]
    plx = skyplane.parallax_displacement(tcb_ns_to_time(ns), f("ra0")[used], f("dec0")[used])
    dra, ddec = skyplane.model_offsets(
        skyplane.SkyModel.from_fit(result), f("relative_time_year")[used], *plx)
    return {
        "theta": f("scan_pos_angle")[used], "w": f("centroid_pos_al")[used],
        "z": f("calculated_pos_ac")[used], "dra": dra, "ddec": ddec,
    }


@pytest.mark.parametrize("sid", LARGE_TRACK)
def test_across_scan_sign_reproduces_the_published_calculated_pos_ac(provider, sid):
    """calculated_pos_ac is AGIS's own across-scan prediction. Our refitted
    model, projected with the chosen sign, must reproduce it; the opposite
    sign must miss by the size of the track. Measured: 0.07-0.23 mas against
    160-1473 mas."""
    d = _used_with_model(provider, sid)
    right = skyplane.across_scan(d["dra"], d["ddec"], d["theta"])
    rms = lambda v: float(np.sqrt(np.mean((v - d["z"]) ** 2)))  # noqa: E731
    assert rms(right) < 0.3, f"{sid}: rms {rms(right):.3f} mas against calculated_pos_ac"
    assert rms(-right) > 100.0, f"{sid}: flipped sign not rejected ({rms(-right):.1f} mas)"


@pytest.mark.parametrize("sid", LARGE_TRACK)
def test_published_coordinates_land_on_the_model_track(provider, sid):
    """Placing each observation from (w, z) alone -- no fit involved -- puts it
    on the refitted track across the scan, and along the scan to within what
    a five-parameter model can explain.

    Gaia BH3 is the exception by design: its measured along-scan positions
    leave the single-star track by several mas, which is the black hole's
    orbit. The across-scan coordinate does not show it, because
    calculated_pos_ac is itself a five-parameter AGIS prediction.
    """
    d = _used_with_model(provider, sid)
    dra, ddec = skyplane.local_plane_to_sky(d["w"], d["z"], d["theta"])
    ddx, ddy = dra - d["dra"], ddec - d["ddec"]
    along = skyplane.along_scan(ddx, ddy, d["theta"])
    across = skyplane.across_scan(ddx, ddy, d["theta"])
    rms = lambda v: float(np.sqrt(np.mean(v ** 2)))  # noqa: E731
    assert rms(across) < 0.3, f"{sid}: across-scan rms {rms(across):.3f} mas"
    if sid == BH3:
        assert rms(along) > 3.0, f"BH3 orbit not visible: along-scan rms {rms(along):.2f}"
    else:
        assert rms(along) < 1.5, f"{sid}: along-scan rms {rms(along):.3f} mas"


def test_local_plane_rotation_inverts_exactly():
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 360, 50)
    w, z = rng.normal(0, 100, 50), rng.normal(0, 100, 50)
    dra, ddec = skyplane.local_plane_to_sky(w, z, theta)
    assert skyplane.along_scan(dra, ddec, theta) == pytest.approx(w, abs=1e-9)
    assert skyplane.across_scan(dra, ddec, theta) == pytest.approx(z, abs=1e-9)
    assert np.hypot(dra, ddec) == pytest.approx(np.hypot(w, z), abs=1e-9)


def test_missing_coordinates_stay_missing():
    dra, ddec = skyplane.local_plane_to_sky([np.nan, 1.0], [0.0, np.nan], [10.0, 10.0])
    assert np.isnan(dra).all() and np.isnan(ddec).all()


def test_error_bar_lies_along_the_scan_with_length_two_sigma():
    import pandas as pd

    frame = pd.DataFrame({
        "centroid_pos_al": [3.0, -1.0], "calculated_pos_ac": [2.0, 5.0],
        "scan_pos_angle": [30.0, 200.0], "centroid_pos_error_al": [0.4, 0.1],
    })
    e = skyplane.epoch_sky_positions(frame)
    theta = frame["scan_pos_angle"].to_numpy()
    ux, uy = skyplane.scan_unit_vector(theta)
    dx = (e["ex1"] - e["ex0"]).to_numpy()
    dy = (e["ey1"] - e["ey0"]).to_numpy()
    assert np.hypot(dx, dy) == pytest.approx([0.8, 0.2], abs=1e-12)
    assert (dx * -uy + dy * ux) == pytest.approx([0.0, 0.0], abs=1e-12)
    w = skyplane.along_scan(e["dra"].to_numpy(), e["ddec"].to_numpy(), theta)
    assert w == pytest.approx([3.0, -1.0], abs=1e-12)
    assert "dra" not in frame, "the input frame must not be modified"
