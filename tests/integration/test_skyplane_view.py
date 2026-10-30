"""The sky-plane tab shows data without a fit, and a model only when given one."""

import numpy as np
import panel as pn
import pytest
from tests.conftest import PRERELEASE_ZIP

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.data.catalog import reference_fits
from gaia_dr4_explorer.domain import SourceContext, SourceKey
from gaia_dr4_explorer.products.astrometry import normalize_epoch_astrometry, skyplane
from gaia_dr4_explorer.ui.components import MAIN_VIEW_LABEL, object_tabs
from gaia_dr4_explorer.ui.skyplane import SkyPlaneView

pytestmark = pytest.mark.integration

HD114762 = 3937211745905473024


class _Payload:
    def __init__(self, ccd):
        self._ccd = ccd

    def table(self, name):
        assert name == "ccd"
        return self._ccd


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    provider = PreReleaseProvider(AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"), local_prerelease_zip=PRERELEASE_ZIP, allow_network=False))
    return _Payload(normalize_epoch_astrometry(provider.raw_table_for(HD114762)).ccd)


@pytest.fixture(autouse=True)
def _bokeh():
    """Drawing needs a plotting extension; without this the tests passed only
    when an earlier test had loaded one."""
    import holoviews as hv
    hv.extension("bokeh")


@pytest.fixture
def context():
    return SourceContext(key=SourceKey("Gaia DR4_RC3", HD114762))


def test_data_view_needs_no_model_and_keeps_rejected_rows(payload, context):
    view = SkyPlaneView(context=context, payload=payload)
    ccd = payload.table("ccd")
    n_finite = int(np.isfinite(np.ma.filled(ccd["centroid_pos_al"], np.nan)).sum())
    # Rejected observations stay inspectable (invariant 5).
    assert len(view.base) + view.n_unplaceable == len(ccd)
    assert len(view.base) == n_finite
    assert (~view.base["used"]).sum() > 0
    assert view._track() is None
    names = [getattr(w, "name", "") for w in view.controls().objects]
    assert "Overplot the model track" not in names
    view.sky()
    # Without a model there is no proper motion to remove: a note, not a plot.
    assert isinstance(view.sky_pm_removed(), pn.pane.HTML)
    assert isinstance(view.sky_pm_parallax_removed(), pn.pane.HTML)
    view.components()


def test_model_overlay_and_proper_motion_removal(payload, context):
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    view = SkyPlaneView(context=context, payload=payload, model=model)
    assert view._track() is not None
    raw = view._epochs()
    moved = view._epochs(pm_removed=True)
    assert view._track(pm_removed=True) is not None
    # Removing -582 mas/yr over +-2.5 yr collapses the RA span to the parallax loop.
    used = raw["used"]
    assert np.ptp(raw.loc[used, "dra"]) > 2000
    assert np.ptp(moved.loc[used, "dra"]) < 80
    # The data themselves are never altered.
    assert view.base["dra"].equals(raw["dra"])
    view.show_model = False
    assert view._track() is None
    view.sky()
    view.sky_pm_removed()
    view.sky_pm_parallax_removed()
    view.components()


def test_main_view_is_the_first_tab_and_returns_to_astrometry():
    tabs = object_tabs(("Overview", "a"), ("Astrometry", "b"), ("Sky plane", "c"))
    assert tabs._names[0] == MAIN_VIEW_LABEL
    assert tabs.active == 2, "opens on Astrometry"
    tabs.active = 3          # the user goes to the sky plane...
    tabs.active = 0          # ...and picks "Main view"
    assert tabs.active == 2


def _sky_figures(view):
    """The Bokeh figures of the two sky panels, rendered in one document."""
    import holoviews as hv
    from bokeh.models import Plot

    hv.extension("bokeh")
    root = view.layout().get_root()
    figs = [m for m in root.references() if isinstance(m, Plot)
            and m.xaxis and "Δα*" in str(m.xaxis[0].axis_label)]
    return figs


def test_sky_panels_do_not_share_ranges(payload, context):
    """Regression: HoloViews links plots with the same dimension names, so the
    proper-motion-removed panel inherited the as-observed ranges and HD 114762's
    parallax ellipse shrank to a dot."""
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    figs = _sky_figures(SkyPlaneView(context=context, payload=payload, model=model))
    assert len(figs) == 3
    for i, a in enumerate(figs):
        for b in figs[i + 1:]:
            assert a.x_range is not b.x_range and a.y_range is not b.y_range
    spans = sorted(abs(f.x_range.end - f.x_range.start) for f in figs)
    # As observed the track spans ~2900 mas in RA; the parallax ellipse ~55 mas;
    # with parallax removed the orbit residuals span a few mas.
    assert spans[2] > 2500 and 20 < spans[1] < 200 and spans[0] < 20, spans


def test_sky_panels_have_equal_scale_on_both_axes(payload, context):
    from gaia_dr4_explorer.products.astrometry.plots import SKY_FRAME

    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    for f in _sky_figures(SkyPlaneView(context=context, payload=payload, model=model)):
        # A stretching container overrides the frame, and with it the scale.
        assert f.frame_width == SKY_FRAME[0] and f.frame_height == SKY_FRAME[1]
        assert not str(f.sizing_mode or "").startswith("stretch"), f.sizing_mode
        mx = abs(f.x_range.end - f.x_range.start) / SKY_FRAME[0]
        my = abs(f.y_range.end - f.y_range.start) / SKY_FRAME[1]
        assert mx == pytest.approx(my, rel=1e-6), "mas per pixel differs between axes"


def test_time_panels_are_sized_and_independent(payload, context):
    """Regression: an hv.Layout's GridPlot has no size and collapsed to zero
    width in the browser; and a linked "dra" y axis would take panel 1's range."""
    import holoviews as hv
    from bokeh.models import GridPlot, Plot

    hv.extension("bokeh")
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    view = SkyPlaneView(context=context, payload=payload, model=model)
    root = view.layout().get_root()
    refs = list(root.references())
    assert not any(isinstance(m, GridPlot) for m in refs)
    figs = [m for m in refs if isinstance(m, Plot)]
    time_figs = [f for f in figs if "time" in str(f.xaxis[0].axis_label)]
    sky_figs = [f for f in figs if "Δα*" in str(f.xaxis[0].axis_label)]
    assert len(time_figs) == 2 and len(sky_figs) == 3
    for f in time_figs:
        assert f.frame_width > 0 and f.frame_height > 0
        assert all(f.y_range is not g.x_range for g in sky_figs)


def test_panels_are_arranged_in_two_rows(payload, context):
    """Sky panels side by side on top, time panels side by side below; each
    pair wraps instead of overflowing a narrow window."""
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    view = SkyPlaneView(context=context, payload=payload, model=model)
    boxes = [o for o in view.panel().objects if isinstance(o, pn.FlexBox)]
    assert boxes and len(boxes[0].objects) == 3, "sky panels are not side by side"
    assert boxes[0].flex_wrap == "wrap"
    comps = view.components()
    assert isinstance(comps, pn.FlexBox) and len(comps.objects) == 2
    assert comps.flex_wrap == "wrap"
    assert isinstance(view.layout()[0], pn.FlexBox), "controls should be a strip above"


def test_sky_controls_are_one_line_of_individually_sized_widgets(payload, context):
    """Regression: pn.Param stacked the checkboxes one per line."""
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    controls = SkyPlaneView(context=context, payload=payload, model=model).controls()
    assert controls.flex_wrap == "wrap"
    boxes = [w for w in controls.objects if isinstance(w, pn.widgets.Checkbox)]
    assert [b.name for b in boxes] == [
        "Overplot the model track", "Show 1-D constraint lines",
        "Show along-scan ±1σ bars", "Show observations rejected by AGIS"]
    assert all(b.width and b.width < 320 for b in boxes)
    # The four checkboxes fit in one line of the ~1090 px pane at 1512 px.
    assert sum(b.width for b in boxes) < 1000


HD183633 = 4181040337841125632
BH3 = 4318465066420528000


def _view(tmp_path_factory, sid):
    provider = PreReleaseProvider(AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"), local_prerelease_zip=PRERELEASE_ZIP, allow_network=False))
    payload = _Payload(normalize_epoch_astrometry(provider.raw_table_for(sid)).ccd)
    model = skyplane.SkyModel.from_reference(reference_fits()[sid])
    return SkyPlaneView(context=SourceContext(key=SourceKey("Gaia DR4_RC3", sid)), payload=payload, model=model)


def test_removing_parallax_leaves_noise_for_a_single_star_and_the_orbit_for_bh3(tmp_path_factory):
    """Panel 3 subtracts proper motion and parallax from the data. For a single
    star the along-scan residual is the measurement noise; for Gaia BH3 it is the
    orbit, tens of times larger."""
    def along_scan_rms(sid):
        view = _view(tmp_path_factory, sid)
        e = view._epochs(pm_removed=True, parallax_removed=True)
        u = e[e["used"]]
        w = skyplane.along_scan(u["dra"] - view.model.delta_alpha_star,
                                u["ddec"] - view.model.delta_delta, u["scan_pos_angle"])
        return float(np.std(w)), float(np.median(u["centroid_pos_error_al"]))

    rms, sigma = along_scan_rms(HD183633)
    assert rms < 1.5 * sigma, (rms, sigma)
    rms_bh3, sigma_bh3 = along_scan_rms(BH3)
    assert rms_bh3 > 30 * sigma_bh3, (rms_bh3, sigma_bh3)


def test_parallax_removal_is_pm_removal_minus_the_published_factors(payload, context):
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    view = SkyPlaneView(context=context, payload=payload, model=model)
    pm = view._epochs(pm_removed=True)
    both = view._epochs(pm_removed=True, parallax_removed=True)
    px, py = skyplane.local_plane_to_sky(
        pm["parallax_factor_al"], pm["parallax_factor_ac"], pm["scan_pos_angle"])
    ok = np.isfinite(px)
    assert np.allclose(both["dra"][ok], (pm["dra"] - model.parallax * px)[ok])
    assert np.allclose(both["ddec"][ok], (pm["ddec"] - model.parallax * py)[ok])
    # Rows with no parallax factor cannot be placed after its removal; none is used by AGIS.
    assert both["dra"][~ok].isna().all() and not both["used"][~ok].any()
    # The track of panel 3 is the constant reference offset.
    t, x, y = view._track(pm_removed=True, parallax_removed=True)
    assert np.allclose(x, model.delta_alpha_star) and np.allclose(y, model.delta_delta)


def test_three_sky_panels_fit_side_by_side_at_1512_px():
    """Measured in the local app at a 1512 px window: the row is ~1048 px wide
    (1060 px of panels wrapped the third one)."""
    from gaia_dr4_explorer.ui import skyplane as ui

    total = 2 * ui.SKY_PANEL_WIDTH + ui.SKY_PANEL_WIDTH_WITH_COLORBAR + 2 * ui.PANEL_GAP[1]
    assert total <= 1040, total
