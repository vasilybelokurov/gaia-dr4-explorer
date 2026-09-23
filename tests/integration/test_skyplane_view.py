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
    assert "show_model" not in view.controls()[1].parameters
    view.sky()
    # Without a model there is no proper motion to remove: a note, not a plot.
    assert isinstance(view.sky_pm_removed(), pn.pane.HTML)
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
    assert len(figs) == 2
    a, b = figs
    assert a.x_range is not b.x_range and a.y_range is not b.y_range
    spans = sorted(abs(f.x_range.end - f.x_range.start) for f in figs)
    # As observed the track spans ~2900 mas in RA; the parallax ellipse ~55 mas.
    assert spans[1] > 2500 and spans[0] < 200, spans


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
    assert len(time_figs) == 2 and len(sky_figs) == 2
    for f in time_figs:
        assert f.frame_width > 0 and f.frame_height > 0
        assert all(f.y_range is not g.x_range for g in sky_figs)


def test_panels_are_arranged_in_two_rows(payload, context):
    """Sky panels side by side on top, time panels side by side below; each
    pair wraps instead of overflowing a narrow window."""
    model = skyplane.SkyModel.from_reference(reference_fits()[HD114762])
    view = SkyPlaneView(context=context, payload=payload, model=model)
    boxes = [o for o in view.panel().objects if isinstance(o, pn.FlexBox)]
    assert boxes and len(boxes[0].objects) == 2, "sky panels are not side by side"
    assert boxes[0].flex_wrap == "wrap"
    comps = view.components()
    assert isinstance(comps, pn.FlexBox) and len(comps.objects) == 2
    assert comps.flex_wrap == "wrap"
    assert isinstance(view.layout()[0], pn.FlexBox), "controls should be a strip above"
