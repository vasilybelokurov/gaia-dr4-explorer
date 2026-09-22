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
from gaia_dr4_explorer.ui.components import MAIN_TAB, show_main_view
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


def test_main_view_button_returns_to_the_astrometry_tab():
    tabs = pn.Tabs(("Overview", "a"), ("Astrometry", "b"), ("Sky plane", "c"), active=2)
    show_main_view(pn.Column(tabs))
    assert tabs.active == MAIN_TAB == 1
