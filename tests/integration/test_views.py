"""The views build real plot objects from real data, without touching the network."""

import holoviews as hv
import numpy as np
import pytest
from tests.conftest import PRERELEASE_ZIP

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.products import registry
from gaia_dr4_explorer.products.astrometry import plots
from gaia_dr4_explorer.products.astrometry.exporters import export_table
from gaia_dr4_explorer.ui.astrometry import AstrometryView
from gaia_dr4_explorer.ui.state import AppState

pytestmark = pytest.mark.integration

BH3 = 4318465066420528000


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    cfg = AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"),
        local_prerelease_zip=PRERELEASE_ZIP,
        allow_network=False,
    )
    provider = PreReleaseProvider(cfg)
    registry.clear()
    registry.load_builtin_plugins()
    plugin = registry.get("epoch_astrometry").bind(provider)
    state = AppState(provider=provider, source_id=BH3, release=provider.release())
    payload = state.payload(plugin)
    return state, plugin, payload


def test_view_builds_every_plot(loaded):
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    assert len(view.frame) == 770
    for builder in (
        plots.coverage_timeline, plots.centroid_vs_time, plots.scan_angle_vs_time,
        plots.parallax_factor_vs_time, plots.uncertainty_distributions,
    ):
        obj = builder(view.filtered())
        assert isinstance(obj, hv.core.dimension.Dimensioned)


def test_axis_labels_always_carry_units():
    assert plots.axis_label("centroid_pos_al") == "centroid_pos_al [mas]"
    assert plots.axis_label("scan_pos_angle") == "scan_pos_angle [deg]"
    assert plots.axis_label("parallax_factor_al") == "parallax_factor_al"


def test_filters_narrow_the_selection(loaded):
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    view.show_rejected = False
    used_only = view.filtered()
    assert len(used_only) == 558
    assert used_only["used"].all()

    view.show_rejected = True
    view.ccds = ["SM"]
    assert set(view.filtered()["ccd_name"]) == {"SM"}

    view.ccds = list(view.param.ccds.objects)
    view.max_sigma = 0.1
    sigma = view.filtered()["centroid_pos_error_al"].to_numpy()
    assert np.all(sigma <= 0.1)


def test_constant_columns_are_detected_for_disabling(loaded):
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    assert "multipeak" in view._constant
    assert "blended" in view._constant


def test_focal_plane_matrix_shows_all_samples(loaded):
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    view.show_rejected = False
    heat = plots.focal_plane_matrix(view.frame, "used_by_agis_al")
    assert len(heat.data) == 770, "the matrix must not honour the used/rejected filter"


def test_residuals_align_with_the_used_selection(loaded):
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    result = state.fit()
    layout = plots.residual_plots(result, view.frame)
    assert isinstance(layout, hv.Layout)
    assert result.residuals.size == int(view.frame["used"].sum())


@pytest.mark.parametrize("fmt", ["parquet", "ecsv", "csv"])
def test_exports_round_trip(loaded, fmt):
    _, plugin, payload = loaded
    blob = plugin.export(payload, fmt)
    assert isinstance(blob, bytes) and len(blob) > 1000
    if fmt == "csv":
        assert b"does not preserve Astropy units" in blob[:200]


def test_ecsv_export_preserves_units(loaded):
    _, _, payload = loaded
    blob = export_table(payload.table("ccd"), "ecsv")
    assert b"unit: mas" in blob


def test_provenance_export_records_the_release(loaded):
    _, plugin, payload = loaded
    blob = plugin.export(payload, "provenance")
    assert b"Gaia DR4_RC3" in blob
    assert b"gaiasupdate" not in blob or b"epoch-astrometry-normalized" in blob


def test_gaia_identifiers_survive_as_strings_for_display(loaded):
    """Bokeh's float64 columns cannot hold a 19-digit identifier."""
    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    ids = view.frame["transit_id"].to_numpy()
    strings = view.frame["transit_id_str"].to_numpy()
    assert all(str(int(i)) == s for i, s in zip(ids, strings, strict=True))
    assert len(strings[0]) >= 18
    # the float round-trip Bokeh would perform is genuinely lossy here
    assert int(float(ids[0])) != int(ids[0]) or ids[0] < 2**53


@pytest.mark.parametrize("quantity", [
    "used_by_agis_al", "present", "centroid_pos_error_al", "ipd_error_al",
    "ccd_proc_flags", "gates",
])
def test_every_matrix_quantity_renders(loaded, quantity):
    import io

    import panel as pn

    state, _, payload = loaded
    view = AstrometryView(context=state.context(), payload=payload)
    heat = plots.focal_plane_matrix(view.frame, quantity)
    buf = io.StringIO()
    pn.Column(pn.pane.HoloViews(heat)).save(buf)
    assert len(buf.getvalue()) > 1000
