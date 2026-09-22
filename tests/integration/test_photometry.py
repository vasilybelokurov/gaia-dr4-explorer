"""Epoch photometry: availability logic offline, retrieval behind the network marker."""

import numpy as np
import pytest

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data.dr3_availability import dr3_availability, for_source
from gaia_dr4_explorer.domain import ProductState, SourceContext, SourceKey
from gaia_dr4_explorer.products.photometry.plugin import EpochPhotometryPlugin

GAIA_4 = 1457486023639239296
QSO_VARIABLE = 2237987199365376
BH3 = 4318465066420528000

WITH_PHOTOMETRY = {GAIA_4, QSO_VARIABLE, 10973744521070720}


def context(source_id: int) -> SourceContext:
    return SourceContext(key=SourceKey("Gaia DR4_RC3", source_id))


def test_availability_table_covers_all_twelve():
    assert len(dr3_availability()) == 12
    assert {s for s, f in dr3_availability().items() if f.has_epoch_photometry} == WITH_PHOTOMETRY
    assert sum(f.has_rvs for f in dr3_availability().values()) == 0


def test_discovery_makes_no_network_call(tmp_path):
    """Offline config: discovery must still answer for every source."""
    plugin = EpochPhotometryPlugin()
    for sid in dr3_availability():
        descriptor = plugin.discover(context(sid))
        expected = (
            ProductState.AVAILABLE if sid in WITH_PHOTOMETRY else ProductState.UNAVAILABLE
        )
        assert descriptor.state is expected, sid
        assert descriptor.release == "Gaia DR3"
        assert descriptor.detail


def test_unavailable_message_explains_and_does_not_blame_the_user():
    plugin = EpochPhotometryPlugin()
    detail = plugin.discover(context(BH3)).detail
    assert "has_epoch_photometry = false" in detail
    assert "2026-12-02" in detail, "must say when DR4 will provide it"


def test_bp_rp_colour_from_shipped_means():
    flags = for_source(BH3)
    assert flags is not None
    assert flags.bp_rp == pytest.approx(11.753893 - 10.538245, abs=1e-6)


@pytest.mark.network
def test_real_retrieval_and_normalization(tmp_path):
    from gaia_dr4_explorer.data.archive import GaiaArchiveProvider

    cfg = AppConfig(cache_dir=tmp_path / "cache", allow_network=True)
    plugin = EpochPhotometryPlugin().bind(GaiaArchiveProvider(cfg))
    payload = plugin.load(context(GAIA_4))

    epochs = payload.table("epochs")
    assert len(epochs) > 0
    bands = set(np.asarray(epochs["band"]).tolist())
    assert bands <= {"G", "BP", "RP"} and "G" in bands
    assert payload.summary["release"] == "Gaia DR3"
    # the derived time column must sit beside the original, not replace it
    assert "time_bjd_offset" in epochs.colnames
    assert "time_jd_tcb" in epochs.colnames
    assert float(np.nanmin(epochs["time_jd_tcb"])) > 2456000.0


@pytest.mark.network
def test_second_load_is_served_from_cache(tmp_path):
    from gaia_dr4_explorer.data.archive import EPOCH_PHOTOMETRY, GaiaArchiveProvider

    cfg = AppConfig(cache_dir=tmp_path / "cache", allow_network=True)
    archive = GaiaArchiveProvider(cfg)
    _, first = archive.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY)
    assert "cache" not in first.steps[0]
    _, second = archive.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY)
    assert "cache" in second.steps[0]
