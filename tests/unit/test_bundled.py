"""The bundled DR3 products: what ships, and that plugins can use it offline."""

import pytest

from gaia_dr4_explorer.data.archive import (
    EPOCH_PHOTOMETRY,
    XP_SAMPLED,
    ArchiveError,
    ArchiveUnavailable,
)
from gaia_dr4_explorer.data.bundled import BundledProductProvider
from gaia_dr4_explorer.data.dr3_availability import dr3_availability
from gaia_dr4_explorer.domain import ProductState, SourceContext, SourceKey
from gaia_dr4_explorer.products.context.plugin import ContextPlugin
from gaia_dr4_explorer.products.photometry.plugin import EpochPhotometryPlugin
from gaia_dr4_explorer.products.spectra.plugin import XpSpectrumPlugin


@pytest.fixture(scope="module")
def provider():
    return BundledProductProvider()


def context(sid):
    return SourceContext(key=SourceKey("Gaia DR4_RC3", sid))


def test_bundle_matches_the_availability_table(provider):
    """Everything DR3 says exists must actually be bundled, or the page lies."""
    missing = []
    for sid, flags in dr3_availability().items():
        if flags.has_epoch_photometry and not provider.has(sid, EPOCH_PHOTOMETRY):
            missing.append(f"{sid} photometry")
        if flags.has_xp_sampled and not provider.has(sid, XP_SAMPLED):
            missing.append(f"{sid} xp")
    assert not missing, f"declared available but not bundled: {missing}"


def test_bundle_carries_no_product_that_is_not_declared(provider):
    for sid, flags in dr3_availability().items():
        if provider.has(sid, EPOCH_PHOTOMETRY):
            assert flags.has_epoch_photometry, sid
        if provider.has(sid, XP_SAMPLED):
            assert flags.has_xp_sampled, sid


def test_unbundled_source_is_an_absence_not_an_error(provider):
    with pytest.raises(ArchiveUnavailable):
        provider.get_datalink_product(4318465066420528000, EPOCH_PHOTOMETRY)


def test_unknown_retrieval_type_is_refused(provider):
    with pytest.raises(ArchiveError, match="not bundled"):
        provider.get_datalink_product(4318465066420528000, "RVS")


@pytest.mark.parametrize("sid", [1457486023639239296, 2237987199365376, 10973744521070720])
def test_photometry_loads_offline(provider, sid):
    payload = EpochPhotometryPlugin().bind(provider).load(context(sid))
    assert len(payload.table("epochs")) > 0
    assert payload.summary["release"] == "Gaia DR3"


def test_spectrum_loads_offline(provider):
    payload = XpSpectrumPlugin().bind(provider).load(context(4318465066420528000))
    s = payload.summary
    assert s["n_samples"] == 343
    assert 330 < s["wavelength_min_nm"] < 345
    assert 1015 < s["wavelength_max_nm"] < 1025


def test_spectrum_discovery_matches_the_bundle(provider):
    plugin = XpSpectrumPlugin().bind(provider)
    for sid, flags in dr3_availability().items():
        state = plugin.discover(context(sid)).state
        expected = ProductState.AVAILABLE if flags.has_xp_sampled else ProductState.UNAVAILABLE
        assert state is expected, sid


def test_context_knows_six_of_twelve(provider):
    plugin = ContextPlugin().bind(provider)
    available = [
        sid for sid in dr3_availability()
        if plugin.discover(context(sid)).state is ProductState.AVAILABLE
    ]
    assert len(available) == 6


def test_context_payload_carries_identity_and_bibliography(provider):
    payload = ContextPlugin().bind(provider).load(context(3937211745905473024))
    assert payload.summary["main_id"] == "HD 114762"
    assert payload.summary["nbref"] > 500
    assert payload.summary["ads_num_found"] > 0
    assert payload.summary["n_cross_ids"] > 5


def test_no_credential_is_bundled(provider):
    """The ADS token fetched these records; it must never have been written."""
    import json

    blob = json.dumps([provider.simbad(), provider.ads()])
    for needle in ("ADS_API_TOKEN", "Bearer ", "Authorization"):
        assert needle not in blob
