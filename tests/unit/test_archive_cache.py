"""Cache integrity, offline. A cache file is a path on disk and nothing more."""

import shutil
from pathlib import Path

import pytest

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data.archive import (
    EPOCH_PHOTOMETRY,
    ArchiveError,
    GaiaArchiveProvider,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dr3_epoch_photometry_gaia4.vot"
GAIA_4 = 1457486023639239296
OTHER = 4318465066420528000


@pytest.fixture
def offline(tmp_path):
    return AppConfig(cache_dir=tmp_path / "cache", allow_network=False)


def _seed(provider, source_id, src=FIXTURE):
    path = provider.cache_path(source_id, EPOCH_PHOTOMETRY, "Gaia DR3")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, path)
    return path


def test_cached_product_is_served_without_network(offline):
    provider = GaiaArchiveProvider(offline)
    _seed(provider, GAIA_4)
    table, record = provider.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY)
    assert len(table) > 0
    assert "cache" in record.steps[0]


def test_a_cache_file_holding_the_wrong_source_is_refused(offline):
    """The defect this guards: Gaia-4's curve displayed as Gaia BH3's."""
    provider = GaiaArchiveProvider(offline)
    _seed(provider, OTHER)          # Gaia-4 data filed under BH3's identifier
    with pytest.raises(ArchiveError, match="refusing to use it"):
        provider.get_datalink_product(OTHER, EPOCH_PHOTOMETRY)


def test_a_corrupt_cache_file_is_refused_not_displayed(offline):
    provider = GaiaArchiveProvider(offline)
    path = _seed(provider, GAIA_4)
    path.write_bytes(b"<VOTABLE>truncated")
    with pytest.raises(ArchiveError):
        provider.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY)


def test_unknown_retrieval_type_never_reaches_a_path(offline):
    provider = GaiaArchiveProvider(offline)
    for bad in ("../../etc/passwd", "ALL", "", "epoch_photometry"):
        with pytest.raises(ArchiveError, match="unknown retrieval type"):
            provider.get_datalink_product(GAIA_4, bad)


def test_cache_path_includes_the_release(offline):
    provider = GaiaArchiveProvider(offline)
    dr3 = provider.cache_path(GAIA_4, EPOCH_PHOTOMETRY, "Gaia DR3")
    dr4 = provider.cache_path(GAIA_4, EPOCH_PHOTOMETRY, "Gaia DR4")
    assert dr3 != dr4


def test_missing_product_offline_is_an_error_not_an_empty_plot(offline):
    provider = GaiaArchiveProvider(offline)
    with pytest.raises(ArchiveError, match="network access is disabled"):
        provider.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY)


def test_a_cache_file_from_the_wrong_release_is_refused(offline):
    provider = GaiaArchiveProvider(offline)
    path = provider.cache_path(GAIA_4, EPOCH_PHOTOMETRY, "Gaia DR4")
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURE, path)   # a DR3 response filed under DR4
    with pytest.raises(ArchiveError, match="declares release"):
        provider.get_datalink_product(GAIA_4, EPOCH_PHOTOMETRY, release="Gaia DR4")


def test_identity_is_checked_from_votable_params_not_columns(offline):
    """EPOCH_PHOTOMETRY has no source_id column; the PARAM is the only identity."""
    from astropy.io.votable import parse_single_table

    vt = parse_single_table(FIXTURE)
    names = {p.name for p in vt.params}
    assert "source_id" in names and "release" in names
    assert "source_id" not in vt.to_table(use_names_over_ids=True).colnames
