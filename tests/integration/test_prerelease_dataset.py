"""Integration tests against the real prerelease archive (committed fixture)."""

import numpy as np
import pytest

from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.products.astrometry import normalize_epoch_astrometry

pytestmark = pytest.mark.integration

EXPECTED_SOURCE_IDS = [
    2237987199365376, 2309425390592896, 10973744521070720, 20694084440761600,
    60730287810150016, 435469040545191680, 1457486023639239296, 1663617687609809280,
    3926186255616949504, 3937211745905473024, 4181040337841125632, 4318465066420528000,
]


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    from tests.conftest import PRERELEASE_ZIP

    from gaia_dr4_explorer.config import AppConfig

    cfg = AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"),
        local_prerelease_zip=PRERELEASE_ZIP,
        allow_network=False,
    )
    return PreReleaseProvider(cfg)


def test_t1_structure(provider):
    table = provider.load_table()
    assert len(table) == 1008
    assert len(table.colnames) == 37
    assert provider.source_ids() == EXPECTED_SOURCE_IDS
    assert np.unique(np.asarray(table["solution_id"])).tolist() == [2888461026632663040]


def test_release_string_survives_parsing(provider):
    assert provider.release() == "Gaia DR4_RC3"


def test_every_source_normalizes(provider):
    total_slots = 0
    total_used = 0
    total_finite = 0
    for sid in provider.source_ids():
        raw = provider.raw_table_for(sid)
        out = normalize_epoch_astrometry(raw, provenance=provider.provenance)
        s = out.summary()
        assert s["n_ccd_slots"] == s["n_transits"] * 10
        assert out.source_id == sid
        assert np.unique(np.asarray(out.ccd["source_id"])).tolist() == [sid]
        total_slots += s["n_ccd_slots"]
        total_used += s["n_used_by_agis_al"]
        total_finite += s["n_ccd_finite"]
    assert total_slots == 10080
    assert total_used == 7467
    assert total_finite == 8941


def test_nan_centroids_are_never_used_by_agis(provider):
    for sid in provider.source_ids():
        out = normalize_epoch_astrometry(provider.raw_table_for(sid))
        al = np.asarray(np.ma.filled(out.ccd["centroid_pos_al"], np.nan), dtype="float64")
        used = np.asarray(np.ma.filled(out.ccd["used_by_agis_al"], False), dtype=bool)
        assert int((~np.isfinite(al) & used).sum()) == 0


def test_ac_columns_are_empty_throughout(provider):
    out = normalize_epoch_astrometry(provider.raw_table_for(4318465066420528000))
    for name in ("centroid_pos_ac", "centroid_pos_error_ac"):
        if name in out.ccd.colnames:
            assert np.all(out.ccd[name].mask)


def test_time_span_matches_the_measured_value(provider):
    table = provider.load_table()
    ns = np.concatenate([np.atleast_1d(v) for v in table["obs_time_tcb"]])
    from gaia_dr4_explorer.products.astrometry.times import tcb_ns_to_time

    t = tcb_ns_to_time(ns.astype("int64"))
    assert t.min().isot.startswith("2014-07-30")
    assert t.max().isot.startswith("2020-01-15")
