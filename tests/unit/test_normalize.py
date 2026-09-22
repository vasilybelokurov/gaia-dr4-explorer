import numpy as np
import pytest
from astropy.table import Table
from tests.conftest import synthetic_table

from gaia_dr4_explorer.products.astrometry.normalize import (
    CCD_NAMES,
    NormalizationError,
    fov_from_transit_id,
    normalize_epoch_astrometry,
)


def test_flattens_to_transits_times_ccds():
    raw = synthetic_table(n_transits=4)
    out = normalize_epoch_astrometry(raw)
    assert len(out.transits) == 4
    assert len(out.ccd) == 40
    assert list(out.ccd["ccd_name"][:10]) == list(CCD_NAMES)
    assert list(out.ccd["ccd_index"][:10]) == list(range(10))


def test_raw_table_is_not_modified():
    raw = synthetic_table(n_transits=3)
    before_cols = list(raw.colnames)
    before_len = len(raw)
    normalize_epoch_astrometry(raw)
    assert list(raw.colnames) == before_cols
    assert len(raw) == before_len


def test_rejects_more_than_one_source():
    raw = synthetic_table(source_ids=[1, 2, 3])
    with pytest.raises(NormalizationError, match="exactly one source"):
        normalize_epoch_astrometry(raw)


def test_rejects_ragged_arrays_rather_than_misaligning():
    raw = synthetic_table(n_transits=3, ragged=True)
    with pytest.raises(NormalizationError, match="inconsistent lengths"):
        normalize_epoch_astrometry(raw)


def test_rejects_empty_table():
    with pytest.raises(NormalizationError):
        normalize_epoch_astrometry(Table({"source_id": np.array([], dtype="int64")}))


def test_bitmask_columns_become_unsigned():
    raw = synthetic_table(n_transits=2)
    out = normalize_epoch_astrometry(raw)
    col = out.ccd["transit_proc_flags"]
    assert col.dtype == np.uint16
    # -32208 as int16 is 33328 as uint16: bit 15 set.
    assert int(np.asarray(col)[0]) == 33328
    assert int(np.asarray(col)[0]) & 0x8000 == 0x8000


def test_masked_values_stay_masked_and_are_never_zero():
    raw = synthetic_table(n_transits=3, mask_parallax_factor=True)
    out = normalize_epoch_astrometry(raw)
    col = out.ccd["parallax_factor_al"]
    assert hasattr(col, "mask")
    assert col.mask[:10].all(), "the masked transit must stay masked in all its CCD rows"
    assert not col.mask[10:].any()


def test_zero_length_column_is_kept_and_masked():
    raw = synthetic_table(n_transits=2, empty_ac=True)
    out = normalize_epoch_astrometry(raw)
    assert "centroid_pos_ac" in out.ccd.colnames, "an empty column must not vanish silently"
    assert np.all(out.ccd["centroid_pos_ac"].mask)
    assert any("zero-length" in w for w in out.warnings)


def test_units_survive():
    raw = synthetic_table(n_transits=2)
    out = normalize_epoch_astrometry(raw)
    assert str(out.ccd["centroid_pos_al"].unit) == "mas"
    assert str(out.ccd["g_mag"].unit) == "mag"
    assert str(out.ccd["obs_time_jd_tcb"].unit) == "d"


def test_original_timestamp_is_preserved_alongside_derived():
    raw = synthetic_table(n_transits=2)
    out = normalize_epoch_astrometry(raw)
    assert out.ccd["obs_time_tcb"].dtype == np.int64
    assert "obs_time_jd_tcb" in out.ccd.colnames
    assert "obs_time_jyear_tcb" in out.ccd.colnames


def test_constant_flag_columns_are_reported():
    raw = synthetic_table(n_transits=3)
    out = normalize_epoch_astrometry(raw)
    assert any("multipeak" in w for w in out.warnings)


def test_unknown_columns_do_not_break_parsing():
    raw = synthetic_table(n_transits=2)
    raw["brand_new_dr4_column"] = np.arange(len(raw))
    out = normalize_epoch_astrometry(raw)
    assert "brand_new_dr4_column" in out.ccd.colnames
    assert any("not in the field registry" in w for w in out.warnings)


def test_fov_decoding_handles_real_transit_ids():
    # gaiasupdate 0.1.2 raises OverflowError on these; ours must not.
    for tid in (248777679747269187, 161720884310443598, 292632412817923741):
        assert fov_from_transit_id(tid) in (0, 1, 2, 3)
