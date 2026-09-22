"""The band mapping must match the columns the archive actually serves."""

from pathlib import Path

import numpy as np
import pytest
from astropy.io.votable import parse_single_table

from gaia_dr4_explorer.products.photometry import plots
from gaia_dr4_explorer.products.photometry.normalize import (
    BANDS,
    TRANSIT_REJECTED_COLUMN,
    normalize_epoch_photometry,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "dr3_epoch_photometry_gaia4.vot"

#: Exactly what DR3 EPOCH_PHOTOMETRY serves, verified 2026-09-22.
SERVED = {
    "transit_id",
    "g_transit_time", "g_transit_flux", "g_transit_flux_error",
    "g_transit_flux_over_error", "g_transit_mag",
    "bp_obs_time", "bp_flux", "bp_flux_error", "bp_flux_over_error", "bp_mag",
    "rp_obs_time", "rp_flux", "rp_flux_error", "rp_flux_over_error", "rp_mag",
    "variability_flag_g_reject", "variability_flag_bp_reject", "variability_flag_rp_reject",
    "g_other_flags", "bp_other_flags", "rp_other_flags", "rejected_by_photometry",
}


@pytest.fixture(scope="module")
def raw():
    return parse_single_table(FIXTURE).to_table(use_names_over_ids=True)


def test_fixture_matches_the_documented_column_set(raw):
    assert set(raw.colnames) == SERVED


def test_every_mapped_column_is_actually_served():
    """Guards the class of bug where a band silently falls back to a default."""
    mapped = {name for cols in BANDS.values() for name in cols.values()}
    missing = sorted(mapped - SERVED)
    assert not missing, f"BANDS references columns the archive does not serve: {missing}"
    assert TRANSIT_REJECTED_COLUMN in SERVED


def test_flux_error_is_not_confused_with_signal_to_noise():
    for band, cols in BANDS.items():
        assert cols["flux_error"] != cols["flux_over_error"], band
        assert cols["flux_error"].endswith("_flux_error"), band


def test_each_band_uses_its_own_rejection_flag():
    flags = {band: cols["rejected"] for band, cols in BANDS.items()}
    assert len(set(flags.values())) == 3, f"bands share a rejection flag: {flags}"
    for band, flag in flags.items():
        assert band.lower() in flag.lower()


def test_normalization_produces_three_bands(raw):
    out = normalize_epoch_photometry(raw)
    assert set(out.bands_present) == {"G", "BP", "RP"}
    assert len(out.epochs) > 100
    assert out.summary()["n_G"] == 65


def test_rejection_flags_are_real_not_assumed(raw):
    out = normalize_epoch_photometry(raw)
    rejected = out.epochs["rejected"]
    assert getattr(rejected, "mask", None) is None or not np.all(rejected.mask), (
        "per-band rejection must come from the served flags, not a default"
    )
    assert not any("assumed accepted" in w for w in out.warnings)


def test_unknown_quality_is_not_reported_as_accepted(raw):
    """Drop a band's flag and the epochs must not silently become 'accepted'."""
    trimmed = raw.copy()
    trimmed.remove_column("variability_flag_rp_reject")
    out = normalize_epoch_photometry(trimmed)
    frame = plots.to_frame(out.epochs)
    rp = frame[frame["band"] == "RP"]
    assert set(rp["status"]) == {"quality unknown"}
    assert any("rejection is unknown" in w for w in out.warnings)


def test_original_time_column_is_kept_beside_the_derived_one(raw):
    out = normalize_epoch_photometry(raw)
    assert "time_bjd_offset" in out.epochs.colnames
    assert "time_jd_tcb" in out.epochs.colnames
    offset = np.asarray(np.ma.filled(out.epochs["time_bjd_offset"], np.nan), dtype="float64")
    jd = np.asarray(np.ma.filled(out.epochs["time_jd_tcb"], np.nan), dtype="float64")
    np.testing.assert_allclose(jd - offset, 2455197.5, rtol=0, atol=1e-6)


def test_times_land_in_the_gaia_mission_window(raw):
    out = normalize_epoch_photometry(raw)
    jd = np.asarray(np.ma.filled(out.epochs["time_jd_tcb"], np.nan), dtype="float64")
    jd = jd[np.isfinite(jd)]
    # 2014-07-25 to 2017-05-28, the DR3 photometry window.
    assert 2456863.0 < jd.min() < 2458000.0
    assert jd.max() < 2458000.0
