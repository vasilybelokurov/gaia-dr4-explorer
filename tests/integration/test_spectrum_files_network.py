"""Download and read one real spectrum from every archive (pytest -m network).

HD 114762 has spectra in each; the expected wavelength ranges are those the
archives list (checked 2026-09-23).
"""

import pytest

from gaia_dr4_explorer.data import external_spectra as xs
from gaia_dr4_explorer.data.spectrum_files import SpectrumFileStore

pytestmark = pytest.mark.network

HD114762 = xs.SkyPosition(198.0792922757309, 17.517123799636206, 2017.5,
                          pmra_masyr=-582.03, pmdec_masyr=-0.71, name="HD 114762")

#: (archive, predicate picking a record, expected wavelength range in nm)
CASES = [
    ("MAST IUE", lambda r: r.archive == "MAST" and r.collection == "IUE", (185, 335)),
    ("MAST APOGEE via SDSS", lambda r: r.archive == "MAST" and "apogee" in r.instrument, (1510, 1700)),
    ("MAST STIS x1d", lambda r: r.archive == "MAST" and r.instrument.startswith("STIS"), (227, 312)),
    ("MAST GHRS c0f+c1f", lambda r: r.archive == "MAST" and r.obs_id == "z2xm0603t", (287, 293)),
    ("ESO UVES", lambda r: r.archive == "ESO", (300, 1050)),
    ("CADC ESPaDOnS Upena", lambda r: r.archive == "CADC" and r.obs_id == "1701499i", (369, 1049)),
    ("CfA echelle", lambda r: r.archive == "CfA TDC", (516, 522)),
    ("PolarBase", lambda r: r.archive == "PolarBase", (369, 1049)),
    ("ELODIE", lambda r: r.archive == "ELODIE", (399, 681)),
]


@pytest.fixture(scope="module")
def records():
    return xs.search_external_spectra(HD114762, timeout_s=180).records()


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return SpectrumFileStore(tmp_path_factory.mktemp("spectra"), xs.LiveTransport(timeout_s=120))


@pytest.mark.parametrize("label,pick,expected", CASES, ids=[c[0] for c in CASES])
def test_spectrum_downloads_and_reads(records, store, label, pick, expected):
    rec = next(r for r in records if pick(r))
    loaded = store.load(rec)
    lo, hi = loaded.spectrum.range_nm
    assert expected[0] <= lo < hi <= expected[1], f"{label}: {lo:.1f}-{hi:.1f} nm"
    assert loaded.spectrum.n_pixels >= 500


def test_raw_frames_are_refused_with_a_reason(records, store):
    rec = next(r for r in records if r.archive == "CADC" and r.collection == "DAO"
               and "Cassegrain" in r.instrument)
    with pytest.raises(ValueError, match="pixels, not wavelength"):
        store.load(rec)
