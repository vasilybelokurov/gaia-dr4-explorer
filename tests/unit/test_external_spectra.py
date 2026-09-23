"""External spectral-archive discovery, tested offline.

The fake rows mirror what each service returned in a live query for
HD 114762 on 2026-09-23 (column names, units and value formats), so the
parsers are tested against the real shapes without touching the network.
"""

import math
import threading
from datetime import UTC, datetime

import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from gaia_dr4_explorer.data import external_spectra as xs

HD114762 = xs.SkyPosition(198.0792922757309, 17.517123799636206, 2017.5,
                          pmra_masyr=-582.03, pmdec_masyr=-0.71)


# ------------------------------------------------------------ fake transport


class FakeTransport(xs.Transport):
    def __init__(self, *, tap=None, ssa=None, mast=None, text=None, fail=(), hang=()):
        self._tap, self._ssa, self._mast, self._text = tap or {}, ssa or {}, mast, text
        self.fail, self.hang = set(fail), set(hang)
        self.calls = []
        self.release = threading.Event()

    def _gate(self, key):
        self.calls.append(key)
        if key in self.hang:
            self.release.wait(5)
        if key in self.fail:
            raise ConnectionError(f"{key} is down")

    def tap(self, url, adql):
        self._gate(url)
        self.last_adql = adql
        return self._tap[url]

    def ssa_url(self, ivoid):
        self._gate(ivoid)
        return ivoid

    def ssa(self, url, ra, dec, diameter_deg):
        return self._ssa.get(url, ([], []))

    def mast(self, request):
        self._gate("mast")
        self.last_mast = request
        return self._mast or {"status": "COMPLETE", "data": []}

    def get_text(self, url, params):
        self._gate(url)
        return self._text or ""


def eso_table():
    t = Table()
    t["obs_collection"] = ["UVES", "UVES"]
    t["instrument_name"] = ["UVES", "UVES"]
    t["obs_id"] = ["a", "b"]
    t["target_name"] = ["hip64426", "hip-64426"]
    t["s_ra"] = [198.08104, 198.081318]
    t["s_dec"] = [17.51732, 17.5172]
    t["t_min"] = MaskedColumn([53894.05684172, 53894.03871881], unit="d")
    t["t_exptime"] = [600.0, 600.0]
    t["em_min"] = MaskedColumn([3.02424e-07, 4.72684e-07], unit="m")
    t["em_max"] = MaskedColumn([3.88428e-07, 6.83504e-07], unit="m")
    t["em_res_power"] = [36840.0, 51690.0]
    t["access_url"] = ["https://archive.eso.org/datalink/links?ID=x"] * 2
    t["access_format"] = ["application/x-votable+xml;content=datalink"] * 2
    t["access_estsize"] = MaskedColumn([2033, 9512], unit="kbyte")
    return t


# ---------------------------------------------------------------- geometry


def test_search_radius_grows_with_proper_motion():
    # 582 mas/yr over the longest stretch of 1975-2027 from J2017.5 (42.5 yr).
    r = HD114762.search_radius_arcsec(20.0)
    assert r == pytest.approx(20.0 + math.hypot(582.03, 0.71) * 42.5 / 1000, rel=1e-9)
    unknown = xs.SkyPosition(10.0, 20.0, 2017.5)
    assert unknown.search_radius_arcsec(20.0) == 20.0, "unknown PM must not be taken as zero"


def test_propagation_moves_ra_by_pm_over_cos_dec():
    ra, dec = HD114762.at(2017.5 - 28.0)       # back to ~1989.5
    dra_arcsec = (ra - HD114762.ra_deg) * 3600 * math.cos(math.radians(HD114762.dec_deg))
    assert dra_arcsec == pytest.approx(582.03e-3 * 28.0, rel=1e-6)   # RA was larger then
    assert (dec - HD114762.dec_deg) * 3600 == pytest.approx(0.71e-3 * 28.0, rel=1e-6)


def test_separation_is_the_closest_approach_to_the_path():
    """Right whether the archive stored the position on the night or a J2000
    catalogue position -- both lie on the star's path."""
    for year in (1989.5, 2000.0, 2010.0):
        ra, dec = HD114762.at(year)
        assert HD114762.separation_arcsec(ra, dec) == pytest.approx(0.0, abs=1e-3)
    # 3 arcsec north of the 2000 position: 3 arcsec off the (east-west) path.
    ra, dec = HD114762.at(2000.0)
    assert HD114762.separation_arcsec(ra, dec + 3 / 3600) == pytest.approx(3.0, abs=0.02)


def test_closest_approach_is_limited_to_the_archive_span():
    # A point on the path extended to 1900 is not a match: the clamp to 1975
    # leaves it (1975-1900) x 0.582 arcsec away.
    ra, dec = HD114762.at(1900.0)
    assert HD114762.separation_arcsec(ra, dec) == pytest.approx(75 * 0.58203, rel=2e-3)


def test_separation_without_proper_motion_is_the_plain_distance():
    p = xs.SkyPosition(10.0, 20.0, 2017.5)
    assert p.separation_arcsec(10.0, 20.0 + 5 / 3600) == pytest.approx(5.0, abs=1e-6)
    assert math.isnan(p.separation_arcsec(float("nan"), 20.0))


# ----------------------------------------------------------------- parsers


def test_obscore_uses_declared_units():
    recs = xs.parse_obscore(eso_table(), archive="ESO")
    assert recs[0].wl_min_nm == pytest.approx(302.424)
    assert recs[1].wl_max_nm == pytest.approx(683.504)
    assert recs[0].size_bytes == 2033000            # kbyte -> byte
    assert recs[0].mjd == pytest.approx(53894.05684172)
    assert recs[0].resolving_power == 36840.0


def test_obscore_falls_back_to_standard_units_when_undeclared():
    t = eso_table()
    t["em_min"].unit = None
    t["em_max"].unit = None
    recs = xs.parse_obscore(t, archive="X")
    assert recs[0].wl_min_nm == pytest.approx(302.424)


def test_resolving_power_given_as_a_resolution_element_is_converted():
    # CfA TDC reports 1.29e-11 m in em_res_power for its 516-521 nm order.
    r = xs.resolving_power(1.2914425e-11, 516.577, 521.123)
    assert r == pytest.approx(0.5 * (516.577 + 521.123) / 0.012914425, rel=1e-9)
    assert math.isnan(xs.resolving_power(float("nan"), 500, 600))
    assert xs.resolving_power(48000.0, 500, 600) == 48000.0


def test_masked_values_stay_missing_not_zero():
    t = eso_table()
    t["em_min"] = MaskedColumn([3e-7, 4e-7], mask=[True, False], unit="m")
    t["access_estsize"] = MaskedColumn([1, 2], mask=[True, False], unit="kbyte")
    recs = xs.parse_obscore(t, archive="ESO")
    assert math.isnan(recs[0].wl_min_nm)
    assert recs[0].size_bytes is None


def test_mast_rows_use_nm_and_become_download_urls():
    resp = {"status": "COMPLETE", "data": [
        {"obs_collection": "IUE", "instrument_name": "LWP", "obs_id": "lwp14968",
         "t_min": 47562.18185, "em_min": 185.118, "em_max": 334.76, "s_ra": 198.0904661,
         "s_dec": 17.5167985, "dataURL": "http://archive.stsci.edu/pub/vospectra/iue2/x.fits",
         "dataRights": "PUBLIC"},
        {"obs_collection": "HST", "instrument_name": "HRS/2", "obs_id": "z2xm0603t",
         "t_min": 50043.08, "em_min": 200, "em_max": 330, "s_ra": 198.0829, "s_dec": 17.5171,
         "dataURL": "mast:HST/product/z2xm0603t_c0f.fits", "dataRights": "PUBLIC"},
        {"obs_collection": "HST", "obs_id": "p", "dataRights": "EXCLUSIVE_ACCESS"},
        {"obs_collection": "HST", "obs_id": "z2xm0601t", "t_min": None, "em_min": None,
         "dataURL": None, "dataRights": "PUBLIC"},
    ]}
    recs = xs.parse_mast(resp)
    assert [r.obs_id for r in recs] == ["lwp14968", "z2xm0603t", "z2xm0601t"]
    assert recs[0].wl_min_nm == pytest.approx(185.118)
    assert recs[1].access_url == xs.MAST_DOWNLOAD + "mast:HST/product/z2xm0603t_c0f.fits"
    assert math.isnan(recs[2].mjd) and recs[2].access_url == ""


def test_caom2_drops_planes_not_yet_public_and_links_datalink():
    t = Table()
    t["collection"] = ["CFHT", "CFHT"]
    t["instrument_name"] = ["ESPaDOnS", "ESPaDOnS"]
    t["observationID"] = ["1701499", "9"]
    t["productID"] = ["1701499i", "9i"]
    t["target_name"] = ["HD 114762", "x"]
    t["ra"] = MaskedColumn([198.08375, 0.0], mask=[False, True])
    t["dec"] = MaskedColumn([17.5171389, 0.0], mask=[False, True])
    t["time_bounds_lower"] = [56763.45, 60000.0]
    t["time_exposure"] = [600.0, 1.0]
    t["energy_bounds_lower"] = MaskedColumn([3.69106e-07, 4e-7], unit="m")
    t["energy_bounds_upper"] = MaskedColumn([1.0481439e-06, 5e-7], unit="m")
    t["energy_resolvingPower"] = [65000.0, 1.0]
    t["dataRelease"] = ["2014-10-31T00:00:00.000", "2099-01-01T00:00:00.000"]
    t["publisherID"] = ["ivo://cadc.nrc.ca/CFHT?1701499/1701499i", "ivo://x"]
    recs = xs.parse_caom2(t, now=datetime(2026, 9, 23, tzinfo=UTC))
    assert len(recs) == 1
    r = recs[0]
    assert r.wl_min_nm == pytest.approx(369.106) and r.wl_max_nm == pytest.approx(1048.1439)
    assert r.access_url == xs.CADC_DATALINK + "ivo://cadc.nrc.ca/CFHT?1701499/1701499i"


POLARBASE_FIELDS = [
    {"name": "Title", "utype": "ssa:DataID.Title", "ucd": "meta.title;meta.dataset", "unit": ""},
    {"name": "SpatialLocation", "utype": "ssa:Char.SpatialAxis.Coverage.Location.Value",
     "ucd": "pos.eq", "unit": "deg"},
    {"name": "TimeLocation", "utype": "ssa:Char.TimeAxis.Coverage.Location.Value",
     "ucd": "time.epoch", "unit": "MJD"},
    {"name": "SpectralLocation", "utype": "ssa:Char.SpectralAxis.Coverage.Location.Value",
     "ucd": "instr.bandpass", "unit": "nm"},
    {"name": "SpectralExtent", "utype": "ssa:Char.SpectralAxis.Coverage.Bounds.Extent",
     "ucd": "instr.bandwidth", "unit": "nm"},
    {"name": "format", "utype": "ssa:access.format", "ucd": "meta.code.mime", "unit": ""},
    {"name": "url", "utype": "ssa:access.reference", "ucd": "meta.ref.url", "unit": ""},
]


def test_ssa_centre_plus_extent_in_nm_polarbase_style():
    rows = [{"Title": "narval_2010_06mar10_norm_hd114",
             "SpatialLocation": np.array([198.08225445, 17.51711954]),
             "TimeLocation": 55262.16376157408, "SpectralLocation": 709.0,
             "SpectralExtent": 678.0, "format": "application/fits",
             "url": "https://www.polarbase.ovgso.fr/download/file/1701498i.fts"}]
    (r,) = xs.parse_ssa(POLARBASE_FIELDS, rows, archive="PolarBase")
    assert (r.wl_min_nm, r.wl_max_nm) == pytest.approx((370.0, 1048.0))
    assert r.mjd == pytest.approx(55262.16376157408)
    assert (r.ra_deg, r.dec_deg) == pytest.approx((198.08225445, 17.51711954))
    assert r.access_url.endswith("1701498i.fts")


def test_ssa_start_stop_in_metres_and_resolution_to_r():
    fields = [
        {"name": "start", "utype": "ssa:Char.SpectralAxis.Coverage.Bounds.Start", "ucd": "", "unit": "m"},
        {"name": "stop", "utype": "ssa:Char.SpectralAxis.Coverage.Bounds.Stop", "ucd": "", "unit": "m"},
        {"name": "fwhm", "utype": "ssa:Char.SpectralAxis.Resolution", "ucd": "", "unit": "m"},
        {"name": "date", "utype": "ssa:Char.TimeAxis.Coverage.Location.Value", "ucd": "", "unit": ""},
    ]
    rows = [{"start": 4e-7, "stop": 6e-7, "fwhm": 1e-11, "date": "2010-03-06T03:55:49"}]
    (r,) = xs.parse_ssa(fields, rows, archive="X")
    assert (r.wl_min_nm, r.wl_max_nm) == pytest.approx((400.0, 600.0))
    assert r.resolving_power == pytest.approx(500.0 / 0.01)
    assert r.mjd == pytest.approx(55261.16376, abs=1e-4)


def test_elodie_listing():
    text = "# header\n$ junk\n" + "\t".join(["a", "b", "c", "d", "19990310", "0023", "x", "1800", "95"])
    (r,) = xs.parse_elodie(text)
    assert r.obs_id == "19990310/0023" and r.snr == 95 and r.exptime_s == 1800
    assert "elodie:19990310/0023" in r.access_url


# ------------------------------------------------------------------ runner


def test_every_archive_reports_a_status_and_one_failure_does_not_stop_the_rest():
    fake = FakeTransport(tap={xs.ESO_TAP: eso_table(), xs.CFA_TAP: Table(eso_table()[:0])},
                         fail={xs.CADC_TAP})
    report = xs.search_external_spectra(HD114762, xs.default_searches(fake), timeout_s=10)
    by = {r.archive: r for r in report.results}
    assert set(by) == {"MAST", "ESO", "CADC", "CfA TDC", "FEROS", "PolarBase", "BeSS", "ELODIE"}
    assert by["ESO"].status is xs.SearchStatus.FOUND and by["ESO"].n_spectra == 2
    assert by["CfA TDC"].status is xs.SearchStatus.NONE
    assert by["MAST"].status is xs.SearchStatus.NONE
    assert by["CADC"].status is xs.SearchStatus.FAILED and "is down" in by["CADC"].error
    assert by["ELODIE"].status is xs.SearchStatus.SKIPPED   # no name given
    assert not report.complete(), "a failed archive means the search is incomplete"


def test_a_hung_archive_times_out_without_blocking_the_report():
    fake = FakeTransport(tap={xs.ESO_TAP: eso_table()}, hang={"mast"})
    searches = [xs.MastSearch(fake), xs.ObsCoreSearch(fake, name="ESO", url=xs.ESO_TAP)]
    report = xs.search_external_spectra(HD114762, searches, timeout_s=0.5)
    fake.release.set()
    by = {r.archive: r for r in report.results}
    assert by["MAST"].status is xs.SearchStatus.TIMEOUT
    assert by["ESO"].status is xs.SearchStatus.FOUND


def test_network_disabled_skips_everything_and_sends_nothing():
    fake = FakeTransport()
    report = xs.search_external_spectra(HD114762, xs.default_searches(fake), allow_network=False)
    assert {r.status for r in report.results} == {xs.SearchStatus.SKIPPED}
    assert fake.calls == []


def test_records_carry_separation_and_radius_includes_pm():
    fake = FakeTransport(tap={xs.ESO_TAP: eso_table()})
    report = xs.search_external_spectra(
        HD114762, [xs.ObsCoreSearch(fake, name="ESO", url=xs.ESO_TAP)], timeout_s=10)
    (res,) = report.results
    assert res.radius_arcsec == pytest.approx(HD114762.search_radius_arcsec(20.0))
    assert str(round(res.radius_arcsec / 3600, 6))[:6] in fake.last_adql
    # The 2006 UVES spectrum is ~6.6 arcsec from the J2017.5 position, but on
    # the star's path.
    r = res.records[0]
    assert r.separation_arcsec < 1.0
    ra0, dec0 = HD114762.ra_deg, HD114762.dec_deg
    assert xs._angular_distance_arcsec(ra0, dec0, r.ra_deg, r.dec_deg) > 5.0


def test_mast_request_filters_public_spectra_and_passes_radius_in_degrees():
    fake = FakeTransport()
    xs.search_external_spectra(HD114762, [xs.MastSearch(fake)], timeout_s=10)
    params = fake.last_mast["params"]
    assert {f["paramName"]: f["values"] for f in params["filters"]} == {
        "dataproduct_type": ["spectrum"], "dataRights": ["PUBLIC"]}
    ra, dec, rdeg = (float(x) for x in params["position"].split(","))
    assert rdeg * 3600 == pytest.approx(HD114762.search_radius_arcsec(40.0))


def test_report_serialises_with_missing_values_as_null():
    fake = FakeTransport(tap={xs.ESO_TAP: eso_table()})
    report = xs.search_external_spectra(
        HD114762, [xs.ObsCoreSearch(fake, name="ESO", url=xs.ESO_TAP)], timeout_s=10)
    text = report.to_json()
    assert '"status": "found"' in text and "NaN" not in text


def test_no_network_at_import():
    import importlib
    import sys

    sys.modules.pop("gaia_dr4_explorer.data.external_spectra", None)
    before = {m for m in sys.modules if m.split(".")[0] in ("pyvo", "requests")}
    importlib.import_module("gaia_dr4_explorer.data.external_spectra")
    after = {m for m in sys.modules if m.split(".")[0] in ("pyvo", "requests")}
    assert after == before, "importing the module must not import network libraries"


