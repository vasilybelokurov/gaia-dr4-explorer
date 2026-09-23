"""Turning a search record into files on disk and a spectrum, offline."""

import io
import json

import numpy as np
import pytest
from astropy.io import fits

from gaia_dr4_explorer.data import external_spectra as xs
from gaia_dr4_explorer.data import spectrum_files as sf


def fits_bytes(n=32, start=4000.0):
    hdu = fits.PrimaryHDU(np.linspace(1.0, 2.0, n).astype("f4"))
    hdu.header.update({"CTYPE1": "AWAV", "CUNIT1": "Angstrom", "CRVAL1": start,
                       "CDELT1": 1.0, "CRPIX1": 1.0, "BUNIT": "adu"})
    buf = io.BytesIO()
    hdu.writeto(buf)
    return buf.getvalue()


class Files(xs.Transport):
    def __init__(self, files=None, mast=None):
        self.files, self._mast, self.requests = files or {}, mast, []

    def get_bytes(self, url, max_bytes):
        self.requests.append(url)
        data = self.files[url]
        if len(data) > max_bytes:
            raise ValueError("too big")
        return data

    def mast(self, request):
        self.requests.append(request["service"])
        return self._mast


def record(**kw):
    base = dict(archive="X", access_url="https://example.org/a.fits",
                access_format="application/fits")
    base.update(kw)
    return xs.SpectrumRecord(**base)


# ------------------------------------------------------------- URL rules


def test_eso_datalink_becomes_the_direct_file_without_a_request():
    rec = record(archive="ESO", access_url=(
        "https://archive.eso.org/datalink/links?ID=ivo://eso.org/ID?ADP.2021-09-14T11:25:35.546"),
        access_format="application/x-votable+xml;content=datalink")
    t = Files()
    assert sf.resolve_file_urls(rec, t) == [
        "https://dataportal.eso.org/dataportal_new/file/ADP.2021-09-14T11:25:35.546"]
    assert t.requests == []


def test_datalink_picks_the_this_fits_file():
    votable = b"""<?xml version="1.0"?><VOTABLE version="1.3"><RESOURCE type="results"><TABLE>
    <FIELD name="ID" datatype="char" arraysize="*"/><FIELD name="access_url" datatype="char" arraysize="*"/>
    <FIELD name="semantics" datatype="char" arraysize="*"/><FIELD name="content_type" datatype="char" arraysize="*"/>
    <DATA><TABLEDATA>
    <TR><TD>a</TD><TD>https://x/preview.png</TD><TD>#preview</TD><TD>image/png</TD></TR>
    <TR><TD>a</TD><TD>https://x/1701499i.fits</TD><TD>#this</TD><TD>application/fits</TD></TR>
    </TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"""
    assert sf.datalink_file(votable) == "https://x/1701499i.fits"
    empty = votable.replace(b"https://x/1701499i.fits", b"")
    with pytest.raises(sf.FileResolutionError, match="no #this"):
        sf.datalink_file(empty)


def test_apogee_is_fetched_from_the_sdss_server_not_mast():
    url = (xs.MAST_DOWNLOAD + "mast:SDSS/apogee/apo25m/M53/2M13121982+1731016/"
           "apStar-dr17-2M13121982+1731016.fits")
    assert sf.apogee_sas_url(url) == (
        "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/apo25m/M53/"
        "apStar-dr17-2M13121982+1731016.fits")


def test_hst_uses_the_product_list_and_prefers_x1d():
    products = {"data": [
        {"productType": "SCIENCE", "productSubGroupDescription": "RAW", "dataRights": "PUBLIC",
         "dataURI": "mast:HST/product/o_raw.fits"},
        {"productType": "SCIENCE", "productSubGroupDescription": "X1D", "dataRights": "PUBLIC",
         "dataURI": "mast:HST/product/o_x1d.fits"},
        {"productType": "PREVIEW", "productSubGroupDescription": None, "dataRights": "PUBLIC",
         "dataURI": "mast:HST/product/o_x1d.png"}]}
    rec = record(archive="MAST", collection="HST", archive_key="24919382",
                 access_url=xs.MAST_DOWNLOAD + "mast:HST/product/o_d1f.fits")
    assert sf.resolve_file_urls(rec, Files(mast=products)) == [
        xs.MAST_DOWNLOAD + "mast:HST/product/o_x1d.fits"]


def test_ghrs_pairs_the_wavelength_file_with_the_flux_file():
    products = {"data": [
        {"productType": "SCIENCE", "productSubGroupDescription": g, "dataRights": "PUBLIC",
         "dataURI": f"mast:HST/product/z_{g.lower()}.fits"} for g in ("C1F", "C0F")]}
    rec = record(archive="MAST", collection="HST", archive_key="1")
    assert sf.resolve_file_urls(rec, Files(mast=products)) == [
        xs.MAST_DOWNLOAD + "mast:HST/product/z_c0f.fits",
        xs.MAST_DOWNLOAD + "mast:HST/product/z_c1f.fits"]


def test_no_spectrum_product_says_what_there_is():
    products = {"data": [{"productType": "SCIENCE", "productSubGroupDescription": "RAW",
                          "dataRights": "PUBLIC", "dataURI": "mast:x"}]}
    rec = record(archive="MAST", collection="HST", archive_key="1", obs_id="z2xm0601t")
    with pytest.raises(sf.FileResolutionError, match="z2xm0601t.*RAW"):
        sf.resolve_file_urls(rec, Files(mast=products))


# ------------------------------------------------------------ the store


def test_download_is_kept_with_a_sidecar_and_reused(tmp_path):
    t = Files({"https://example.org/a.fits": fits_bytes()})
    store = sf.SpectrumFileStore(tmp_path, t)
    first = store.load(record())
    assert first.downloaded and first.spectrum.n_pixels == 32
    side = json.loads(first.paths[0].with_name(first.paths[0].name + ".json").read_text())
    assert side["url"] == "https://example.org/a.fits" and len(side["sha256"]) == 64
    second = store.load(record())
    assert not second.downloaded and t.requests.count("https://example.org/a.fits") == 1


def test_cached_file_is_never_rewritten(tmp_path):
    t = Files({"https://example.org/a.fits": fits_bytes()})
    store = sf.SpectrumFileStore(tmp_path, t)
    path = store.load(record()).paths[0]
    mtime = path.stat().st_mtime_ns
    t.files["https://example.org/a.fits"] = fits_bytes(start=9000.0)
    store.load(record())
    assert path.stat().st_mtime_ns == mtime


def test_a_non_fits_answer_is_refused_and_not_kept(tmp_path):
    t = Files({"https://example.org/a.fits": b"<html>login required</html>"})
    store = sf.SpectrumFileStore(tmp_path, t)
    with pytest.raises(sf.FileResolutionError, match="did not return a FITS"):
        store.load(record())
    assert not store.path_for("https://example.org/a.fits").exists()


def test_the_size_limit_is_enforced(tmp_path):
    t = Files({"https://example.org/a.fits": fits_bytes()})
    with pytest.raises(ValueError, match="too big"):
        sf.SpectrumFileStore(tmp_path, t, max_bytes=100).load(record())


def test_with_the_network_off_only_cached_files_load(tmp_path):
    t = Files({"https://example.org/a.fits": fits_bytes()})
    with pytest.raises(sf.FileResolutionError, match="network access is off"):
        sf.SpectrumFileStore(tmp_path, t, allow_network=False).load(record())
    sf.SpectrumFileStore(tmp_path, t).load(record())
    assert sf.SpectrumFileStore(tmp_path, t, allow_network=False).load(record()).spectrum


def test_cache_paths_are_unique_and_readable(tmp_path):
    store = sf.SpectrumFileStore(tmp_path, Files())
    a = store.path_for("https://a.org/x/spec.fits")
    b = store.path_for("https://b.org/x/spec.fits")
    assert a != b and a.name.endswith("spec.fits")
