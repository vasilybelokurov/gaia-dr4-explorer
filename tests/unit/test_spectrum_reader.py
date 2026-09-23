"""The spectrum reader, on small files built in each archive's layout.

Each fixture copies the structure, column names, units and header keywords
of a real file downloaded from that archive on 2026-09-23, at a few pixels.
"""

import numpy as np
import pytest
from astropy.io import fits

from gaia_dr4_explorer.data import spectrum_reader
from gaia_dr4_explorer.data.spectrum_reader import SpectrumReadError, read_spectrum


@pytest.fixture(autouse=True)
def few_pixels_allowed(monkeypatch, request):
    """Layout fixtures use a handful of pixels; the minimum has its own test."""
    if "minimum" not in request.node.name:
        monkeypatch.setattr(spectrum_reader, "MIN_PIXELS", 1)


def write(path, *hdus):
    fits.HDUList([fits.PrimaryHDU(), *hdus] if not isinstance(hdus[0], fits.PrimaryHDU)
                 else list(hdus)).writeto(path)
    return path


def test_table_with_one_row_per_pixel_polarbase(tmp_path):
    cols = [fits.Column("AWAV", "E", unit="1 nm", array=[370.0, 500.0, 1048.0]),
            fits.Column("FLUX_NOR", "E", unit="1.E-26 jy", array=[0.9, 1.0, np.nan]),
            fits.Column("FLUX_ERR", "E", unit="c", array=[0.01, 0.01, 0.01])]
    s = read_spectrum(write(tmp_path / "pb.fts", fits.BinTableHDU.from_columns(cols)))
    assert s.wavelength_nm == pytest.approx([370.0, 500.0, 1048.0])
    assert np.isnan(s.flux[2]), "missing flux stays missing"
    assert s.flux_unit == "1.E-26 jy" and not s.normalised   # reported as declared
    assert s.flux_error is not None


def test_table_with_the_whole_spectrum_in_one_row_eso(tmp_path):
    n = 5
    wave = np.linspace(3024.0, 3884.0, n)[None, :]
    cols = [fits.Column("WAVE", f"{n}D", unit="angstrom", array=wave),
            fits.Column("FLUX_REDUCED", f"{n}E", unit="adu", array=np.full((1, n), 7.0)),
            fits.Column("ERR_REDUCED", f"{n}E", unit="adu", array=np.full((1, n), 1.0)),
            fits.Column("FLUX", f"{n}E",
                        unit="10**(-16)erg.cm**(-2).s**(-1).angstrom**(-1)",
                        array=np.full((1, n), 2.0)),
            fits.Column("ERR", f"{n}E", unit="10**(-16)erg.cm**(-2).s**(-1).angstrom**(-1)",
                        array=np.full((1, n), 0.1))]
    s = read_spectrum(write(tmp_path / "eso.fits", fits.BinTableHDU.from_columns(cols)))
    assert s.range_nm == pytest.approx((302.4, 388.4))
    assert (s.flux == 2.0).all(), "the calibrated FLUX is preferred over FLUX_REDUCED"
    assert s.layout == "table (one row)"


def test_table_with_log10_wavelength_and_inverse_variance_boss(tmp_path):
    loglam = np.log10([3600.0, 5000.0, 10400.0])
    cols = [fits.Column("flux", "E", array=[1.0, 2.0, 3.0]),
            fits.Column("loglam", "E", array=loglam),
            fits.Column("ivar", "E", array=[4.0, 0.0, 1.0])]
    primary = fits.PrimaryHDU()
    primary.header["BUNIT"] = "1E-17 erg/cm^2/s/Ang"
    s = read_spectrum(write(tmp_path / "boss.fits", primary, fits.BinTableHDU.from_columns(cols)))
    assert s.wavelength_nm == pytest.approx([360.0, 500.0, 1040.0], rel=1e-5)
    assert s.flux_unit == "1E-17 erg/cm^2/s/Ang"
    assert s.flux_error[0] == pytest.approx(0.5)
    assert np.isnan(s.flux_error[1]), "ivar = 0 means no measurement, not zero error"


def test_image_with_linear_wavelength_in_tenths_of_nm_elodie(tmp_path):
    hdu = fits.PrimaryHDU(np.ones(4, dtype="f4"))
    hdu.header.update({"CTYPE1": "AWAV", "CUNIT1": "0.1 nm", "CRVAL1": 4000.0,
                       "CDELT1": 0.05, "CD1_1": 0.05, "CRPIX1": 1.0, "BUNIT": "instrumental"})
    s = read_spectrum(write(tmp_path / "elodie.fits", hdu))
    assert s.wavelength_nm == pytest.approx([400.0, 400.005, 400.01, 400.015])
    assert s.flux_unit == "instrumental"


def test_image_with_angstrom_axis_cfa(tmp_path):
    hdu = fits.PrimaryHDU(np.ones(3, dtype="f4"))
    hdu.header.update({"CTYPE1": "WAVE-V2W", "CUNIT1": "Angstrom", "CRVAL1": 5165.76953125,
                       "CDELT1": 0.0222085674157303, "CRPIX1": 1.0, "DC-FLAG": 0})
    s = read_spectrum(write(tmp_path / "cfa.fits", hdu))
    assert s.wavelength_nm[0] == pytest.approx(516.576953125)
    assert any("V2W" in n for n in s.notes)


def test_two_dimensional_log_linear_image_apogee(tmp_path):
    data = np.vstack([np.arange(3, dtype="f4"), np.full(3, 99, dtype="f4")])
    img = fits.ImageHDU(data)
    img.header.update({"CTYPE1": "LOG-LINEAR", "CRVAL1": 4.179, "CDELT1": 6e-06, "CRPIX1": 1,
                       "BUNIT": "Flux (10^-17 erg/s/cm^2/Ang)"})
    s = read_spectrum(write(tmp_path / "apstar.fits", img))
    assert s.wavelength_nm[0] == pytest.approx(10 ** 4.179 / 10)
    assert s.flux.tolist() == [0.0, 1.0, 2.0], "row 0 is the combined spectrum"
    assert any("Angstrom" in n for n in s.notes), "an assumed unit must be stated"


def test_wavelength_and_flux_in_two_files_hst_ghrs(tmp_path):
    w = fits.PrimaryHDU(np.array([2879.0, 2900.0, 2924.0]))
    w.header.update({"CTYPE1": "PIXEL", "BUNIT": "ANGSTROMS"})
    f = fits.PrimaryHDU(np.array([1.0, 2.0, 3.0]))
    f.header.update({"CTYPE1": "PIXEL", "BUNIT": "COUNTS"})
    s = read_spectrum([write(tmp_path / "x_c0f.fits", w), write(tmp_path / "x_c1f.fits", f)])
    assert s.range_nm == pytest.approx((287.9, 292.4))
    assert s.flux_unit == "COUNTS"


def test_a_pixel_axis_alone_is_not_a_spectrum(tmp_path):
    hdu = fits.PrimaryHDU(np.ones(3))
    hdu.header.update({"CTYPE1": "PIXEL", "CRVAL1": 1.0, "CD1_1": 1.0, "CRPIX1": 1.0})
    with pytest.raises(SpectrumReadError, match="pixels"):
        read_spectrum(write(tmp_path / "c1f_alone.fits", hdu))


def test_an_undeclared_table_unit_is_guessed_and_the_guess_is_stated(tmp_path):
    cols = [fits.Column("WAVE", "E", array=[4000.0, 5000.0]),
            fits.Column("FLUX", "E", array=[1.0, 1.0])]
    s = read_spectrum(write(tmp_path / "t.fits", fits.BinTableHDU.from_columns(cols)))
    assert s.wavelength_nm == pytest.approx([400.0, 500.0])
    assert any("assumed Angstrom" in n for n in s.notes)


def test_a_file_without_a_spectrum_is_refused(tmp_path):
    cols = [fits.Column("RA", "E", array=[1.0]), fits.Column("DEC", "E", array=[2.0])]
    with pytest.raises(SpectrumReadError):
        read_spectrum(write(tmp_path / "cat.fits", fits.BinTableHDU.from_columns(cols)))


def test_a_non_wavelength_unit_is_refused(tmp_path):
    cols = [fits.Column("WAVE", "E", unit="s", array=[1.0]),
            fits.Column("FLUX", "E", array=[1.0])]
    with pytest.raises(SpectrumReadError, match="not a wavelength"):
        read_spectrum(write(tmp_path / "bad.fits", fits.BinTableHDU.from_columns(cols)))


def test_echelle_orders_one_per_row_are_joined_in_wavelength_order_stis(tmp_path):
    wave = np.array([[2500.0, 2510.0, 2520.0], [2400.0, 2410.0, 2420.0]])
    flux = np.array([[3.0, 3.0, 3.0], [1.0, 1.0, 1.0]])
    cols = [fits.Column("WAVELENGTH", "3D", unit="Angstrom", array=wave),
            fits.Column("FLUX", "3E", unit="erg/s/cm**2/Angstrom", array=flux),
            fits.Column("ERROR", "3E", unit="erg/s/cm**2/Angstrom", array=flux / 10)]
    s = read_spectrum(write(tmp_path / "x1d.fits", fits.BinTableHDU.from_columns(cols)))
    assert np.all(np.diff(s.wavelength_nm) > 0)
    assert s.flux.tolist() == [1.0, 1.0, 1.0, 3.0, 3.0, 3.0], "flux follows its wavelength"
    assert s.flux_error.tolist() == pytest.approx([0.1, 0.1, 0.1, 0.3, 0.3, 0.3])
    assert "2 orders" in s.layout


def test_minimum_pixel_count_refuses_a_three_pixel_spectrum(tmp_path):
    """A raw GHRS file was once read as 3 'pixels' at 19.8 nm."""
    hdu = fits.PrimaryHDU(np.ones(3))
    hdu.header.update({"CTYPE1": "WAVE", "CUNIT1": "Angstrom", "CRVAL1": 198.0,
                       "CDELT1": 0.01, "CRPIX1": 1.0})
    with pytest.raises(SpectrumReadError, match="only 3 pixels"):
        read_spectrum(write(tmp_path / "d1f.fits", hdu))


def test_upena_rows_are_read_as_wavelength_intensity_error(tmp_path):
    n = 5
    rows = []
    for _ in range(4):
        rows += [np.linspace(369.1, 1048.0, n), np.full(n, 0.98), np.full(n, 0.004)]
    hdu = fits.PrimaryHDU(np.array(rows))
    hdu.header["COMMENT"] = "Fully reduced by Upena at CFHT"
    hdu.header["COMMENT"] = "Upena uses J-F. Donati's software Libre-ESpRIT"
    s = read_spectrum(write(tmp_path / "1701499i.fits", hdu))
    assert s.range_nm == pytest.approx((369.1, 1048.0))
    assert (s.flux == 0.98).all() and s.normalised
    assert "Upena" in s.layout


def test_a_12_row_image_without_upena_in_the_header_is_not_guessed(tmp_path):
    hdu = fits.PrimaryHDU(np.ones((12, 5)))
    with pytest.raises(SpectrumReadError):
        read_spectrum(write(tmp_path / "unknown.fits", hdu))
