"""Read a 1-D spectrum from an archive FITS file, whatever its layout.

Archives package spectra differently. The layouts handled here were found in
real files from each archive the app searches (2026-09-23):

=====================================  ==========================================
table, one row per pixel               PolarBase (``AWAV`` in "1 nm", ``FLUX_NOR``)
table, whole spectrum in one row       ESO Phase 3 (``WAVE``, ``FLUX``), IUE (VO)
table, one echelle order per row       HST STIS ``x1d`` (orders joined)
table with log10 wavelength            SDSS BOSS (``loglam``, ``flux``, ``ivar``)
1-D image with a linear wavelength     CfA TDC, ELODIE (``CRVAL1``/``CDELT1``)
2-D image with log-linear wavelength   SDSS APOGEE apStar (row 0: combined)
two files: wavelengths + fluxes        HST GHRS (``_c0f`` + ``_c1f``)
rows of wavelength/intensity/error     CFHT Upena (ESPaDOnS at CADC), no WCS
=====================================  ==========================================

Wavelengths are converted to nm from the units each file declares. Where a
file declares none, the assumption is stated in :attr:`Spectrum.notes`, never
made silently. Fluxes keep their own units: archives mix calibrated fluxes,
counts and normalised spectra, and converting between them is not possible.
Missing values stay NaN.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Column names recognised, in order of preference (matched case-insensitively).
WAVE_COLUMNS = ("wave", "wavelength", "lambda", "awav", "loglam")
#: Calibrated flux before reduced/raw variants: ESO files carry both.
FLUX_COLUMNS = ("flux", "flux_nor", "flux_norm", "flux_reduced")
ERROR_COLUMNS = ("err", "error", "sigma", "flux_err", "err_reduced", "ivar")


class SpectrumReadError(ValueError):
    """The file is not a 1-D spectrum this reader understands."""


@dataclass(frozen=True)
class Spectrum:
    """A 1-D spectrum. ``wavelength_nm`` is in nm; flux in ``flux_unit``, the
    unit the file declares (which archives do not always get right)."""

    wavelength_nm: np.ndarray
    flux: np.ndarray
    flux_unit: str
    flux_error: np.ndarray | None = None
    normalised: bool = False
    layout: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.wavelength_nm.shape != self.flux.shape:
            raise SpectrumReadError("wavelength and flux have different lengths")

    @property
    def n_pixels(self) -> int:
        return int(self.flux.size)

    @property
    def range_nm(self) -> tuple[float, float]:
        w = self.wavelength_nm[np.isfinite(self.wavelength_nm)]
        return (float(w.min()), float(w.max())) if w.size else (float("nan"), float("nan"))


def read_spectrum(paths: Path | str | list) -> Spectrum:
    """Read one spectrum from a file, or from a wavelength + flux file pair.

    Parameters
    ----------
    paths : path or list of paths
        One FITS file, or ``[wavelength_file, flux_file]`` for layouts that
        split them (HST GHRS ``_c0f`` + ``_c1f``).
    """
    from astropy.io import fits

    items = [Path(p) for p in (paths if isinstance(paths, list | tuple) else [paths])]
    if len(items) == 2:
        return _check(_read_pair(*items))
    if len(items) != 1:
        raise SpectrumReadError(f"expected one file or a pair, got {len(items)}")
    with fits.open(items[0], memmap=False) as hdul:
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU | fits.TableHDU) and hdu.data is not None:
                found = _from_table(hdu, hdul[0].header)
                if found is not None:
                    return _check(found)
        if _is_upena(hdul[0]):
            return _check(_from_upena(hdul[0]))
        for hdu in hdul:
            if getattr(hdu, "is_image", False) and hdu.data is not None and hdu.data.ndim in (1, 2):
                return _check(_from_image(hdu))
    raise SpectrumReadError("no table with wavelength and flux columns, and no 1-D image")


# ------------------------------------------------------------------ layouts


def _from_table(hdu, primary_header) -> Spectrum | None:
    names = {n.lower(): n for n in hdu.columns.names}
    wcol = next((names[c] for c in WAVE_COLUMNS if c in names), None)
    fcol = next((names[c] for c in FLUX_COLUMNS if c in names), None)
    if wcol is None or fcol is None:
        return None
    ecol = next((names[c] for c in ERROR_COLUMNS if c in names), None)
    data = hdu.data
    notes = []

    rows = np.ndim(data[wcol]) == 2 and len(data) > 1

    def column(name):
        values = np.asarray(data[name], dtype=float)
        if values.ndim == 2:     # spectrum in one row, or one echelle order per row
            values = values.ravel()
        return np.where(np.isfinite(values), values, np.nan)

    wave = column(wcol)
    layout = "table (one row)" if np.ndim(data[wcol]) == 2 else "table"
    order = None
    if rows:
        # Echelle orders, one per row (HST STIS x1d): joined in wavelength order.
        order = np.argsort(wave, kind="stable")
        layout = f"table ({len(data)} orders joined)"
        notes.append(f"{len(data)} echelle orders joined in wavelength order; "
                     "overlapping order ends are kept as they are")
    unit = _column_unit(hdu, wcol)
    if wcol.lower() == "loglam":
        wave = 10.0 ** wave
        unit = unit or "Angstrom"
        layout = "table (log10 wavelength)"
    if not unit:
        unit = _guess_wave_unit(wave)
        notes.append(f"wavelength unit not declared; assumed {unit} from the values")
    wave_nm = _to_nm(wave, unit)

    flux = column(fcol)
    if order is not None:
        wave_nm, flux = wave_nm[order], flux[order]
    flux_unit = _column_unit(hdu, fcol) or str(primary_header.get("BUNIT", "") or "")
    error = None
    if ecol is not None:
        error = column(ecol)
        if ecol.lower() == "ivar":
            with np.errstate(divide="ignore", invalid="ignore"):
                error = np.where(error > 0, 1.0 / np.sqrt(error), np.nan)
        if order is not None:
            error = error[order]
    # Normalisation is not inferred from column names: PolarBase names the
    # column FLUX_NOR, with unit "1.E-26 jy", in both its normalised (median
    # 0.98) and unnormalised (median 0.096) products. The declared unit is
    # reported as given.
    return Spectrum(wave_nm, flux, flux_unit or "unknown", error, False, layout,
                    tuple(notes))


def _from_image(hdu) -> Spectrum:
    header = hdu.header
    data = np.asarray(hdu.data, dtype=float)
    notes = []
    if data.ndim == 2:
        data = data[0]
        notes.append("first row of a 2-D image (the combined spectrum in APOGEE apStar)")
    ctype = str(header.get("CTYPE1", "")).upper()
    if ctype == "PIXEL":
        raise SpectrumReadError("image axis is in pixels, not wavelength")
    crval = header.get("CRVAL1")
    delta = header.get("CD1_1", header.get("CDELT1"))
    crpix = header.get("CRPIX1", 1.0)
    if crval is None or delta is None:
        raise SpectrumReadError("image has no wavelength solution (CRVAL1/CDELT1)")
    pix = np.arange(data.size, dtype=float) + 1.0
    axis = crval + delta * (pix - crpix)
    log = "LOG" in ctype or int(header.get("DC-FLAG", 0) or 0) == 1
    unit = str(header.get("CUNIT1", "") or "")
    if log:
        axis = 10.0 ** axis
        layout = "image (log-linear wavelength)"
    else:
        layout = "image (linear wavelength)"
    if not unit:
        unit = "Angstrom"
        notes.append("wavelength unit not declared; assumed Angstrom (the FITS convention "
                     "for these archives)")
    if ctype.endswith("-V2W"):
        notes.append("axis declared WAVE-V2W; treated as linear over this short range")
    flux = np.where(np.isfinite(data), data, np.nan)
    return Spectrum(_to_nm(axis, unit), flux, str(header.get("BUNIT", "") or "unknown"),
                    None, False, layout, tuple(notes))


#: Fewer pixels than this is not a spectrum (a raw GHRS file once read as 3).
MIN_PIXELS = 16


def _check(s: Spectrum) -> Spectrum:
    n = int(np.isfinite(s.wavelength_nm).sum())
    if n < MIN_PIXELS:
        raise SpectrumReadError(f"only {n} pixels with a wavelength: not a spectrum")
    return s


def _is_upena(hdu) -> bool:
    text = str(hdu.header.get("COMMENT", ""))
    return (hdu.data is not None and np.ndim(hdu.data) == 2 and hdu.data.shape[0] in (6, 12)
            and ("Upena" in text or "Libre-ESpRIT" in text))


def _from_upena(hdu) -> Spectrum:
    """CFHT ESPaDOnS/Narval reductions by Upena (Libre-ESpRIT), no WCS.

    Rows come in threes -- wavelength [nm], intensity, error -- normalised
    first, then unnormalised; 12-row files repeat both without the automatic
    wavelength correction. Checked against PolarBase's labelled copy of the
    same observation (1701499i): rows 0-2 match its normalised product
    (median 0.98). Rows 0-2 are used.
    """
    d = np.asarray(hdu.data, dtype=float)
    return Spectrum(d[0], np.where(np.isfinite(d[1]), d[1], np.nan), "normalised",
                    d[2], True, "Upena rows (wavelength, intensity, error)",
                    ("Upena reduction: normalised intensity with automatic wavelength "
                     "correction (rows 0-2)",))


def _read_pair(wave_path: Path, flux_path: Path) -> Spectrum:
    from astropy.io import fits

    with fits.open(wave_path, memmap=False) as w, fits.open(flux_path, memmap=False) as f:
        wave = np.asarray(w[0].data, dtype=float)
        flux = np.asarray(f[0].data, dtype=float)
        wunit = str(w[0].header.get("BUNIT", "") or "")
        funit = str(f[0].header.get("BUNIT", "") or "unknown")
    if wave.shape != flux.shape:
        raise SpectrumReadError(f"wavelength {wave.shape} and flux {flux.shape} differ")
    if wave.ndim == 2:
        wave, flux = wave[0], flux[0]
    unit = {"ANGSTROMS": "Angstrom", "ANGSTROM": "Angstrom"}.get(wunit.upper(), wunit)
    notes = ()
    if not unit:
        unit = "Angstrom"
        notes = ("wavelength unit not declared; assumed Angstrom",)
    return Spectrum(_to_nm(wave, unit), flux, funit, None, False,
                    "wavelength + flux file pair", notes)


# ------------------------------------------------------------------ helpers


def _column_unit(hdu, name) -> str:
    unit = hdu.columns[name].unit
    return str(unit).strip() if unit else ""


def _to_nm(values: np.ndarray, unit: str) -> np.ndarray:
    import astropy.units as u

    text = unit.strip()
    try:
        q = u.Unit(text, parse_strict="raise")
    except ValueError:
        aliases = {"angstroms": "Angstrom", "a": "Angstrom", "angs": "Angstrom"}
        try:
            q = u.Unit(aliases.get(text.lower(), text), format="fits")
        except ValueError as exc:
            raise SpectrumReadError(f"unrecognised wavelength unit {unit!r}") from exc
    try:
        return (values * q).to_value(u.nm, equivalencies=u.spectral())
    except u.UnitConversionError as exc:
        raise SpectrumReadError(f"{unit!r} is not a wavelength unit") from exc


def _guess_wave_unit(values: np.ndarray) -> str:
    w = values[np.isfinite(values) & (values > 0)]
    if not w.size:
        raise SpectrumReadError("wavelength column is empty")
    median = float(np.median(w))
    if median > 1e3:
        return "Angstrom"
    if median > 50:
        return "nm"
    if median < 1e-4:
        return "m"
    return "micron"
