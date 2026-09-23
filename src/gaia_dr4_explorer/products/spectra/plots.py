"""BP/RP spectrum plots.

The DR3 ``XP_SAMPLED`` product is a single mean spectrum sampled on a common
wavelength grid. DR4 will add epoch spectra; the multi-epoch views this module
anticipates are described in PLAN.md and are not built against absent data.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd

#: Nominal BP/RP changeover, for orientation only. The sampled product is a
#: single combined spectrum, so this is drawn as a guide, not a split.
BP_RP_CROSSOVER_NM = 640.0


def to_frame(table) -> pd.DataFrame:
    """Flatten the sampled spectrum to a plotting frame."""
    data = {}
    for name in table.colnames:
        col = table[name]
        if getattr(col, "mask", None) is not None:
            data[name] = np.ma.filled(col, np.nan)
        else:
            data[name] = np.asarray(col)
    frame = pd.DataFrame(data)
    if "flux" in frame and "flux_error" in frame:
        with np.errstate(invalid="ignore", divide="ignore"):
            frame["flux_over_error"] = frame["flux"] / frame["flux_error"]
    return frame


def mean_spectrum(frame: pd.DataFrame, *, show_errors: bool = True) -> hv.Overlay:
    """Flux against wavelength, with the uncertainty band."""
    if frame.empty:
        return _empty("No spectrum served for this source")
    layers = []
    if show_errors and "flux_error" in frame.columns:
        band = frame.assign(
            lo=frame["flux"] - frame["flux_error"],
            hi=frame["flux"] + frame["flux_error"],
        )
        layers.append(
            hv.Area(band, kdims=["wavelength"], vdims=["lo", "hi"], label="±1σ").opts(
                color="#9ecae1", alpha=0.45, line_width=0
            )
        )
    layers.append(
        hv.Curve(frame, kdims=["wavelength"], vdims=["flux"], label="mean spectrum").opts(
            color="#08519c", line_width=1.6, tools=["hover"]
        )
    )
    layers.append(
        hv.VLine(BP_RP_CROSSOVER_NM).opts(color="#999999", line_dash="dotted", line_width=1)
    )
    return hv.Overlay(layers).opts(
        responsive=True, height=360, legend_position="top_right",
        xlabel="wavelength [nm]",
        ylabel="flux [W m⁻² nm⁻¹]",
        title="Gaia DR3 BP/RP mean spectrum (sampled)",
    )


def signal_to_noise(frame: pd.DataFrame) -> hv.Curve:
    """Flux over its uncertainty, across the range."""
    if "flux_over_error" not in frame.columns:
        return _empty("No flux uncertainties served")
    return hv.Curve(
        frame, kdims=["wavelength"], vdims=["flux_over_error"],
    ).opts(
        responsive=True, height=260, color="#08519c", tools=["hover"],
        xlabel="wavelength [nm]", ylabel="flux / flux_error [dimensionless]",
        title="Signal-to-noise across the spectrum",
    )


def _empty(message: str) -> hv.Text:
    return hv.Text(0.5, 0.5, message).opts(
        xaxis=None, yaxis=None, responsive=True, height=200, color="#888888"
    )


# ------------------------------------------------- spectra from other archives

#: Points drawn per spectrum. Longer spectra are thinned by keeping each bin's
#: minimum and maximum, so narrow lines survive; the plot says when it happens.
MAX_DRAWN_POINTS = 20_000


def thin_minmax(x: np.ndarray, y: np.ndarray, max_points: int = MAX_DRAWN_POINTS):
    """Reduce (x, y) to at most ~``max_points`` keeping each bin's extremes.

    Returns ``(x, y, thinned)``. Non-finite samples are dropped first.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size <= max_points:
        return x, y, False
    nbins = max_points // 2
    edges = np.linspace(0, x.size, nbins + 1).astype(int)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            continue
        seg = y[a:b]
        i, j = a + int(np.argmin(seg)), a + int(np.argmax(seg))
        for k in sorted((i, j)) if i != j else (i,):
            xs.append(x[k])
            ys.append(y[k])
    return np.asarray(xs), np.asarray(ys), True


def normalise_by_median(flux: np.ndarray) -> np.ndarray:
    """Flux divided by its median over finite, positive samples."""
    good = flux[np.isfinite(flux) & (flux > 0)]
    if not good.size:
        return np.full_like(flux, np.nan, dtype=float)
    return flux / np.median(good)


def external_spectra(loaded, *, normalise: bool = True) -> hv.Element:
    """Overlay of downloaded spectra, flux against wavelength in nm.

    Parameters
    ----------
    loaded : sequence of LoadedSpectrum
    normalise : bool
        Divide each spectrum by its median. Archives mix calibrated fluxes,
        counts and normalised spectra, so this is the default for comparison.
    """
    if not loaded:
        return _empty("Tick a spectrum in the lists above to download and plot it")
    layers, thinned_any = [], False
    units = {item.spectrum.flux_unit for item in loaded}
    for item in loaded:
        s, r = item.spectrum, item.record
        flux = normalise_by_median(s.flux) if normalise else s.flux
        x, y, thinned = thin_minmax(s.wavelength_nm, flux)
        thinned_any |= thinned
        label = " ".join(p for p in (r.archive, r.instrument, _date(r.mjd), r.obs_id[:24]) if p)
        layers.append(hv.Curve((x, y), kdims=["wavelength_nm"], vdims=["flux"], label=label)
                      .opts(line_width=1, tools=["hover"]))
    if normalise:
        ylabel = "flux / median"
    elif len(units) == 1:
        ylabel = f"flux [{next(iter(units))}]"
    else:
        ylabel = "flux (units differ between files; normalise to compare)"
    title = "Downloaded spectra" + (" — thinned for display" if thinned_any else "")
    return hv.Overlay(layers).opts(
        responsive=True, height=380, legend_position="top_right",
        xlabel="Wavelength [nm]", ylabel=ylabel, title=title, shared_axes=False,
    )


def _date(mjd: float) -> str:
    if mjd is None or not np.isfinite(mjd):
        return ""
    from astropy.time import Time

    return Time(mjd, format="mjd").to_value("iso", subfmt="date")
