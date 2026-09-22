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
