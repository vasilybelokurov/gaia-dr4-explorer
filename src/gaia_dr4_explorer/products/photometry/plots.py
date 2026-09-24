"""Light-curve plots.

As elsewhere: these take a loaded table and return HoloViews objects, do no
I/O, and size themselves by their container.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd

#: Gaia band colours, conventional and colour-blind distinguishable by shape too.
BAND_COLOURS = {"G": "#2c7d32", "BP": "#1f5fbf", "RP": "#c0392b"}
BAND_MARKERS = {"G": "o", "BP": "s", "RP": "d"}


def to_frame(epochs) -> pd.DataFrame:
    """Flatten the normalized epoch table to a plotting frame."""
    data = {}
    for name in epochs.colnames:
        col = epochs[name]
        if getattr(col, "mask", None) is not None:
            fill = np.nan if np.issubdtype(col.dtype, np.floating) else False
            data[name] = np.ma.filled(col, fill)
        else:
            data[name] = np.asarray(col)
    frame = pd.DataFrame(data)
    # Three states, not two: a masked per-band flag means the archive did not
    # say, which is not the same as saying the epoch was accepted.
    known = ~np.asarray(
        getattr(epochs["rejected"], "mask", np.zeros(len(epochs), dtype=bool))
    )
    rejected = frame["rejected"].astype(bool).to_numpy()
    if "transit_rejected" in frame.columns:
        # The transit-level flag rejects every band of that transit.
        rejected = rejected | frame["transit_rejected"].astype(bool).to_numpy()
    frame["status"] = np.where(
        ~known, "quality unknown", np.where(rejected, "rejected", "accepted")
    )
    if "transit_id" in frame.columns:
        frame["transit_id_str"] = frame["transit_id"].astype("int64").astype(str)
    frame["time_yr"] = 2010.0 + (frame["time_jd_tcb"] - 2455197.5) / 365.25
    return frame


_HOVER = ("transit_id_str", "band", "mag", "flux_error", "flux_over_error",
          "status", "transit_rejected", "other_flags", "time_jd_tcb")


def _hover(frame: pd.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    return [c for c in _HOVER if c in frame.columns and c not in exclude]


def light_curve(frame: pd.DataFrame, *, bands: list[str] | None = None) -> hv.Overlay:
    """Magnitude against time, one series per band, brighter upwards."""
    bands = bands or sorted(set(frame["band"]))
    layers = []
    for band in ("BP", "G", "RP"):
        if band not in bands:
            continue
        sub = frame[frame["band"] == band]
        if sub.empty:
            continue
        accepted = sub[sub["status"] == "accepted"]
        unknown = sub[sub["status"] == "quality unknown"]
        rejected = sub[sub["status"] == "rejected"]
        if not unknown.empty:
            # Neither accepted nor rejected: shown, and shown as different.
            layers.append(
                hv.Points(
                    unknown, kdims=["time_yr", "mag"],
                    vdims=_hover(unknown, ("mag",)), label=f"{band} quality unknown",
                ).opts(
                    color="#999999", marker="triangle", size=6, alpha=0.9,
                    tools=["hover"],
                )
            )
        if not rejected.empty:
            # Rejected epochs stay visible, drawn hollow rather than hidden.
            layers.append(
                hv.Points(
                    rejected, kdims=["time_yr", "mag"],
                    vdims=_hover(rejected, ("mag",)), label=f"{band} rejected",
                ).opts(
                    color=BAND_COLOURS[band], marker="x", size=7, alpha=0.9,
                    tools=["hover"],
                )
            )
        if not accepted.empty:
            layers.append(
                hv.Points(
                    accepted, kdims=["time_yr", "mag"],
                    vdims=_hover(accepted, ("mag",)), label=band,
                ).opts(
                    color=BAND_COLOURS[band], marker=BAND_MARKERS[band], size=5,
                    alpha=0.85, tools=["hover", "box_select"],
                )
            )
    if not layers:
        return _empty("No photometry matches the current selection")
    return hv.Overlay(layers).opts(
        responsive=True, height=360, legend_position="right",
        xlabel="Observation time [yr, TCB]", ylabel="magnitude [mag]",
        invert_yaxis=True,   # brighter is up, as magnitudes require
        title="Gaia epoch photometry",
    )


def uncertainty_vs_time(frame: pd.DataFrame) -> hv.Overlay:
    """Flux signal-to-noise against time, per band."""
    layers = []
    for band in ("BP", "G", "RP"):
        sub = frame[(frame["band"] == band) & np.isfinite(frame["flux_over_error"])]
        if sub.empty:
            continue
        layers.append(
            hv.Points(
                sub, kdims=["time_yr", "flux_over_error"],
                vdims=_hover(sub, ("flux_over_error",)), label=band,
            ).opts(color=BAND_COLOURS[band], size=4, alpha=0.8, tools=["hover"])
        )
    if not layers:
        return _empty("No flux uncertainties served for this source")
    return hv.Overlay(layers).opts(
        responsive=True, height=280, legend_position="right",
        xlabel="Observation time [yr, TCB]",
        ylabel="flux / flux_error [dimensionless]",
        title="Signal-to-noise per transit",
    )


def magnitude_distribution(frame: pd.DataFrame) -> hv.Layout:
    """One histogram per band, to show scatter at a glance."""
    panels = []
    for band in ("BP", "G", "RP"):
        sub = frame[(frame["band"] == band) & (frame["status"] == "accepted")]
        values = sub["mag"].to_numpy()
        values = values[np.isfinite(values)]
        if values.size < 2:
            continue
        freq, edges = np.histogram(values, bins=25)
        panels.append(
            hv.Histogram((edges, freq)).opts(
                responsive=True, height=240, color=BAND_COLOURS[band],
                xlabel=f"{band} magnitude [mag]", ylabel="transits",
                title=f"{band}: {values.size} accepted, spread "
                      f"{values.max() - values.min():.3f} mag",
            )
        )
    if not panels:
        return _empty("Not enough accepted photometry to summarise")
    return hv.Layout(panels).cols(len(panels))


def phase_fold(frame: pd.DataFrame, period_days: float, *, epoch_jd: float | None = None):
    """Fold the light curve on an explicitly supplied period.

    No period search is performed: a period must be supplied by the caller or
    taken from a Gaia variability product.
    """
    if not np.isfinite(period_days) or period_days <= 0:
        raise ValueError(f"period must be a positive number of days, got {period_days!r}")
    data = frame.copy()
    zero = epoch_jd if epoch_jd is not None else float(np.nanmin(data["time_jd_tcb"]))
    data["phase"] = ((data["time_jd_tcb"] - zero) / period_days) % 1.0
    layers = []
    for band in ("BP", "G", "RP"):
        sub = data[data["band"] == band]
        if sub.empty:
            continue
        layers.append(
            hv.Points(sub, kdims=["phase", "mag"], vdims=_hover(sub, ("mag",)), label=band).opts(
                color=BAND_COLOURS[band], marker=BAND_MARKERS[band], size=5,
                alpha=0.85, tools=["hover"],
            )
        )
    if not layers:
        return _empty("Nothing to fold")
    return hv.Overlay(layers).opts(
        responsive=True, height=320, legend_position="right", invert_yaxis=True,
        xlabel=f"phase [dimensionless]  (P = {period_days:g} d)",
        ylabel="magnitude [mag]", title="Phase-folded light curve",
    )


def _empty(message: str) -> hv.Text:
    return hv.Text(0.5, 0.5, message).opts(
        xaxis=None, yaxis=None, responsive=True, height=200, color="#888888"
    )


# ------------------------------------------------------------------ ZTF

#: ZTF band colours, distinct from Gaia's G/BP/RP.
ZTF_COLOURS = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}


def ztf_light_curve(frame: pd.DataFrame, *, show_flagged: bool = True) -> hv.Element:
    """ZTF magnitude against time, per band; flagged points as grey crosses.

    Parameters
    ----------
    frame : DataFrame
        From :func:`gaia_dr4_explorer.data.ztf.parse_ztf_csv`: ``mjd``,
        ``mag``, ``magerr``, ``band``, ``clean``, ``oid``, ``sep_arcsec``.
    show_flagged : bool
        Draw points with ``catflags != 0``. They are never removed from the
        data, only hidden from the plot on request.
    """
    if frame is None or not len(frame):
        return _empty("No ZTF points near this source")
    f = frame.copy()
    f["year"] = 2000.0 + (f["mjd"] - 51544.5) / 365.25
    f["oid_str"] = f["oid"].astype("int64").astype(str)   # 64-bit ids lose digits as floats
    hover = [c for c in ("mjd", "magerr", "catflags", "oid_str", "sep_arcsec", "limitmag")
             if c in f.columns]
    layers = []
    for band in ("g", "r", "i"):
        d = f[f["band"] == band]
        clean = d[d["clean"]]
        if len(clean):
            layers.append(hv.ErrorBars(clean, kdims=["year"], vdims=["mag", "magerr"]).opts(
                color=ZTF_COLOURS[band], alpha=0.4, line_width=1))
            layers.append(hv.Scatter(clean, kdims=["year"], vdims=["mag", *hover],
                                     label=f"ZTF {band} ({len(clean)})").opts(
                color=ZTF_COLOURS[band], size=4, tools=["hover"]))
        flagged = d[~d["clean"]]
        if show_flagged and len(flagged):
            layers.append(hv.Scatter(flagged, kdims=["year"], vdims=["mag", *hover],
                                     label=f"ZTF {band} flagged ({len(flagged)})").opts(
                color="#999999", marker="x", size=6, alpha=0.8, tools=["hover"]))
    if not layers:
        return _empty("Only flagged ZTF points; tick “show flagged points” to see them")
    return hv.Overlay(layers).opts(
        responsive=True, height=360, legend_position="right", invert_yaxis=True,
        xlabel="Observation time [yr, from ZTF MJD]", ylabel="ZTF magnitude [mag]",
        title="ZTF light curve", shared_axes=False,
    )
