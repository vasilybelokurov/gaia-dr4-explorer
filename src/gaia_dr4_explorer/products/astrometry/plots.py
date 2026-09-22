"""Plot builders for epoch astrometry.

Every function here takes an already-loaded table and returns a HoloViews
object.  None of them perform I/O, and none of them convert units: axis labels
are taken from the schema registry so that what is drawn is what was published.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd

from gaia_dr4_explorer.products.astrometry.normalize import CCD_NAMES
from gaia_dr4_explorer.products.astrometry.schema import FIELDS

#: Columns offered on hover, in order, where present.
HOVER_COLUMNS = (
    "transit_id_str",
    "ccd_name",
    "fov",
    "obs_time_jyear_tcb",
    "centroid_pos_al",
    "centroid_pos_error_al",
    "scan_pos_angle",
    "used_by_agis_al",
    "ipd_error_al",
    "gates",
    "ccd_proc_flags",
)

_USED_COLOUR = "#1f77b4"
_REJECTED_COLOUR = "#d62728"


def axis_label(field: str, fallback: str | None = None) -> str:
    """``"centroid_pos_al [mas]"`` -- never a bare number."""
    spec = FIELDS.get(field)
    if spec is None:
        return fallback or field
    unit = spec.canonical_unit
    return f"{field} [{unit}]" if unit else field


def to_frame(ccd) -> pd.DataFrame:
    """Flatten a normalized CCD table to a plotting frame, masks become NaN."""
    data = {}
    for name in ccd.colnames:
        col = ccd[name]
        if getattr(col, "mask", None) is not None:
            if np.issubdtype(col.dtype, np.floating):
                data[name] = np.ma.filled(col, np.nan)
            elif np.issubdtype(col.dtype, np.bool_):
                data[name] = np.ma.filled(col, False)
            else:
                data[name] = np.ma.filled(col, 0)
        else:
            data[name] = np.asarray(col)
    frame = pd.DataFrame(data)
    frame["used"] = frame.get("used_by_agis_al", pd.Series(False, index=frame.index)).astype(bool)
    frame["status"] = np.where(frame["used"], "used by AGIS", "rejected")
    # Bokeh carries numeric columns as float64, whose 53-bit mantissa cannot
    # hold a 19-digit Gaia identifier: a tooltip would show a corrupted number.
    # Keep a string copy for display and leave the integer column untouched.
    for name in ("transit_id", "source_id", "solution_id"):
        if name in frame.columns:
            frame[f"{name}_str"] = frame[name].astype("int64").astype(str)
    return frame


def _hover(frame: pd.DataFrame, *, exclude: tuple[str, ...] = ()) -> list[str]:
    """Hover columns present in *frame*, never repeating a key dimension."""
    return [c for c in HOVER_COLUMNS if c in frame.columns and c not in exclude]


def coverage_timeline(frame: pd.DataFrame, *, colour_by: str = "ccd_name") -> hv.Overlay:
    """Observation time against transit, one point per CCD sample."""
    kdims = [
        hv.Dimension("obs_time_jyear_tcb", label="Observation time [yr, TCB]"),
        hv.Dimension("transit_index", label="Transit (ordered by time)"),
    ]
    frame = frame.copy()
    order = {t: i for i, t in enumerate(sorted(frame["transit_id"].unique()))}
    frame["transit_index"] = frame["transit_id"].map(order)
    keys = ("obs_time_jyear_tcb", "transit_index")
    vdims = _hover(frame, exclude=keys) + [c for c in (colour_by, "status")
                                           if c not in _hover(frame, exclude=keys)]
    points = hv.Points(frame, kdims=list(kdims), vdims=vdims)
    return points.opts(
        color=colour_by, cmap="Category10", size=4, alpha=0.8, tools=["hover", "box_select"],
        width=820, height=360, legend_position="right",
        title="Focal-plane coverage: every CCD observation",
    )


def centroid_vs_time(frame: pd.DataFrame, *, show_errors: bool = True) -> hv.Overlay:
    """Along-scan centroid against time, split by AGIS use."""
    layers = []
    for status, colour in (("used by AGIS", _USED_COLOUR), ("rejected", _REJECTED_COLOUR)):
        sub = frame[frame["status"] == status]
        if sub.empty:
            continue
        pts = hv.Points(
            sub,
            kdims=[
                hv.Dimension("obs_time_jyear_tcb", label="Observation time [yr, TCB]"),
                hv.Dimension("centroid_pos_al", label=axis_label("centroid_pos_al")),
            ],
            vdims=_hover(sub, exclude=("obs_time_jyear_tcb", "centroid_pos_al")),
            label=status,
        ).opts(color=colour, size=5, alpha=0.85, tools=["hover", "box_select"])
        layers.append(pts)
        if show_errors and "centroid_pos_error_al" in sub.columns:
            bars = hv.ErrorBars(
                sub, kdims=["obs_time_jyear_tcb"],
                vdims=["centroid_pos_al", "centroid_pos_error_al"], label=status,
            ).opts(color=colour, alpha=0.35, line_width=1)
            layers.append(bars)
    if not layers:
        return _empty("No observations match the current filter")
    return hv.Overlay(layers).opts(
        width=820, height=360, legend_position="top_right",
        title="Along-scan centroid",
    )


def scan_angle_vs_time(frame: pd.DataFrame) -> hv.Points:
    """Scan position angle against time.  The published convention is unchanged."""
    return hv.Points(
        frame,
        kdims=[
            hv.Dimension("obs_time_jyear_tcb", label="Observation time [yr, TCB]"),
            hv.Dimension("scan_pos_angle", label=axis_label("scan_pos_angle")),
        ],
        vdims=_hover(frame, exclude=("obs_time_jyear_tcb", "scan_pos_angle")),
    ).opts(
        color="status", cmap={"used by AGIS": _USED_COLOUR, "rejected": _REJECTED_COLOUR},
        size=4, alpha=0.8, tools=["hover"], width=820, height=300,
        title="Scan position angle (published convention, not unwrapped)",
    )


def parallax_factor_vs_time(frame: pd.DataFrame) -> hv.Overlay:
    """AL parallax factor against time, with missing transits marked."""
    good = frame[np.isfinite(frame["parallax_factor_al"])]
    missing = frame[~np.isfinite(frame["parallax_factor_al"])]
    pts = hv.Points(
        good,
        kdims=[
            hv.Dimension("obs_time_jyear_tcb", label="Observation time [yr, TCB]"),
            hv.Dimension("parallax_factor_al", label=axis_label("parallax_factor_al")),
        ],
        vdims=_hover(good, exclude=("obs_time_jyear_tcb", "parallax_factor_al")),
        label="measured",
    ).opts(color=_USED_COLOUR, size=4, alpha=0.8, tools=["hover"])
    layers = [pts]
    if not missing.empty:
        # Missing values are shown on their own band, never imputed to zero.
        band = hv.Scatter(
            (missing["obs_time_jyear_tcb"], np.full(len(missing), np.nan)), label="missing",
        )
        floor = float(np.nanmin(good["parallax_factor_al"])) if not good.empty else -1.0
        band = hv.Scatter(
            (missing["obs_time_jyear_tcb"], np.full(len(missing), floor - 0.1)), label="missing",
        ).opts(color="#999999", marker="x", size=6)
        layers.append(band)
    return hv.Overlay(layers).opts(
        width=820, height=300, legend_position="top_right",
        title="Parallax factor AL (transit-level; missing transits marked, not imputed)",
    )


def uncertainty_distributions(frame: pd.DataFrame) -> hv.Layout:
    """Centroid and IPD uncertainty distributions, and sigma by CCD."""
    used = frame[frame["used"]]
    panels = []
    for field, title in (
        ("centroid_pos_error_al", "Centroid uncertainty AL"),
        ("ipd_error_al", "IPD uncertainty AL"),
    ):
        if field not in frame.columns:
            continue
        values = used[field].to_numpy()
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        freq, edges = np.histogram(values, bins=40)
        panels.append(
            hv.Histogram((edges, freq)).opts(
                xlabel=axis_label(field), ylabel="CCD observations",
                width=400, height=280, title=title, color=_USED_COLOUR,
            )
        )
    if "centroid_pos_error_al" in frame.columns:
        panels.append(
            hv.BoxWhisker(
                used, kdims=["ccd_name"], vdims=["centroid_pos_error_al"],
            ).opts(
                width=820, height=300, xlabel="Focal-plane strip",
                ylabel=axis_label("centroid_pos_error_al"),
                title="Centroid uncertainty by CCD (AGIS-used only)",
            )
        )
    if not panels:
        return _empty("No uncertainty data").opts(width=820)
    return hv.Layout(panels).cols(2)


def focal_plane_matrix(frame: pd.DataFrame, quantity: str = "used_by_agis_al") -> hv.HeatMap:
    """Transit by focal-plane strip, coloured by the chosen quantity.

    Rows are transits, columns are SM and AF1..AF9.  This is the compact view of
    which samples exist and which AGIS kept.
    """
    if quantity not in frame.columns and quantity != "present":
        return _empty(f"{quantity!r} is not available for this source")
    order = {t: i for i, t in enumerate(sorted(frame["transit_id"].unique()))}
    data = frame.copy()
    data["transit_index"] = data["transit_id"].map(order)
    if quantity == "present":
        data["value"] = np.isfinite(data["centroid_pos_al"]).astype(float)
        label, cmap = "present (1) / missing (0)", "Greys"
    elif quantity == "used_by_agis_al":
        data["value"] = data["used"].astype(float)
        label, cmap = "used by AGIS (1) / rejected (0)", "RdYlBu"
    else:
        data["value"] = pd.to_numeric(data[quantity], errors="coerce").astype(float)
        label, cmap = axis_label(quantity), "viridis"
    # The kdims must not also appear as vdims: HoloViews grids every vdim and
    # would fail to find the column it has already consumed as a key.
    kdims = ["ccd_name", "transit_index"]
    vdims = ["value"] + [c for c in _hover(data) if c not in kdims]
    present = [n for n in CCD_NAMES if n in set(data["ccd_name"])]
    data["ccd_name"] = pd.Categorical(data["ccd_name"], categories=present, ordered=True)
    data = data.sort_values(["transit_index", "ccd_name"])
    heat = hv.HeatMap(data, kdims=kdims, vdims=vdims)
    return heat.opts(
        cmap=cmap, colorbar=True, tools=["hover"], width=560,
        height=max(320, min(900, 8 * len(order))),
        xlabel="Focal-plane strip", ylabel="Transit (ordered by time)",
        clabel=label, title=f"Transit x focal plane: {label}",
    )


def residual_plots(result, frame: pd.DataFrame) -> hv.Layout:
    """Residual diagnostics for a fitted source update."""
    residuals = np.asarray(result.residuals, dtype="float64")
    used = frame[frame["used"]].reset_index(drop=True)
    if len(used) != residuals.size:
        return _empty(
            f"Cannot align {residuals.size} residuals with {len(used)} AGIS-used rows; "
            "the fit and the displayed selection disagree"
        ).opts(width=820)
    data = used.copy()
    data["residual"] = residuals
    sigma = data["centroid_pos_error_al"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        data["normalized_residual"] = residuals / sigma

    panels = [
        hv.Points(
            data, kdims=["obs_time_jyear_tcb", "residual"],
            vdims=_hover(data, exclude=("obs_time_jyear_tcb",)),
        ).opts(
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], width=410, height=280,
            xlabel="Observation time [yr, TCB]", ylabel="residual [mas]",
            title="Residual vs time",
        ),
        hv.Points(
            data, kdims=["scan_pos_angle", "residual"],
            vdims=_hover(data, exclude=("scan_pos_angle",)),
        ).opts(
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], width=410, height=280,
            xlabel=axis_label("scan_pos_angle"), ylabel="residual [mas]",
            title="Residual vs scan angle",
        ),
        hv.Points(
            data, kdims=["parallax_factor_al", "residual"],
            vdims=_hover(data, exclude=("parallax_factor_al",)),
        ).opts(
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], width=410, height=280,
            xlabel=axis_label("parallax_factor_al"), ylabel="residual [mas]",
            title="Residual vs parallax factor",
        ),
    ]
    finite = data["residual"].to_numpy()
    finite = finite[np.isfinite(finite)]
    if finite.size:
        freq, edges = np.histogram(finite, bins=40)
        panels.append(
            hv.Histogram((edges, freq)).opts(
                width=410, height=280, color=_USED_COLOUR,
                xlabel="residual [mas]", ylabel="CCD observations",
                title="Residual distribution",
            )
        )
    panels.append(
        hv.BoxWhisker(data, kdims=["ccd_name"], vdims=["residual"]).opts(
            width=820, height=300, xlabel="Focal-plane strip", ylabel="residual [mas]",
            title="Residual by CCD",
        )
    )
    return hv.Layout(panels).cols(2)


def _empty(message: str) -> hv.Text:
    return hv.Text(0.5, 0.5, message).opts(
        xaxis=None, yaxis=None, width=820, height=200, color="#888888"
    )
