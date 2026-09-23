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
from gaia_dr4_explorer.products.astrometry.times import DR4_REFERENCE_EPOCH_JYEAR

#: Plots are sized by their container, never by a fixed pixel width: a hard
#: width clips the right-hand end of the axis in a narrower browser window.
#: Only the height is set here.

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
    """Axis label naming the Gaia field and its unit.

    ``centroid_pos_al`` becomes ``"centroid_pos_al [mas]"``.  The field name is
    kept deliberately, so every axis can be traced back to a published column.
    A dimensionless quantity says so rather than leaving the reader to guess
    whether the unit was simply forgotten.
    """
    spec = FIELDS.get(field)
    if spec is None:
        return fallback or field
    unit = spec.canonical_unit
    return f"{field} [{unit}]" if unit else f"{field} [dimensionless]"


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
        responsive=True, height=360, legend_position="right",
        title="Focal-plane coverage: every CCD observation",
        xlabel="Observation time [yr, TCB]", ylabel="Transit (ordered by time)",
    )


def centroid_vs_time(frame: pd.DataFrame, *, show_errors: bool = True) -> hv.Overlay:
    """Along-scan centroid against time, split by AGIS use."""
    layers = []
    # Rejected observations are drawn first so the AGIS-used majority lands on
    # top.  The 10 CCDs of one transit sit almost on top of each other, so
    # whichever series is drawn last hides the other: with the order reversed a
    # source that is 72% used renders as if it were entirely rejected.
    for status, colour, marker in (
        ("rejected", _REJECTED_COLOUR, "x"),
        ("used by AGIS", _USED_COLOUR, "o"),
    ):
        sub = frame[frame["status"] == status]
        if sub.empty:
            continue
        if show_errors and "centroid_pos_error_al" in sub.columns:
            layers.append(
                hv.ErrorBars(
                    sub, kdims=["obs_time_jyear_tcb"],
                    vdims=["centroid_pos_al", "centroid_pos_error_al"], label=status,
                ).opts(color=colour, alpha=0.3, line_width=1)
            )
        layers.append(
            hv.Points(
                sub,
                kdims=[
                    hv.Dimension("obs_time_jyear_tcb", label="Observation time [yr, TCB]"),
                    hv.Dimension("centroid_pos_al", label=axis_label("centroid_pos_al")),
                ],
                vdims=_hover(sub, exclude=("obs_time_jyear_tcb", "centroid_pos_al")),
                label=status,
            ).opts(
                color=colour, size=5, alpha=0.85, marker=marker,
                tools=["hover", "box_select"],
            )
        )
    if not layers:
        return _empty("No observations match the current filter")
    return hv.Overlay(layers).opts(
        responsive=True, height=360, legend_position="top_right",
        title="Along-scan centroid",
        # An Overlay inherits axis labels from its first element, which here is
        # an ErrorBars layer carrying bare column names.  State them.
        xlabel="Observation time [yr, TCB]",
        ylabel=axis_label("centroid_pos_al"),
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
        size=4, alpha=0.8, tools=["hover"], responsive=True, height=300,
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
        responsive=True, height=300, legend_position="top_right",
        title="Parallax factor AL (transit-level; missing transits marked, not imputed)",
        xlabel="Observation time [yr, TCB]",
        ylabel=axis_label("parallax_factor_al"),
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
                responsive=True, height=280, title=title, color=_USED_COLOUR,
            )
        )
    if "centroid_pos_error_al" in frame.columns:
        panels.append(
            hv.BoxWhisker(
                used, kdims=["ccd_name"], vdims=["centroid_pos_error_al"],
            ).opts(
                responsive=True, height=300, xlabel="Focal-plane strip",
                ylabel=axis_label("centroid_pos_error_al"),
                title="Centroid uncertainty by CCD (AGIS-used only)",
            )
        )
    if not panels:
        return _empty("No uncertainty data").opts(responsive=True)
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
        cmap=cmap, colorbar=True, tools=["hover"], responsive=True,
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
        ).opts(responsive=True)
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
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], responsive=True, height=280,
            xlabel="Observation time [yr, TCB]", ylabel="residual [mas]",
            title="Residual vs time",
        ),
        hv.Points(
            data, kdims=["scan_pos_angle", "residual"],
            vdims=_hover(data, exclude=("scan_pos_angle",)),
        ).opts(
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], responsive=True, height=280,
            xlabel=axis_label("scan_pos_angle"), ylabel="residual [mas]",
            title="Residual vs scan angle",
        ),
        hv.Points(
            data, kdims=["parallax_factor_al", "residual"],
            vdims=_hover(data, exclude=("parallax_factor_al",)),
        ).opts(
            size=4, alpha=0.8, color=_USED_COLOUR, tools=["hover"], responsive=True, height=280,
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
                responsive=True, height=280, color=_USED_COLOUR,
                xlabel="residual [mas]", ylabel="CCD observations",
                title="Residual distribution",
            )
        )
    panels.append(
        hv.BoxWhisker(data, kdims=["ccd_name"], vdims=["residual"]).opts(
            responsive=True, height=300, xlabel="Focal-plane strip", ylabel="residual [mas]",
            title="Residual by CCD",
        )
    )
    return hv.Layout(panels).cols(2)


def _empty(message: str) -> hv.Text:
    return hv.Text(0.5, 0.5, message).opts(
        xaxis=None, yaxis=None, responsive=True, height=200, color="#888888"
    )


# --------------------------------------------------------------- sky plane

#: Hover columns for the sky view: the published inputs of each point.
SKY_HOVER = HOVER_COLUMNS + ("calculated_pos_ac", "dra", "ddec")


def _segments(frame, cols) -> list:
    a, b, c, d = (frame[k].to_numpy(dtype="float64") for k in cols)
    ok = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    return [np.array([[a[i], b[i]], [c[i], d[i]]]) for i in np.flatnonzero(ok)]


#: Frame of a sky panel, in screen pixels. Fixed, so that the limits below
#: can give both axes the same scale in mas per pixel.
SKY_FRAME = (720, 480)


def equal_scale_limits(x, y, frame=SKY_FRAME, pad: float = 0.06):
    """Axis limits with equal mas per pixel on both axes, covering (x, y).

    Bokeh's own aspect constraint disables responsive sizing and can make a
    panel thousands of pixels wide for a fast-moving star, so the equal scale
    is imposed through the limits instead.

    Returns
    -------
    (xlim, ylim) or None when there is nothing finite to frame.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    x, y = x[ok], y[ok]
    cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
    scale = (1 + pad) * max(np.ptp(x) / frame[0], np.ptp(y) / frame[1], 0.1 / frame[1])
    hx, hy = 0.5 * scale * frame[0], 0.5 * scale * frame[1]
    return (cx - hx, cx + hx), (cy - hy, cy + hy)


def sky_epochs(
    epochs, track=None, *, show_rejected: bool = True, show_constraints: bool = True,
    show_errors: bool = True, proper_motion_removed: bool = False,
    model_label: str = "model",
) -> hv.Overlay:
    """Gaia epoch astrometry on the sky, with an optional model drawn over it.

    Parameters
    ----------
    epochs : DataFrame
        Output of :func:`skyplane.epoch_sky_positions`, with ``used`` and, for
        the constraint lines, ``x0, y0, x1, y1``.
    track : tuple of ndarray, optional
        ``(t_year, dra, ddec)`` model track, drawn last so it sits on the data.
    show_rejected : bool
        Draw observations with ``used_by_agis_al = False`` as grey crosses.
        They never set the axis range, so a far outlier cannot hide the rest.
    show_constraints : bool
        Draw the line each observation constrains, perpendicular to the scan.
    show_errors : bool
        Draw ±1σ along-scan error bars.

    Notes
    -----
    Each point is (w, z) rotated to the sky: w = ``centroid_pos_al`` measured,
    z = ``calculated_pos_ac`` predicted by AGIS. Only the along-scan position
    is a measurement.
    """
    used = epochs[epochs["used"]] if len(epochs) else epochs
    rejected = epochs[~epochs["used"]] if len(epochs) else epochs
    layers = []
    if show_constraints and {"x0", "y0", "x1", "y1"} <= set(epochs.columns) and len(used):
        layers.append(hv.Path(_segments(used, ("x0", "y0", "x1", "y1"))).opts(
            color="#c8c8c8", line_width=1, alpha=0.8))
    if show_errors and len(used):
        layers.append(hv.Path(_segments(used, ("ex0", "ey0", "ex1", "ey1"))).opts(
            color="#555555", line_width=1, alpha=0.8))
    if show_rejected and len(rejected):
        layers.append(hv.Points(
            rejected, kdims=["dra", "ddec"],
            vdims=[c for c in SKY_HOVER if c in rejected.columns and c not in ("dra", "ddec")],
            label=f"rejected by AGIS ({len(rejected)})",
        ).opts(color="#999999", marker="x", size=6, alpha=0.7, tools=["hover"]))
    if len(used):
        layers.append(hv.Points(
            used, kdims=["dra", "ddec"],
            vdims=[c for c in SKY_HOVER if c in used.columns and c not in ("dra", "ddec")],
            label=f"Gaia measurements ({len(used)})",
        ).opts(
            color="obs_time_jyear_tcb", cmap="viridis", colorbar=True, size=5,
            alpha=0.9, tools=["hover"], clabel="observation time [yr, TCB]",
        ))
    if track is not None:
        _, tdra, tddec = track
        layers.append(hv.Curve(
            (tdra, tddec), kdims=["dra"], vdims=["ddec"], label=model_label,
        ).opts(color="#d62728", line_width=1.6, alpha=0.9))
    if not layers:
        return _empty("No observations to place on the sky")

    xs = [used["dra"].to_numpy()] if len(used) else []
    ys = [used["ddec"].to_numpy()] if len(used) else []
    if track is not None:
        xs.append(np.asarray(track[1]))
        ys.append(np.asarray(track[2]))
    opts = dict(
        frame_width=SKY_FRAME[0], frame_height=SKY_FRAME[1],
        legend_position="right", title="",
        xlabel="Δα* [mas]  (from the reference point ra0, dec0)", ylabel="Δδ [mas]",
        invert_xaxis=True,   # RA increases to the left, as on the sky
        # Each panel keeps its own ranges: HoloViews otherwise links every plot
        # with the same dimension names, so the proper-motion-removed panel
        # inherited the as-observed ranges and its data shrank to a dot.
        shared_axes=False,
    )
    limits = equal_scale_limits(np.concatenate(xs), np.concatenate(ys)) if xs else None
    if limits:
        # invert_xaxis flips the drawn direction; the limits stay ascending.
        opts["xlim"], opts["ylim"] = limits
    return hv.Overlay(layers).opts(**opts)


def sky_offsets_vs_time(epochs, track=None, *, model_label: str = "model") -> hv.Layout:
    """Δα* and Δδ of the used observations against time, model drawn over them."""
    used = epochs[epochs["used"]] if len(epochs) else epochs
    panels = []
    for k, (col, label) in enumerate((("dra", "Δα* [mas]"), ("ddec", "Δδ [mas]"))):
        if col not in used or not len(used):
            continue
        layer = hv.Points(
            used, kdims=["obs_time_jyear_tcb", col],
            vdims=[c for c in SKY_HOVER if c in used.columns and c != col],
            label="Gaia measurements",
        ).opts(color=_USED_COLOUR, size=4, alpha=0.85, tools=["hover"])
        if track is not None:
            t, tdra, tddec = track
            layer = layer * hv.Curve(
                (t + DR4_REFERENCE_EPOCH_JYEAR, (tdra, tddec)[k]),
                kdims=["obs_time_jyear_tcb"], vdims=[col], label=model_label,
            ).opts(color="#d62728", line_width=1.4)
        panels.append(layer.opts(
            responsive=True, height=260, legend_position="top_left",
            xlabel="Observation time [yr, TCB]", ylabel=label, title=f"{label} against time",
        ))
    if not panels:
        return _empty("No sky-plane positions to show")
    return hv.Layout(panels).cols(1)
