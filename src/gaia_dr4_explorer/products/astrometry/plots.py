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


def sky_track(
    track_t, track_dra, track_ddec, epochs=None, *, show_constraints: bool = True,
    subtract_proper_motion: bool = False, title: str = "Reconstructed sky track",
) -> hv.Overlay:
    """The model path on the sky, with each 1-D measurement shown as such.

    Parameters
    ----------
    track_t, track_dra, track_ddec : ndarray
        Densely sampled model track, years from the reference epoch and mas.
    epochs : DataFrame, optional
        Per-epoch frame with ``dra``, ``ddec``, ``x0, y0, x1, y1`` constraint
        endpoints, ``obs_time_jyear_tcb`` and the usual hover columns.
    show_constraints : bool
        Draw the perpendicular line each measurement actually constrains.
    subtract_proper_motion : bool
        Show the parallax alone, with the linear motion removed.

    Notes
    -----
    Gaia measures one coordinate per observation. The perpendicular position of
    every plotted point comes from the model, not from data; the short lines
    are the locus the measurement really constrains.
    """
    layers = [
        hv.Curve(
            (track_dra, track_ddec), kdims=["dra"], vdims=["ddec"], label="model track",
        ).opts(color="#444444", line_width=1.4, alpha=0.9)
    ]
    if epochs is not None and len(epochs):
        if show_constraints and {"x0", "y0", "x1", "y1"} <= set(epochs.columns):
            segs = [
                np.column_stack([[r.x0, r.x1], [r.y0, r.y1]])
                for r in epochs.itertuples()
            ]
            layers.append(
                hv.Path(segs).opts(color="#bbbbbb", line_width=1, alpha=0.7)
            )
        layers.append(
            hv.Points(
                epochs, kdims=["dra", "ddec"],
                vdims=[c for c in _hover(epochs) if c in epochs.columns]
                + (["obs_time_jyear_tcb"] if "obs_time_jyear_tcb" in epochs else []),
                label="measurement",
            ).opts(
                color="obs_time_jyear_tcb" if "obs_time_jyear_tcb" in epochs else _USED_COLOUR,
                cmap="viridis", colorbar=True, size=5, alpha=0.9,
                tools=["hover"], clabel="observation time [yr, TCB]",
            )
        )
    label = (
        "parallax only, proper motion removed" if subtract_proper_motion
        else "parallax and proper motion"
    )
    return hv.Overlay(layers).opts(
        responsive=True, height=520, legend_position="top_left",
        xlabel="Δα* [mas]  (offset from the transit reference point)",
        ylabel="Δδ [mas]",
        title=f"{title} — {label}",
        invert_xaxis=True,   # RA increases to the left, as on the sky
    )


def sky_offsets_vs_time(epochs) -> hv.Layout:
    """The two tangent-plane components against time, model and measurement."""
    panels = []
    for col, label in (("dra", "Δα* [mas]"), ("ddec", "Δδ [mas]")):
        if col not in epochs:
            continue
        panels.append(
            hv.Points(
                epochs, kdims=["obs_time_jyear_tcb", col],
                vdims=[c for c in _hover(epochs) if c in epochs.columns],
            ).opts(
                responsive=True, height=260, color=_USED_COLOUR, size=4,
                alpha=0.85, tools=["hover"],
                xlabel="Observation time [yr, TCB]", ylabel=label,
                title=f"{label} against time",
            )
        )
    if not panels:
        return _empty("No sky-plane positions to show")
    return hv.Layout(panels).cols(1)
