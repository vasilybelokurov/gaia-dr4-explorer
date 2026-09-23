"""The sky-plane tab.

Shows Gaia's epoch astrometry on the sky first -- each CCD observation placed
from its published local-plane coordinates -- and draws a fitted model over it
only when one is supplied.
"""

from __future__ import annotations

import numpy as np
import panel as pn
import param

from gaia_dr4_explorer.products.astrometry import plots, skyplane
from gaia_dr4_explorer.products.astrometry.times import DR4_REFERENCE_EPOCH_JYEAR
from gaia_dr4_explorer.ui.components import caveat

ONE_DIMENSIONAL_NOTE = (
    "Each point is one CCD observation, placed on the sky from Gaia's own "
    "published local-plane coordinates: the along-scan position "
    "<code>centroid_pos_al</code> is <b>measured</b>; the across-scan position "
    "<code>calculated_pos_ac</code> is <b>predicted by the AGIS solution</b>, "
    "because Gaia measures only one coordinate per observation and the measured "
    "across-scan centroid is not published in this release. The dark bars are "
    "±1σ along the scan; the grey lines, perpendicular to it, are the locus each "
    "observation actually constrains. Spread along a grey line is unmeasured."
)


#: Legends sit under side-by-side panels, so they must fit within one frame;
#: the model's full description is in the text above the plots.
LEGEND_MODEL_LABEL = "model"

#: Space between side-by-side panels, so a colour bar never meets its neighbour.
PANEL_GAP = (0, 40, 0, 0)


def _heading(text: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f"<div style='font-size:13px;font-weight:600;margin:14px 0 0 0'>{text}</div>")


class SkyPlaneView(param.Parameterized):
    """Epoch astrometry on the sky for one source, with an optional model."""

    show_model = param.Boolean(default=True, label="Overplot the model track")
    show_constraints = param.Boolean(default=True, label="Show 1-D constraint lines")
    show_errors = param.Boolean(default=True, label="Show along-scan ±1σ bars")
    show_rejected = param.Boolean(default=True, label="Show observations rejected by AGIS")
    constraint_length = param.Number(
        default=0.0, bounds=(0.0, 50.0), step=0.5,
        label="Constraint line half-length [mas] (0 = auto)",
    )

    def __init__(self, context, payload, model: skyplane.SkyModel | None = None,
                 *, model_label: str = "", model_note: str = "", **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.model = model
        self.model_label = model_label or "model"
        self.model_note = model_note
        frame = plots.to_frame(payload.table("ccd"))
        # Every published row is kept, rejected ones included (invariant 5);
        # only rows with no coordinates at all cannot be placed.
        placeable = np.isfinite(frame["centroid_pos_al"]) & np.isfinite(
            frame["calculated_pos_ac"]) & np.isfinite(frame["scan_pos_angle"])
        self.n_unplaceable = int((~placeable).sum())
        self.base = skyplane.epoch_sky_positions(frame[placeable].reset_index(drop=True))

    # ------------------------------------------------------------- derived

    def _epochs(self, *, pm_removed: bool = False):
        e = self.base.copy()
        if pm_removed and self.model is not None:
            t = e["relative_time_year"].to_numpy(dtype="float64")
            shift_x, shift_y = self.model.pmra_star * t, self.model.pmdec * t
            for a, b in (("dra", "ddec"), ("ex0", "ey0"), ("ex1", "ey1")):
                e[a] = e[a] - shift_x
                e[b] = e[b] - shift_y
        used = e[e["used"]]
        span = max(np.ptp(used["dra"]), np.ptp(used["ddec"])) if len(used) else 1.0
        half = self.constraint_length or float(max(0.03 * span, 0.05))
        x0, y0, x1, y1 = skyplane.constraint_segments(
            e["dra"], e["ddec"], e["scan_pos_angle"], half)
        e["x0"], e["y0"], e["x1"], e["y1"] = x0, y0, x1, y1
        return e

    def _track(self, *, pm_removed: bool = False):
        if self.model is None or not self.show_model:
            return None
        model = self.model.without_proper_motion() if pm_removed else self.model
        t = self.base["relative_time_year"].to_numpy(dtype="float64")
        t = t[np.isfinite(t)]
        if not t.size:
            return None
        t0, t1 = float(t.min()), float(t.max())
        pad = 0.02 * (t1 - t0)
        return skyplane.smooth_track(
            model, t0 - pad, t1 + pad,
            float(self.base["ra0"].iloc[0]), float(self.base["dec0"].iloc[0]),
            reference_epoch_jyear=DR4_REFERENCE_EPOCH_JYEAR,
        )

    # --------------------------------------------------------------- views

    def _sky(self, *, pm_removed: bool):
        return pn.pane.HoloViews(
            plots.sky_epochs(
                self._epochs(pm_removed=pm_removed), self._track(pm_removed=pm_removed),
                show_rejected=self.show_rejected, show_constraints=self.show_constraints,
                show_errors=self.show_errors, proper_motion_removed=pm_removed,
                model_label=LEGEND_MODEL_LABEL,
                # One colour bar, on the right-hand panel: the panels share it.
                colorbar=pm_removed,
            ),
            # No stretch sizing: it overrides the fixed frame the plot needs
            # for equal mas per pixel, and collapsed panel 3 in the browser.
        )

    @param.depends("show_model", "show_constraints", "show_errors", "show_rejected",
                   "constraint_length")
    def sky(self):
        """Panel 1: measured and modelled positions on the sky, as observed."""
        return self._sky(pm_removed=False)

    @param.depends("show_model", "show_constraints", "show_errors", "show_rejected",
                   "constraint_length")
    def sky_pm_removed(self):
        """Panel 2: the same, with the model's proper motion taken out."""
        if self.model is None:
            return pn.pane.HTML(
                "<div style='font-size:12px;color:#666;padding:12px 0'>Removing the "
                "proper motion needs a model: fit it with the button above.</div>")
        return self._sky(pm_removed=True)

    @param.depends("show_model")
    def components(self):
        """Panel 3: Δα* and Δδ against time, measured and modelled."""
        return pn.Row(*[
            pn.pane.HoloViews(p, margin=PANEL_GAP) for p in plots.sky_offsets_vs_time(
                self._epochs(), self._track(), model_label=LEGEND_MODEL_LABEL)
        ])

    def controls(self) -> pn.Row:
        """A horizontal strip, so the plots below can use the full width."""
        names = ["show_constraints", "show_errors", "show_rejected", "constraint_length"]
        if self.model is not None:
            names = ["show_model", *names]
        return pn.Row(
            pn.pane.HTML("<b>Sky plane</b>", margin=(12, 12, 0, 0)),
            pn.Param(self.param, parameters=names, show_name=False,
                     default_layout=pn.Row),
            sizing_mode="stretch_width",
        )

    def panel(self, *, extra=None) -> pn.Column:
        used = int(self.base["used"].sum())
        rejected = int((~self.base["used"]).sum())
        counts = (
            f"{used} observations used by AGIS, {rejected} rejected"
            + (f", {self.n_unplaceable} without coordinates" if self.n_unplaceable else "")
            + f". Offsets are from the reference point (ra0, dec0); model time is "
            f"measured from J{DR4_REFERENCE_EPOCH_JYEAR}."
        )
        if self.model is not None:
            m = self.model
            model_html = (
                f"<b>{self.model_label}</b> (red): ϖ = {m.parallax:.4f} mas &nbsp;·&nbsp; "
                f"μα* = {m.pmra_star:.3f} mas/yr &nbsp;·&nbsp; μδ = {m.pmdec:.3f} mas/yr"
                f"<br><span style='color:#777;font-size:11px'>{self.model_note}</span>"
            )
        else:
            model_html = "No model overplotted."
        return pn.Column(
            caveat(ONE_DIMENSIONAL_NOTE),
            pn.pane.HTML(
                f"<div style='font-size:12px;color:#444;margin-top:6px'>{model_html}"
                f"<br><span style='color:#777;font-size:11px'>{counts}</span></div>"
            ),
            *([extra] if extra is not None else []),
            pn.Row(
                pn.Column(_heading("1. On the sky — measured and modelled"), self.sky,
                          margin=PANEL_GAP),
                pn.Column(_heading("2. On the sky — proper motion removed"),
                          self.sky_pm_removed),
            ),
            _heading("3. Δα* and Δδ against time"),
            pn.Column(self.components),
            sizing_mode="stretch_width",
        )

    def layout(self, *, extra=None) -> pn.Column:
        return pn.Column(self.controls(), self.panel(extra=extra), sizing_mode="stretch_width")
