"""The sky-plane tab.

Shows where the fitted model puts the source on the sky over the mission, with
each epoch drawn as the one-dimensional constraint it actually is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import panel as pn
import param

from gaia_dr4_explorer.products.astrometry import plots, skyplane
from gaia_dr4_explorer.products.astrometry.times import (
    DR4_REFERENCE_EPOCH_JYEAR,
    tcb_ns_to_time,
)
from gaia_dr4_explorer.ui.components import caveat

ONE_DIMENSIONAL_NOTE = (
    "Gaia measures <b>one coordinate per observation</b>: the along-scan "
    "position. The across-scan centroid is not published in this release at "
    "all. So there is no measured (α, δ) for an epoch — the perpendicular "
    "position of every point below comes from the fitted model, and the short "
    "grey lines are the locus each measurement genuinely constrains. Treat the "
    "spread along a grey line as unmeasured, not as agreement."
)


class SkyPlaneView(param.Parameterized):
    """Reconstructed sky track for one source."""

    show_constraints = param.Boolean(default=True, label="Show 1-D constraint lines")
    subtract_proper_motion = param.Boolean(
        default=False, label="Remove proper motion (parallax only)"
    )
    constraint_length = param.Number(
        default=0.0, bounds=(0.0, 50.0), step=0.5,
        label="Constraint line half-length [mas] (0 = auto)",
    )

    def __init__(self, context, payload, result, **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.result = result
        self.model = skyplane.SkyModel.from_fit(result)
        self._build()

    def _build(self) -> None:
        ccd = self.payload.table("ccd")
        f = lambda n: np.asarray(np.ma.filled(ccd[n], np.nan), dtype="float64")  # noqa: E731
        used = np.asarray(np.ma.filled(ccd["used_by_agis_al"], False), dtype=bool)
        ns = np.asarray(np.ma.filled(ccd["obs_time_tcb"], 0), dtype="int64")[used]

        self.times = tcb_ns_to_time(ns)
        self.theta = f("scan_pos_angle")[used]
        self.t_year = f("relative_time_year")[used]
        self.ra0 = f("ra0")[used]
        self.dec0 = f("dec0")[used]
        self.residuals = np.asarray(self.result.residuals, dtype="float64")

        plx_dra, plx_ddec = skyplane.parallax_displacement(self.times, self.ra0, self.dec0)
        self.plx = (plx_dra, plx_ddec)
        self.jyear = f("obs_time_jyear_tcb")[used]
        self.frame_base = pd.DataFrame({
            "obs_time_jyear_tcb": self.jyear,
            "transit_id_str": np.asarray(ccd["transit_id"])[used].astype("int64").astype(str),
            "ccd_name": np.asarray(ccd["ccd_name"])[used],
            "scan_pos_angle": self.theta,
            "centroid_pos_al": f("centroid_pos_al")[used],
            "centroid_pos_error_al": f("centroid_pos_error_al")[used],
        })

    def _model(self) -> skyplane.SkyModel:
        if not self.subtract_proper_motion:
            return self.model
        m = self.model
        return skyplane.SkyModel(m.delta_alpha_star, m.delta_delta, m.parallax, 0.0, 0.0)

    def _epochs(self) -> pd.DataFrame:
        model = self._model()
        t = self.t_year if not self.subtract_proper_motion else self.t_year
        dra, ddec = skyplane.model_offsets(model, t, *self.plx)
        if self.residuals.size != dra.size:
            return pd.DataFrame()
        pdra, pddec = skyplane.measured_positions(dra, ddec, self.residuals, self.theta)
        frame = self.frame_base.copy()
        frame["dra"], frame["ddec"] = pdra, pddec
        half = self.constraint_length or self._auto_length(pdra, pddec)
        x0, y0, x1, y1 = skyplane.constraint_segments(pdra, pddec, self.theta, half)
        frame["x0"], frame["y0"], frame["x1"], frame["y1"] = x0, y0, x1, y1
        return frame

    @staticmethod
    def _auto_length(dra, ddec) -> float:
        span = max(np.ptp(dra), np.ptp(ddec))
        return float(max(span * 0.03, 0.05))

    def _track(self):
        model = self._model()
        t0, t1 = float(np.nanmin(self.t_year)), float(np.nanmax(self.t_year))
        pad = 0.02 * (t1 - t0)
        return skyplane.smooth_track(
            model, t0 - pad, t1 + pad, float(self.ra0[0]), float(self.dec0[0]),
            reference_epoch_jyear=DR4_REFERENCE_EPOCH_JYEAR,
        )

    @param.depends("show_constraints", "subtract_proper_motion", "constraint_length")
    def track(self):
        epochs = self._epochs()
        t, dra, ddec = self._track()
        return pn.pane.HoloViews(
            plots.sky_track(
                t, dra, ddec, epochs,
                show_constraints=self.show_constraints,
                subtract_proper_motion=self.subtract_proper_motion,
            ),
            sizing_mode="stretch_width",
        )

    @param.depends("subtract_proper_motion")
    def components(self):
        return pn.pane.HoloViews(
            plots.sky_offsets_vs_time(self._epochs()), sizing_mode="stretch_width"
        )

    def controls(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML("<b>Sky plane</b>"),
            pn.Param(
                self.param,
                parameters=["show_constraints", "subtract_proper_motion",
                            "constraint_length"],
                show_name=False,
            ),
            width=300, sizing_mode="fixed", margin=(0, 18, 0, 0),
        )

    def panel(self) -> pn.Column:
        p = self.result.as_dict()
        summary = (
            f"ϖ = {p['parallax'][0]:.4f} ± {p['parallax'][1]:.4f} mas &nbsp;·&nbsp; "
            f"μα* = {p['pmra_star'][0]:.3f} mas/yr &nbsp;·&nbsp; "
            f"μδ = {p['pmdec'][0]:.3f} mas/yr"
        )
        return pn.Column(
            caveat(ONE_DIMENSIONAL_NOTE),
            pn.pane.HTML(
                f"<div style='font-size:12px;color:#444;margin-top:6px'>Model: {summary}"
                f"<br><span style='color:#777;font-size:11px'>Offsets are from each "
                f"transit's reference point (ra0, dec0); time is measured from "
                f"J{DR4_REFERENCE_EPOCH_JYEAR}. {self.result.caveat}</span></div>"
            ),
            pn.Tabs(
                ("Sky track", pn.Column(self.track, sizing_mode="stretch_width")),
                ("Components vs time", pn.Column(self.components, sizing_mode="stretch_width")),
                dynamic=True, sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


def needs_fit_panel(message: str) -> pn.Column:
    return pn.Column(
        pn.pane.HTML(
            "<h3 style='margin:8px 0 4px 0'>Sky track needs a fit</h3>"
            f"<div style='font-size:12px;color:#555;max-width:720px'>{message}</div>"
        ),
        sizing_mode="stretch_width",
    )
