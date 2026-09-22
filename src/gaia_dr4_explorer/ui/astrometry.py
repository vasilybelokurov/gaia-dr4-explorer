"""The astrometry tab: filters, the linked views, and the flag table."""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param

from gaia_dr4_explorer.products.astrometry import plots
from gaia_dr4_explorer.products.astrometry.normalize import CCD_NAMES

MATRIX_QUANTITIES = {
    "Used by AGIS": "used_by_agis_al",
    "Present / missing": "present",
    "Centroid uncertainty AL": "centroid_pos_error_al",
    "IPD uncertainty AL": "ipd_error_al",
    "CCD processing flags": "ccd_proc_flags",
    "Gate": "gates",
}


class AstrometryView(param.Parameterized):
    """Filterable views over one source's CCD-level astrometry."""

    show_used = param.Boolean(default=True, label="AGIS-used")
    show_rejected = param.Boolean(default=True, label="Rejected")
    ccds = param.ListSelector(default=list(CCD_NAMES), objects=list(CCD_NAMES),
                              label="Focal-plane strips")
    max_sigma = param.Number(default=0.0, bounds=(0.0, 5.0), step=0.01,
                             label="Max sigma_AL [mas] (0 = no cut)")
    matrix_quantity = param.Selector(default="used_by_agis_al",
                                     objects=list(MATRIX_QUANTITIES.values()))

    def __init__(self, context, payload, **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.frame = plots.to_frame(payload.table("ccd"))
        self._constant = self._find_constant_columns()

    # -------------------------------------------------------------- filtering

    def _find_constant_columns(self) -> dict[str, object]:
        """Columns with no spread: their filters can do nothing, so say so."""
        out = {}
        for name in ("multipeak", "blended", "used_by_agis_ac", "g_mag"):
            if name in self.frame.columns:
                values = pd.unique(self.frame[name].dropna())
                if len(values) == 1:
                    out[name] = values[0]
        return out

    def filtered(self) -> pd.DataFrame:
        frame = self.frame
        keep = np.zeros(len(frame), dtype=bool)
        if self.show_used:
            keep |= frame["used"].to_numpy()
        if self.show_rejected:
            keep |= ~frame["used"].to_numpy()
        frame = frame[keep]
        if self.ccds:
            frame = frame[frame["ccd_name"].isin(self.ccds)]
        if self.max_sigma > 0 and "centroid_pos_error_al" in frame.columns:
            sigma = frame["centroid_pos_error_al"].to_numpy()
            frame = frame[np.isfinite(sigma) & (sigma <= self.max_sigma)]
        return frame

    # ------------------------------------------------------------------ panes

    @param.depends("show_used", "show_rejected", "ccds", "max_sigma")
    def coverage(self):
        return _wrap(plots.coverage_timeline(self.filtered()))

    @param.depends("show_used", "show_rejected", "ccds", "max_sigma")
    def centroid(self):
        return _wrap(plots.centroid_vs_time(self.filtered()))

    @param.depends("show_used", "show_rejected", "ccds", "max_sigma")
    def geometry(self):
        frame = self.filtered()
        return pn.Column(
            _wrap(plots.scan_angle_vs_time(frame)),
            _wrap(plots.parallax_factor_vs_time(frame)),
        )

    @param.depends("show_used", "show_rejected", "ccds", "max_sigma")
    def uncertainties(self):
        return _wrap(plots.uncertainty_distributions(self.filtered()))

    @param.depends("matrix_quantity")
    def matrix(self):
        # The matrix always shows every sample: hiding rejected observations
        # here would defeat its purpose.
        return _wrap(plots.focal_plane_matrix(self.frame, self.matrix_quantity))

    @param.depends("show_used", "show_rejected", "ccds", "max_sigma")
    def flag_table(self):
        frame = self.filtered()
        columns = [
            c for c in (
                "transit_id_str", "ccd_name", "fov", "obs_time_jyear_tcb", "centroid_pos_al",
                "centroid_pos_error_al", "ipd_error_al", "scan_pos_angle",
                "used_by_agis_al", "used_by_agis_ac", "gates", "ccd_proc_flags",
                "transit_acq_flags", "transit_proc_flags", "multipeak", "blended", "g_class",
            ) if c in frame.columns
        ]
        return pn.widgets.Tabulator(
            frame[columns], pagination="local", page_size=25, height=520,
            sizing_mode="stretch_width", header_filters=True, disabled=True,
        )

    def controls(self) -> pn.Column:
        disabled_notes = [
            f"<code>{name}</code> is <code>{_short(value)}</code> for every observation of this "
            "source, so filtering on it can do nothing."
            for name, value in self._constant.items()
        ]
        items = [
            pn.pane.HTML("<b>Filters</b>"),
            pn.Param(
                self.param,
                parameters=["show_used", "show_rejected", "ccds", "max_sigma"],
                widgets={"ccds": {"type": pn.widgets.MultiChoice, "height": 120}},
                show_name=False,
            ),
        ]
        if disabled_notes:
            items.append(
                pn.pane.HTML(
                    '<div style="font-size:11px;color:#777;border-top:1px solid #eee;'
                    'padding-top:6px;">' + "<br>".join(disabled_notes) + "</div>"
                )
            )
        return pn.Column(*items, width=260)

    def panel(self) -> pn.Column:
        matrix_select = pn.widgets.Select(
            name="Cell quantity", options=MATRIX_QUANTITIES, value=self.matrix_quantity,
            width=240,
        )
        matrix_select.link(self, value="matrix_quantity")
        return pn.Column(
            pn.Tabs(
                ("Coverage", pn.Column(self.coverage)),
                ("Centroid AL", pn.Column(self.centroid)),
                ("Scan geometry", pn.Column(self.geometry)),
                ("Uncertainties", pn.Column(self.uncertainties)),
                ("Focal-plane matrix", pn.Column(matrix_select, self.matrix)),
                ("Flags", pn.Column(self.flag_table)),
                dynamic=True,
            ),
            sizing_mode="stretch_width",
        )


def _short(value) -> str:
    """Compact repr for a constant column's value.

    numpy scalars are not Python ``float``/``bool``, so test the numpy types
    too or a float32 magnitude prints all 8 of its meaningless digits.
    """
    if isinstance(value, (float, np.floating)):
        return f"{value:.4g}"
    if isinstance(value, (bool, np.bool_)):
        return "True" if value else "False"
    return str(value)


def _wrap(obj) -> pn.pane.HoloViews:
    return pn.pane.HoloViews(obj, sizing_mode="fixed")


hv.extension("bokeh")
