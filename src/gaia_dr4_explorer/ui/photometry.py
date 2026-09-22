"""The photometry tab."""

from __future__ import annotations

import panel as pn
import param

from gaia_dr4_explorer.products.photometry import plots
from gaia_dr4_explorer.ui.components import caveat


class PhotometryView(param.Parameterized):
    """Light curves for one source, with the release stated everywhere."""

    bands = param.ListSelector(default=["G", "BP", "RP"], objects=["G", "BP", "RP"],
                               label="Bands")
    show_rejected = param.Boolean(default=True, label="Show rejected epochs")
    period_days = param.Number(default=0.0, bounds=(0.0, None),
                               label="Fold on period [d] (0 = no fold)")

    def __init__(self, context, payload, **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.frame = plots.to_frame(payload.table("epochs"))
        present = [b for b in ("G", "BP", "RP") if b in set(self.frame["band"])]
        self.param.bands.objects = present
        self.bands = present

    def filtered(self):
        frame = self.frame[self.frame["band"].isin(self.bands)]
        if not self.show_rejected:
            frame = frame[frame["status"] == "accepted"]
        return frame

    @param.depends("bands", "show_rejected")
    def curve(self):
        return pn.pane.HoloViews(
            plots.light_curve(self.filtered(), bands=self.bands),
            sizing_mode="stretch_width",
        )

    @param.depends("bands", "show_rejected")
    def snr(self):
        return pn.pane.HoloViews(
            plots.uncertainty_vs_time(self.filtered()), sizing_mode="stretch_width"
        )

    @param.depends("bands", "show_rejected")
    def distribution(self):
        return pn.pane.HoloViews(
            plots.magnitude_distribution(self.filtered()), sizing_mode="stretch_width"
        )

    @param.depends("bands", "show_rejected", "period_days")
    def folded(self):
        if self.period_days <= 0:
            return pn.pane.HTML(
                "<div style='font-size:12px;color:#666;padding:20px 0'>"
                "Enter a period above to fold the light curve.<br>"
                "<b>No period search is performed.</b> A period must come from you "
                "or from a Gaia variability product, so that what is plotted is "
                "never an artefact of a search this application ran silently.</div>"
            )
        return pn.pane.HoloViews(
            plots.phase_fold(self.filtered(), self.period_days),
            sizing_mode="stretch_width",
        )

    def controls(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML("<b>Photometry</b>"),
            pn.Param(
                self.param, parameters=["bands", "show_rejected", "period_days"],
                widgets={"bands": {"type": pn.widgets.CheckBoxGroup, "inline": True}},
                show_name=False,
            ),
            width=300, sizing_mode="fixed", margin=(0, 18, 0, 0),
        )

    def panel(self) -> pn.Column:
        summary = self.payload.summary
        bits = []
        for band in ("G", "BP", "RP"):
            n = summary.get(f"n_{band}")
            if n:
                bits.append(
                    f"{band}: {n} transits, median {summary[f'median_{band}_mag']:.3f} mag, "
                    f"spread {summary[f'ptp_{band}_mag']:.3f} mag"
                )
        return pn.Column(
            caveat(
                "Gaia <b>DR3</b> epoch photometry. DR4 epoch photometry is not in the "
                "June-2026 prerelease and is not public until 2026-12-02, so these are "
                "not the same measurements as the DR4 epoch astrometry on the other "
                "tabs. Do not place a DR3 and a DR4 quantity on one axis."
            ),
            pn.pane.HTML(
                "<div style='font-size:12px;color:#444;margin-top:6px'>"
                + "<br>".join(bits) + "</div>"
            ),
            pn.Tabs(
                ("Light curve", pn.Column(self.curve, sizing_mode="stretch_width")),
                ("Signal-to-noise", pn.Column(self.snr, sizing_mode="stretch_width")),
                ("Distribution", pn.Column(self.distribution, sizing_mode="stretch_width")),
                ("Phase fold", pn.Column(self.folded, sizing_mode="stretch_width")),
                dynamic=True, sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )


def unavailable_panel(descriptor) -> pn.Column:
    """What to show when a source has no epoch photometry: an explanation."""
    return pn.Column(
        pn.pane.HTML(
            "<h3 style='margin:8px 0 4px 0'>No epoch photometry for this source</h3>"
            f"<div style='font-size:12px;color:#555;max-width:720px'>{descriptor.detail}</div>"
            "<div style='font-size:12px;color:#555;max-width:720px;margin-top:10px'>"
            "Of the 12 prerelease sources, Gaia DR3 published epoch photometry for "
            "three: <b>Gaia-4</b> and the two variable QSOs. Those are the ones with "
            "a light curve to show today."
            "</div>"
        ),
        sizing_mode="stretch_width",
    )
