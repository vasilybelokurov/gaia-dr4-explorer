"""The spectra tab."""

from __future__ import annotations

import panel as pn
import param

from gaia_dr4_explorer.products.spectra import plots
from gaia_dr4_explorer.ui.components import caveat


class SpectrumView(param.Parameterized):
    """Mean BP/RP spectrum for one source."""

    show_errors = param.Boolean(default=True, label="Show ±1σ band")

    def __init__(self, context, payload, **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.frame = plots.to_frame(payload.table("spectrum"))

    @param.depends("show_errors")
    def spectrum(self):
        return pn.pane.HoloViews(
            plots.mean_spectrum(self.frame, show_errors=self.show_errors),
            sizing_mode="stretch_width",
        )

    def snr(self):
        return pn.pane.HoloViews(
            plots.signal_to_noise(self.frame), sizing_mode="stretch_width"
        )

    def controls(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML("<b>Spectrum</b>"),
            pn.Param(self.param, parameters=["show_errors"], show_name=False),
            width=300, sizing_mode="fixed", margin=(0, 18, 0, 0),
        )

    def panel(self) -> pn.Column:
        s = self.payload.summary
        return pn.Column(
            caveat(
                "Gaia <b>DR3</b> BP/RP mean spectrum. DR4 will publish mean <i>and</i> "
                "epoch spectra from 2026-12-02; neither is in the June-2026 prerelease. "
                "Do not place a DR3 and a DR4 quantity on one axis."
            ),
            pn.pane.HTML(
                f"<div style='font-size:12px;color:#444;margin-top:6px'>"
                f"{s['n_samples']} samples from {s['wavelength_min_nm']:.0f} to "
                f"{s['wavelength_max_nm']:.0f} nm; peak at "
                f"{s['peak_wavelength_nm']:.0f} nm</div>"
            ),
            self.spectrum,
            self.snr(),
            sizing_mode="stretch_width",
        )


def unavailable_panel(descriptor) -> pn.Column:
    return pn.Column(
        pn.pane.HTML(
            "<h3 style='margin:8px 0 4px 0'>No BP/RP spectrum for this source</h3>"
            f"<div style='font-size:12px;color:#555;max-width:720px'>{descriptor.detail}</div>"
            "<div style='font-size:12px;color:#555;max-width:720px;margin-top:10px'>"
            "Gaia DR3 published BP/RP spectra for eight of the twelve prerelease "
            "sources. The four without are the three faintest and one QSO.</div>"
        ),
        sizing_mode="stretch_width",
    )
