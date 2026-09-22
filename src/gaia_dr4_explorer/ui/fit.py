"""The source-fit tab.

Values here are recomputed from epoch data with ``gaiasupdate``; they are not
official catalogue values, and the interface says so on every render.
"""

from __future__ import annotations

import panel as pn

from gaia_dr4_explorer.products.astrometry import plots
from gaia_dr4_explorer.ui.components import caveat, provenance_badge

F2_EXPLANATION = (
    "<b>Two goodness-of-fit numbers, deliberately.</b> gaiasupdate reports "
    "&chi;<sup>2</sup> against the <i>measurement</i> variance, which excludes the AGIS "
    "source excess noise &mdash; even though the fit itself is weighted by the "
    "<i>total</i> variance, which includes it. For a source with large excess noise the "
    "two differ enormously (Gaia BH3: F2 = 894 versus 0.13). Neither is the catalogue "
    "<code>astrometric_gof_al</code>."
)


class FitView:
    """Renders a source update, computing it only when asked."""

    def __init__(self, state, view):
        self.state = state
        self.view = view
        self.button = pn.widgets.Button(
            name="Run DR4-like source update", button_type="primary", width=260
        )
        self.button.on_click(self._run)
        self.output = pn.Column(
            pn.pane.HTML(
                '<div style="color:#666;font-size:12px;">Not computed yet. The fit uses the '
                "official gaiasupdate source-update routine and takes a second or two.</div>"
            )
        )

    def _run(self, _event) -> None:
        self.button.loading = True
        try:
            result = self.state.fit(force=True)
        except Exception as exc:
            self.output.objects = [caveat(f"The fit failed: {exc}")]
            return
        finally:
            self.button.loading = False
        self.output.objects = self._render(result)

    def _render(self, result) -> list:
        rows = "".join(
            f'<tr><td style="padding:2px 16px 2px 0;">{name}</td>'
            f'<td style="text-align:right;font-family:monospace;">{value:+.6f}</td>'
            f'<td style="padding-left:8px;font-family:monospace;color:#666;">'
            f"&plusmn; {error:.6f}</td>"
            f'<td style="padding-left:10px;color:#666;">{unit}</td></tr>'
            for name, (value, error, unit) in result.as_dict().items()
        )
        stats = {
            "measurements used": f"{result.n_measurements:,}",
            "outliers": f"{result.n_outliers} (the DR4-like path does no robust rejection)",
            "F2, measurement variance": f"{result.f2_measurement_variance:+.4f}",
            "F2, total variance": f"{result.f2_total_variance:+.4f}",
            "chi2, measurement variance": f"{result.chi2_measurement_variance:,.1f}",
            "chi2, total variance": f"{result.chi2_total_variance:,.1f}",
            "AGIS excess noise (input)": f"{result.excess_noise_input_mas:.4f} mas",
        }
        stat_rows = "".join(
            f'<tr><td style="padding:2px 16px 2px 0;color:#555;">{k}</td>'
            f'<td style="font-family:monospace;">{v}</td></tr>'
            for k, v in stats.items()
        )
        return [
            caveat(result.caveat),
            provenance_badge("computed"),
            pn.pane.HTML(
                f'<table style="font-size:13px;">{rows}</table>'
                f'<hr style="border:none;border-top:1px solid #eee;margin:10px 0;">'
                f'<table style="font-size:12px;">{stat_rows}</table>'
            ),
            pn.pane.HTML(f'<div style="font-size:11px;color:#555;max-width:780px;">'
                         f"{F2_EXPLANATION}</div>"),
            pn.pane.HoloViews(
                plots.residual_plots(result, self.view.frame),
                sizing_mode="stretch_width",
            ),
        ]

    def panel(self) -> pn.Column:
        return pn.Column(self.button, self.output, sizing_mode="stretch_width")
