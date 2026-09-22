"""The overview tab: identity and summary statistics, by provenance."""

from __future__ import annotations

import panel as pn

from gaia_dr4_explorer.data.catalog import entry
from gaia_dr4_explorer.products.astrometry import plots
from gaia_dr4_explorer.ui.components import notes_panel, stat_table


def overview_panel(context, payload, release: str) -> pn.Column:
    """Three clearly separated blocks, so no number's origin is ambiguous."""
    summary = dict(payload.summary)
    warnings = summary.pop("warnings", [])
    cat = entry(context.source_id)

    identity = {
        "source_id": context.source_id,
        "common name": context.common_name or "not named on the release page",
        "sample category": cat.category_label if cat else "unknown",
        "release": release,
    }
    page = {
        "G magnitude": cat.page_g_mag if cat else float("nan"),
        "parallax [mas]": cat.page_parallax_mas if cat else float("nan"),
    }
    measured = {
        "FoV transits": summary.get("n_transits"),
        "CCD samples": summary.get("n_ccd_slots"),
        "with finite AL centroid": summary.get("n_ccd_finite"),
        "used by AGIS (AL)": summary.get("n_used_by_agis_al"),
        "fraction used": summary.get("frac_used"),
        "time span [yr]": summary.get("span_yr"),
        "median sigma_AL used [mas]": summary.get("median_sigma_al_used_mas"),
        "AGIS excess noise [mas]": summary.get("agis_source_excess_noise_mas"),
    }

    return pn.Column(
        pn.pane.HTML(
            f"<h2 style='margin:0 0 2px 0'>{context.label}</h2>"
            f"<div style='color:#888;font-family:monospace;font-size:13px'>"
            f"{context.source_id}</div>"
        ),
        pn.Row(
            stat_table(identity, kind="measured", title="Identity"),
            stat_table(page, kind="page", title="Release-page values"),
            stat_table(measured, kind="measured", title="Computed from the VOTable"),
        ),
        pn.pane.HTML(
            '<div style="font-size:11px;color:#777;max-width:760px;">'
            "The AGIS excess noise is an <b>input</b> published with the epoch data, not a "
            "quantity this application fits. In this sample it is what separates the "
            "binaries: 6.58, 1.23 and 0.12 mas for the three orbit sources, and 0.00 for "
            "every parallax and magnitude example.</div>"
        ),
        _epoch_preview(payload),
        notes_panel(list(warnings)),
        sizing_mode="stretch_width",
    )


def _epoch_preview(payload) -> pn.Column:
    """A look at the actual measurements.

    Without this the landing page of an epoch-astrometry explorer contains no
    epoch astrometry, only counts of it.
    """
    frame = plots.to_frame(payload.table("ccd"))
    plot = plots.centroid_vs_time(frame).opts(
        responsive=True, height=300,
        title="Along-scan centroid, every CCD observation",
    )
    return pn.Column(
        pn.pane.HTML(
            "<h3 style='margin:14px 0 2px 0'>Epoch data</h3>"
            "<div style='font-size:11px;color:#777;margin-bottom:6px'>"
            "Every CCD measurement, AGIS-used in blue and rejected in red. "
            "The <b>Astrometry</b> tab has the filters, the focal-plane matrix "
            "and the flag table.</div>"
        ),
        pn.pane.HoloViews(plot, sizing_mode="stretch_width"),
        sizing_mode="stretch_width",
    )
