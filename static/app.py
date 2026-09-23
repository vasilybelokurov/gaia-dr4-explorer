"""Static (Pyodide) build of the Gaia DR4 Object Explorer.

This is the inspection-only profile. It runs entirely in the browser, so it
cannot do the two things that need a Python process or a network call:

* the DR4-like source update, because ``gaiasupdate`` imports ``astroquery`` at
  module scope and ``astroquery`` is not in the Pyodide distribution (and
  Pyodide ships pandas 3.x against gaiasupdate's ``pandas<3.0``). Fit results
  are shown from the precomputed reference table instead;
* Gaia DR3 epoch photometry, which needs a live archive request.

Everything else -- the prerelease data, normalization, and every astrometry
view -- is the same code the server profile runs.
"""

import io
import zipfile

import panel as pn

from _bundled_data import prerelease_zip_bytes
from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.data.bundled import BundledProductProvider
from gaia_dr4_explorer.data.catalog import catalog, reference_fits
from gaia_dr4_explorer.data.external_spectra import ExternalSpectraProvider
from gaia_dr4_explorer.domain import SourceContext, SourceKey
from gaia_dr4_explorer.domain.product import ProductState
from gaia_dr4_explorer.products import registry
from gaia_dr4_explorer.products.astrometry import normalize_epoch_astrometry
from gaia_dr4_explorer.ui import context as context_ui
from gaia_dr4_explorer.ui import photometry as photometry_ui
from gaia_dr4_explorer.ui import spectra as spectra_ui
from gaia_dr4_explorer.ui.astrometry import AstrometryView
from gaia_dr4_explorer.ui.components import caveat, object_tabs, release_badge
from gaia_dr4_explorer.ui.external_spectra import external_spectra_section
from gaia_dr4_explorer.ui.overview import overview_panel

pn.extension("tabulator", sizing_mode="stretch_width")

MEMBER = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"
PREFERRED = (4318465066420528000, 4181040337841125632, 3937211745905473024)


def load_table():
    """Parse the embedded archive, exactly as the server profile parses the file."""
    from astropy.io.votable import parse_single_table

    with zipfile.ZipFile(io.BytesIO(prerelease_zip_bytes())) as zf:
        xml = zf.read(MEMBER)
    votable = parse_single_table(io.BytesIO(xml))
    table = votable.to_table(use_names_over_ids=True)
    table.meta["votable_params"] = {
        p.name: p.value for p in votable.params if p.name is not None
    }
    return table


TABLE = load_table()
RELEASE = str(TABLE.meta["votable_params"].get("release", "Gaia DR4_RC3"))
SOURCE_IDS = sorted({int(v) for v in TABLE["source_id"]})
ENTRIES = catalog()
FITS = reference_fits()
EXTERNAL = ExternalSpectraProvider(allow_network=False)

# The Gaia archive sends no CORS header, so a page cannot reach it. The DR3
# products for these twelve sources ship with the package instead.
BUNDLED = BundledProductProvider()
registry.load_builtin_plugins()
PRODUCT_TABS = [
    (registry.get("epoch_photometry").bind(BUNDLED), "Photometry", photometry_ui),
    (registry.get("xp_spectrum").bind(BUNDLED), "Spectra", spectra_ui),
    (registry.get("context").bind(BUNDLED), "Context", context_ui),
]


def subset(source_id):
    import numpy as np

    return TABLE[np.asarray(TABLE["source_id"]) == int(source_id)]


def fit_panel(source_id):
    """Precomputed fit results: this build cannot run gaiasupdate."""
    values = FITS.get(int(source_id), {})
    rows = "".join(
        f"<tr><td style='padding:2px 16px 2px 0;color:#555'>{k.replace('fit_', '')}</td>"
        f"<td style='font-family:monospace'>{v:,.6g}</td></tr>"
        for k, v in values.items()
    )
    return pn.Column(
        caveat(
            "This page runs in your browser, so it cannot run <code>gaiasupdate</code>: "
            "that package imports <code>astroquery</code>, which is not available in "
            "Pyodide. The values below were computed in advance with "
            "<code>gaiasupdate 0.1.2</code> and shipped with the page. They are "
            "recomputed from epoch data and are <b>not</b> official Gaia catalogue values. "
            "Run the app locally for a live fit."
        ),
        pn.pane.HTML(f"<table style='font-size:12px;margin-top:8px'>{rows}</table>"),
        pn.pane.HTML(
            "<div style='font-size:11px;color:#555;max-width:760px;margin-top:10px'>"
            "<b>Two goodness-of-fit numbers.</b> gaiasupdate computes &chi;<sup>2</sup> "
            "against the <i>measurement</i> variance, which excludes the AGIS source "
            "excess noise, even though the fit is weighted by the <i>total</i> variance "
            "including it. For Gaia BH3 that is F2 = 894 versus 0.13. Neither is the "
            "catalogue <code>astrometric_gof_al</code>.</div>"
        ),
    )


def sky_panel(source_id, context, payload):
    """Epoch astrometry on the sky, with the precomputed model drawn over it.

    The observations are placed from their published local-plane coordinates,
    exactly as in the server profile; only the model comes precomputed.
    """
    from gaia_dr4_explorer.products.astrometry import skyplane
    from gaia_dr4_explorer.ui.skyplane import SkyPlaneView

    values = FITS.get(int(source_id), {})
    try:
        model = skyplane.SkyModel.from_reference(values)
    except KeyError:
        model = None
    return SkyPlaneView(
        context=context, payload=payload, model=model,
        model_label="DR4-like refit (gaiasupdate, precomputed)",
        model_note="Computed in advance with gaiasupdate 0.1.2 and shipped with this "
                   "page; recomputed from epoch data, not an official catalogue value.",
    ).layout()


def build(source_id):
    entry = ENTRIES.get(int(source_id))
    context = SourceContext(
        key=SourceKey(RELEASE, int(source_id)),
        common_name=(entry.common_name or None) if entry else None,
        sample_category=entry.sample_category if entry else None,
    )
    normalized = normalize_epoch_astrometry(subset(source_id))

    class _Payload:
        raw = normalized.raw
        summary = normalized.summary() | {"warnings": normalized.warnings}
        provenance = normalized.provenance
        tables = {"transits": normalized.transits, "ccd": normalized.ccd}

        def table(self, name):
            return self.tables[name]

    payload = _Payload()
    view = AstrometryView(context=context, payload=payload)
    tabs = [
        ("Overview", overview_panel(context, payload, RELEASE)),
        ("Astrometry", pn.Row(view.controls(), view.panel())),
        ("Source fit", fit_panel(source_id)),
        ("Sky plane", sky_panel(source_id, context, payload)),
    ]
    for plugin, title, module in PRODUCT_TABS:
        descriptor = plugin.discover(context)
        if descriptor.state is not ProductState.AVAILABLE:
            content = module.unavailable_panel(descriptor)
        else:
            try:
                content = plugin.build_view(context, plugin.load(context)).panel()
            except Exception as exc:
                content = pn.pane.HTML(
                    f"<div style='color:#c0392b'>Could not load {title}: {exc}</div>")
        if title == "Spectra":
            # The archive search is shipped with the page: a browser cannot
            # query the archives, but its file links open directly.
            content = pn.Column(content, external_spectra_section(
                EXTERNAL.bundled(int(source_id))))
        tabs.append((title, content))
    return object_tabs(*tabs, dynamic=True)


options = {
    (ENTRIES[s].label if s in ENTRIES else str(s)): s for s in SOURCE_IDS
}
default = next((s for s in PREFERRED if s in SOURCE_IDS), SOURCE_IDS[0])
selector = pn.widgets.Select(name="Source", options=options, value=default)
body = pn.Column(build(default))
header = pn.pane.HTML("")


def on_select(event):
    sid = int(event.new)
    entry = ENTRIES.get(sid)
    name = entry.common_name if entry and entry.common_name else "Unnamed source"
    header.object = (
        f"<span style='font-size:15px'><b>{name}</b>"
        f"<span style='color:#888'> &nbsp;{sid}</span></span>"
    )
    body.objects = [build(sid)]


selector.param.watch(on_select, "value")
on_select(type("E", (), {"new": default})())

pn.template.FastListTemplate(
    title="Gaia DR4 Object Explorer",
    header=[header, release_badge(RELEASE)],
    sidebar=[
        pn.pane.HTML(
            "<b>Data source</b><br><span style='font-size:12px;color:#555'>"
            "Gaia DR4 prerelease (June 2026), bundled with this page</span>"
        ),
        selector,
        pn.pane.HTML("<hr style='border:none;border-top:1px solid #eee'>"),
        pn.pane.HTML(
            "<div style='font-size:11px;color:#666'>"
            "<b>Browser build.</b> Runs entirely client-side, with the Gaia DR4 "
            "prerelease and the DR3 photometry, spectra and SIMBAD/ADS records "
            "bundled into the page. Only the live <code>gaiasupdate</code> fit "
            "needs a Python process; its results here are precomputed.<br><br>"
            f"<a href='https://github.com/vasilybelokurov/gaia-dr4-explorer'>Source</a>"
            f" &middot; v{__version__}</div>"
        ),
    ],
    main=[body],
    sidebar_width=330,
    accent="#2c3e50",
).servable()
