"""Spectra of the object in other archives: a section of the Spectra tab.

Shows what each archive *lists* near the source -- metadata only. Nothing is
downloaded; each spectrum links to the archive's own file or DataLink page.
The search itself runs in :mod:`gaia_dr4_explorer.data.external_spectra`.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import panel as pn

from gaia_dr4_explorer.data.external_spectra import SearchReport, SearchStatus

STATUS_STYLE = {
    SearchStatus.FOUND: ("#1a7f37", "found"),
    SearchStatus.NONE: ("#6e7781", "none"),
    SearchStatus.FAILED: ("#c0392b", "failed"),
    SearchStatus.TIMEOUT: ("#b35900", "timed out"),
    SearchStatus.SKIPPED: ("#8c6d1f", "not searched"),
}

EXPLANATION = (
    "What other archives list near this source, <b>metadata only</b>: nothing has "
    "been downloaded. Each archive is asked in a cone whose radius includes the "
    "star's proper motion over 1975–2027. <b>Sep.</b> is the closest approach to the "
    "star's path over that span, because archives store either the position on the "
    "night or a J2000 catalogue position. A search with failed, timed-out or "
    "unsearched archives is incomplete, never a statement that no spectra exist."
)


def records_frame(report: SearchReport, archive: str) -> pd.DataFrame:
    """One archive's spectra as a display table; missing values stay empty."""
    rows = []
    for r in next(res for res in report.results if res.archive == archive).records:
        rows.append({
            "collection": r.collection,
            "instrument": r.instrument,
            "obs_id": r.obs_id,
            "target": r.target_name,
            "date": _mjd_to_date(r.mjd),
            "λ min [nm]": _round(r.wl_min_nm, 1),
            "λ max [nm]": _round(r.wl_max_nm, 1),
            "R": _round(r.resolving_power, 0),
            "S/N": _round(r.snr, 1),
            "size [MB]": None if r.size_bytes is None else round(r.size_bytes / 1e6, 2),
            "sep. [″]": _round(r.separation_arcsec, 2),
            "link": r.access_url,
        })
    return pd.DataFrame(rows)


def summary_html(report: SearchReport) -> str:
    """Per-archive status table."""
    body = []
    for res in report.results:
        colour, label = STATUS_STYLE[res.status]
        detail = res.error if res.status is not SearchStatus.FOUND else ""
        elapsed = "" if math.isnan(res.elapsed_s) else f"{res.elapsed_s:.1f} s"
        body.append(
            f"<tr><td style='padding:2px 14px 2px 0'><b>{res.archive}</b></td>"
            f"<td style='padding:2px 14px 2px 0;color:{colour}'>{label}</td>"
            f"<td style='padding:2px 14px 2px 0;text-align:right'>"
            f"{res.n_spectra if res.status is SearchStatus.FOUND else ''}</td>"
            f"<td style='padding:2px 14px 2px 0;color:#777'>{res.radius_arcsec:.0f}″</td>"
            f"<td style='padding:2px 14px 2px 0;color:#777'>{elapsed}</td>"
            f"<td style='color:#777;font-size:11px'>{_escape(detail)[:160]}</td></tr>"
        )
    head = ("<tr style='color:#555;text-align:left'><th>Archive</th><th>Status</th>"
            "<th>Spectra</th><th>Radius</th><th>Time</th><th></th></tr>")
    return f"<table style='font-size:12px;border-collapse:collapse'>{head}{''.join(body)}</table>"


def report_panel(report: SearchReport, *, origin: str) -> pn.Column:
    """The status table and, per archive with spectra, a collapsed table of them."""
    n = sum(r.n_spectra for r in report.results)
    n_arch = sum(r.status is SearchStatus.FOUND for r in report.results)
    incomplete = [r.archive for r in report.results
                  if r.status not in (SearchStatus.FOUND, SearchStatus.NONE)]
    head = (f"<div style='font-size:12px;color:#444;margin:4px 0'><b>{n}</b> "
            f"{'spectrum' if n == 1 else 'spectra'} in "
            f"<b>{n_arch}</b> of {len(report.results)} archives. {origin} "
            f"{report.searched_at.replace('T', ' ').replace('+00:00', ' UTC')}.")
    if incomplete:
        head += (f" <span style='color:#b35900'>Incomplete: {', '.join(incomplete)} did "
                 "not answer or were not searched.</span>")
    items = [pn.pane.HTML(head + "</div>"), pn.pane.HTML(summary_html(report))]
    for res in report.results:
        if res.status is not SearchStatus.FOUND:
            continue
        table = pn.widgets.Tabulator(
            records_frame(report, res.archive), disabled=True, show_index=False,
            pagination="local", page_size=15, sizing_mode="stretch_width",
            formatters={"link": {"type": "link", "label": "open", "target": "_blank"}},
            header_filters=True,
        )
        noun = "spectrum" if res.n_spectra == 1 else "spectra"
        items.append(pn.Card(table, title=f"{res.archive} — {res.n_spectra} {noun}",
                             collapsed=True, sizing_mode="stretch_width"))
    return pn.Column(*items, sizing_mode="stretch_width")


#: Label of the button that starts a live search.
SEARCH_LABEL = "Search for additional spectra"

_ORIGINS = {"shipped": "Search shipped with the app, run",
            "saved": "Saved search, run",
            "live": "Live search, run"}


def external_spectra_section(
    latest: SearchReport | None,
    live_search: Callable[[], SearchReport] | None = None,
    *,
    origin: str = "shipped",
) -> pn.Column:
    """The section under the Gaia XP spectrum.

    Nothing is searched until the button is pressed: a search takes ~15 s,
    and for an arbitrary DR4 source there is usually no earlier result.

    Parameters
    ----------
    latest : SearchReport or None
        The newest result available without searching (saved or shipped).
    live_search : callable, optional
        Runs a fresh search. Given in the desktop app, absent in the browser
        build, which cannot query archives.
    origin : {"shipped", "saved"}
        Where ``latest`` came from, for the label.
    """
    out = pn.Column(sizing_mode="stretch_width")
    if latest is not None:
        out.objects = [report_panel(latest, origin=_ORIGINS.get(origin, origin))]
    elif live_search is None:
        out.objects = [pn.pane.HTML(
            "<div style='font-size:12px;color:#666'>No archive search is shipped for "
            "this source, and this build cannot query archives. Run the app locally "
            "to search.</div>")]
    else:
        out.objects = [pn.pane.HTML(
            "<div style='font-size:12px;color:#666'>Not searched yet. A search asks "
            "every archive at once and takes about 15 s.</div>")]

    items = [
        pn.pane.HTML("<h3 style='margin:18px 0 4px 0'>Spectra in other archives</h3>"),
        pn.pane.HTML(f"<div style='font-size:11px;color:#555;max-width:900px'>"
                     f"{EXPLANATION}</div>"),
    ]
    if live_search is not None:
        button = pn.widgets.Button(
            name=SEARCH_LABEL if latest is None else f"{SEARCH_LABEL} (search again)",
            button_type="primary" if latest is None else "default", width=300)

        def run(_event) -> None:
            button.loading = True
            try:
                out.objects = [report_panel(live_search(), origin=_ORIGINS["live"])]
            except Exception as exc:
                out.objects = [pn.pane.HTML(
                    f"<div style='color:#c0392b'>Search failed: {_escape(str(exc))}</div>")]
            finally:
                button.loading = False

        button.on_click(run)
        items.append(button)
    items.append(out)
    return pn.Column(*items, sizing_mode="stretch_width")


def _round(x: float, nd: int):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, nd)


def _mjd_to_date(mjd: float) -> str:
    if mjd is None or math.isnan(mjd):
        return ""
    from astropy.time import Time

    return Time(mjd, format="mjd").to_value("iso", subfmt="date")


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
