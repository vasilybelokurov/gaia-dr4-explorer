"""The context tab: SIMBAD identity and ADS bibliography."""

from __future__ import annotations

import panel as pn
import param

from gaia_dr4_explorer.products.context.plugin import ADS_ABS_URL, SIMBAD_OBJECT_URL


class ContextView(param.Parameterized):
    """What is known about this object outside Gaia."""

    def __init__(self, context, payload, **params):
        super().__init__(**params)
        self.context = context
        self.payload = payload
        self.simbad = payload.raw.get("simbad") or {}
        self.ads = payload.raw.get("ads") or {}

    def _identity(self) -> pn.pane.HTML:
        if not self.simbad:
            return pn.pane.HTML(
                "<div style='font-size:12px;color:#666'>Not in SIMBAD.</div>"
            )
        s = self.simbad
        link = f"{SIMBAD_OBJECT_URL}{str(s.get('main_id', '')).replace(' ', '+')}"
        rows = [
            ("main identifier", f"<a href='{link}' target='_blank'>{s.get('main_id')}</a>"),
            ("object type", f"{s.get('otype_txt') or s.get('otype')}"),
            ("spectral type", s.get("sp_type") or "—"),
            ("references in SIMBAD", s.get("nbref")),
            ("redshift", s.get("rvz_redshift") if s.get("rvz_redshift") is not None else "—"),
        ]
        body = "".join(
            f"<tr><td style='padding:2px 16px 2px 0;color:#555'>{k}</td>"
            f"<td style='font-family:monospace'>{v}</td></tr>"
            for k, v in rows
        )
        return pn.pane.HTML(
            f"<h3 style='margin:6px 0 4px 0'>SIMBAD</h3>"
            f"<table style='font-size:12px'>{body}</table>"
        )

    def _cross_ids(self) -> pn.pane.HTML:
        ids = self.simbad.get("cross_ids") or []
        if not ids:
            return pn.pane.HTML("")
        chips = "".join(
            f"<span style='display:inline-block;background:#eef2f7;border-radius:10px;"
            f"padding:2px 9px;margin:2px;font-size:11px;font-family:monospace'>{i}</span>"
            for i in ids
        )
        return pn.pane.HTML(
            f"<h3 style='margin:14px 0 4px 0'>Cross-identifiers "
            f"<span style='color:#888;font-weight:normal;font-size:12px'>"
            f"({len(ids)})</span></h3>{chips}"
        )

    def _bibliography(self) -> pn.pane.HTML:
        docs = self.ads.get("docs") or []
        found = self.ads.get("num_found", 0)
        query = self.ads.get("query", "")
        if not docs:
            return pn.pane.HTML(
                "<h3 style='margin:14px 0 4px 0'>Literature</h3>"
                "<div style='font-size:12px;color:#666'>No ADS abstract mentions this "
                "object by name.</div>"
            )
        rows = "".join(
            f"<tr><td style='padding:3px 10px 3px 0;white-space:nowrap'>"
            f"<a href='{ADS_ABS_URL}{d['bibcode']}/abstract' target='_blank'"
            f" style='font-family:monospace;font-size:11px'>{d['bibcode']}</a></td>"
            f"<td style='padding:3px 8px 3px 0;color:#666;font-size:11px'>"
            f"{d.get('citation_count', 0)} cites</td>"
            f"<td style='font-size:12px'>{d.get('title') or ''}</td></tr>"
            for d in docs
        )
        return pn.pane.HTML(
            f"<h3 style='margin:14px 0 4px 0'>Literature "
            f"<span style='color:#888;font-weight:normal;font-size:12px'>"
            f"({found} papers matching abs:\"{query}\", most cited first)</span></h3>"
            f"<table style='font-size:12px'>{rows}</table>"
        )

    def panel(self) -> pn.Column:
        return pn.Column(
            pn.pane.HTML(
                "<div style='font-size:11px;color:#777;max-width:760px'>"
                "External records, fetched once and shipped with the application: "
                "SIMBAD by <code>Gaia DR3 &lt;source_id&gt;</code>, and ADS by an "
                "abstract search on the object's name. Cross-identification is "
                "SIMBAD's, not ours.</div>"
            ),
            self._identity(),
            self._cross_ids(),
            self._bibliography(),
            sizing_mode="stretch_width",
        )


def unavailable_panel(descriptor) -> pn.Column:
    return pn.Column(
        pn.pane.HTML(
            "<h3 style='margin:8px 0 4px 0'>No external context</h3>"
            f"<div style='font-size:12px;color:#555;max-width:720px'>{descriptor.detail}</div>"
        ),
        sizing_mode="stretch_width",
    )
