"""The raw / metadata tab: published schema, provenance, and exports."""

from __future__ import annotations

import io

import numpy as np
import panel as pn

from gaia_dr4_explorer.products.astrometry.exporters import (
    FORMATS,
    LOSSY_CSV_NOTE,
    filename,
)
from gaia_dr4_explorer.products.astrometry.schema import FIELDS


def raw_panel(context, payload, plugin) -> pn.Column:
    """Field-by-field metadata plus download buttons."""
    raw = payload.raw
    rows = []
    for name in raw.colnames:
        col = raw[name]
        spec = FIELDS.get(name)
        n_masked = int(np.sum(col.mask)) if getattr(col, "mask", None) is not None else 0
        rows.append(
            f"<tr><td style='font-family:monospace'>{name}</td>"
            f"<td>{col.dtype}</td>"
            f"<td>{col.unit or '—'}</td>"
            f"<td>{spec.level.value if spec else '?'}</td>"
            f"<td>{'yes' if spec and spec.is_bitmask else ''}</td>"
            f"<td>{n_masked or ''}</td>"
            f"<td style='color:#555'>{(col.description or (spec.description if spec else '')) or ''}</td>"
            "</tr>"
        )
    header = (
        "<tr><th>field</th><th>dtype</th><th>unit</th><th>level</th>"
        "<th>bitmask</th><th>masked rows</th><th>description</th></tr>"
    )
    schema = pn.pane.HTML(
        f"<table style='font-size:11px;border-collapse:collapse' border=0>"
        f"{header}{''.join(rows)}</table>",
        sizing_mode="stretch_width",
    )

    params = payload.raw.meta.get("votable_params", {})
    votable_meta = pn.pane.HTML(
        "<b>VOTable parameters</b><table style='font-size:11px;'>"
        + "".join(
            f"<tr><td style='padding-right:14px;color:#555'>{k}</td>"
            f"<td style='font-family:monospace'>{v}</td></tr>"
            for k, v in params.items()
        )
        + "</table>"
    )

    provenance = pn.pane.JSON(
        payload.provenance.to_dict() if payload.provenance else {},
        name="Provenance", depth=3, sizing_mode="stretch_width", height=300,
    )

    return pn.Column(
        pn.pane.HTML("<h3 style='margin-bottom:4px'>Published schema</h3>"),
        schema,
        pn.Row(votable_meta, margin=(10, 0)),
        pn.pane.HTML("<h3 style='margin-bottom:4px'>Provenance</h3>"),
        provenance,
        pn.pane.HTML("<h3 style='margin-bottom:4px'>Export</h3>"),
        _downloads(context, payload, plugin),
        pn.pane.HTML(f"<div style='font-size:11px;color:#a04000'>{LOSSY_CSV_NOTE}</div>"),
        sizing_mode="stretch_width",
    )


def _downloads(context, payload, plugin) -> pn.Row:
    buttons = []
    for fmt, label in (
        ("raw", "Raw VOTable"),
        ("parquet", "Normalized Parquet"),
        ("ecsv", "Normalized ECSV"),
        ("csv", "Normalized CSV (lossy)"),
        ("provenance", "Provenance JSON"),
    ):
        ext_fmt = "votable" if fmt == "raw" else ("json" if fmt == "provenance" else fmt)
        what = {"raw": "raw", "provenance": "provenance"}.get(fmt, "ccd")
        name = filename(context.source_id, context.release, what, ext_fmt)

        def _make(fmt=fmt):
            def _cb():
                return io.BytesIO(plugin.export(payload, fmt))

            return _cb

        buttons.append(
            pn.widgets.FileDownload(
                callback=_make(), filename=name, label=label, button_type="default", width=210
            )
        )
    return pn.Row(*buttons, sizing_mode="stretch_width")


__all__ = ["raw_panel", "FORMATS"]
