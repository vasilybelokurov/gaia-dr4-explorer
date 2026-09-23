"""Small shared UI pieces."""

from __future__ import annotations

import panel as pn

PROVENANCE_COLOURS = {
    "page": "#8c6d1f",     # quoted on the ESA release page
    "measured": "#1f77b4",  # computed from the VOTable
    "computed": "#6a3d9a",  # computed by us with gaiasupdate
}

PROVENANCE_LABELS = {
    "page": "from the ESA release page",
    "measured": "measured from the VOTable",
    "computed": "recomputed with gaiasupdate",
}


#: Index of the per-object main view among the object's own tabs -- the epoch
#: astrometry tab the app opens on. The overview is a summary; the main view
#: is the data.
MAIN_TAB = 1

#: Label of the pseudo-tab that returns to the main view.
MAIN_VIEW_LABEL = "↩ Main view"


def object_tabs(*tabs, main: int = MAIN_TAB, **params) -> pn.Tabs:
    """The object's tabs, led by a "Main view" entry in the same row.

    A tab header cannot hold a button, so "Main view" is the first tab: picking
    it switches straight back to the main view, which is where it opens.

    Parameters
    ----------
    *tabs : (title, content) pairs
    main : int
        Index of the main view within ``tabs``.
    """
    t = pn.Tabs((MAIN_VIEW_LABEL, pn.Spacer(height=0)), *tabs, active=main + 1, **params)

    def back(event) -> None:
        if event.new == 0:
            t.active = main + 1

    t.param.watch(back, "active")
    return t


def provenance_badge(kind: str) -> pn.pane.HTML:
    """A coloured label saying where a block of numbers came from."""
    colour = PROVENANCE_COLOURS.get(kind, "#555555")
    label = PROVENANCE_LABELS.get(kind, kind)
    return pn.pane.HTML(
        f'<span style="background:{colour};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;letter-spacing:0.3px;">{label}</span>',
        margin=(0, 0, 4, 0),
    )


def release_badge(release: str) -> pn.pane.HTML:
    """Release marker; release candidates are called out in red."""
    candidate = "RC" in release.upper()
    colour = "#c0392b" if candidate else "#2c3e50"
    suffix = " — release candidate, identifiers may change" if candidate else ""
    return pn.pane.HTML(
        f'<span style="background:{colour};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:11px;">{release}{suffix}</span>'
    )


#: Keys whose values are identifiers, never quantities: no thousands separators.
IDENTIFIER_KEYS = frozenset({"source_id", "transit_id", "solution_id"})


def stat_table(rows: dict[str, object], *, kind: str, title: str | None = None) -> pn.Column:
    """A labelled block of key/value statistics with its provenance badge."""
    body = "".join(
        f'<tr><td style="padding:2px 14px 2px 0;color:#555;">{k}</td>'
        f'<td style="padding:2px 0;font-family:monospace;">'
        f"{v if k in IDENTIFIER_KEYS else _fmt(v)}</td></tr>"
        for k, v in rows.items()
    )
    items = []
    if title:
        items.append(pn.pane.HTML(f"<b>{title}</b>", margin=(0, 0, 2, 0)))
    items.append(provenance_badge(kind))
    items.append(pn.pane.HTML(f'<table style="font-size:12px;">{body}</table>'))
    return pn.Column(*items, margin=(6, 18, 6, 0))


def notes_panel(messages: list[str], *, title: str = "Notes on this source") -> pn.pane.HTML:
    if not messages:
        return pn.pane.HTML("")
    items = "".join(f"<li>{m}</li>" for m in messages)
    return pn.pane.HTML(
        f'<details style="font-size:12px;color:#444;"><summary>{title} '
        f"({len(messages)})</summary><ul>{items}</ul></details>"
    )


def caveat(text: str) -> pn.pane.HTML:
    return pn.pane.HTML(
        f'<div style="border-left:3px solid #c0392b;padding:6px 10px;'
        f'background:#fdf3f2;font-size:12px;color:#5a2d2a;">{text}</div>'
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "—"
        if abs(value) >= 1e5 or (value != 0 and abs(value) < 1e-3):
            return f"{value:.4e}"
        return f"{value:,.4f}".rstrip("0").rstrip(".")
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
