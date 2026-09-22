"""Application shell: header, sidebar, and the product tabs."""

from __future__ import annotations

import panel as pn

from gaia_dr4_explorer.data.catalog import catalog
from gaia_dr4_explorer.products import registry
from gaia_dr4_explorer.ui.components import release_badge
from gaia_dr4_explorer.ui.fit import FitView
from gaia_dr4_explorer.ui.overview import overview_panel
from gaia_dr4_explorer.ui.raw_table import raw_panel
from gaia_dr4_explorer.ui.state import AppState

TITLE = "Gaia DR4 Object Explorer"


def build_app(provider, *, allow_fit: bool = True) -> pn.template.FastListTemplate:
    """Assemble the application.  Loads nothing until a source is selected."""
    pn.extension("tabulator", notifications=True)

    plugins = registry.load_builtin_plugins()
    astrometry = registry.get("epoch_astrometry")
    astrometry.bind(provider)

    source_ids = provider.source_ids()
    state = AppState(provider=provider, source_id=source_ids[0], release=provider.release())

    entries = catalog()
    options = {
        (entries[sid].label if sid in entries else str(sid)): sid for sid in source_ids
    }
    selector = pn.widgets.Select(name="Source", options=options, value=state.source_id)
    id_input = pn.widgets.TextInput(name="or enter a source_id", placeholder=str(source_ids[0]))
    reload_button = pn.widgets.Button(name="Reload", button_type="default", width=120)
    copy_button = pn.widgets.Button(name="Copy source_id", button_type="default", width=140)
    status = pn.pane.HTML("")
    header_label = pn.pane.HTML("")
    body = pn.Column(sizing_mode="stretch_width")

    def render() -> None:
        context = state.context()
        header_label.object = (
            f"<span style='font-size:15px'><b>{context.label}</b>"
            f"<span style='color:#888'> &nbsp;{context.source_id}</span></span>"
        )
        try:
            payload = state.payload(astrometry)
        except Exception as exc:
            body.objects = [
                pn.pane.HTML(
                    f"<div style='color:#c0392b'>Could not load epoch astrometry: {exc}</div>"
                )
            ]
            return
        view = astrometry.build_view(context, payload)
        tabs = [
            ("Overview", overview_panel(context, payload, state.release)),
            ("Astrometry", pn.Row(view.controls(), view.panel(), sizing_mode="stretch_width")),
        ]
        if allow_fit:
            tabs.append(("Source fit", FitView(state, view).panel()))
        else:
            tabs.append(("Source fit", _static_fit_notice(context)))
        tabs.append(("Raw / metadata", raw_panel(context, payload, astrometry)))
        body.objects = [pn.Tabs(*tabs, dynamic=True, sizing_mode="stretch_width")]

    def on_select(event) -> None:
        sid = int(event.new)
        if state.source_id != sid:
            state.source_id = sid   # the state watcher calls render()
        else:
            render()

    def on_id_input(event) -> None:
        text = (event.new or "").strip()
        if not text:
            return
        try:
            sid = int(text)
        except ValueError:
            status.object = f"<span style='color:#c0392b'>{text!r} is not a source_id</span>"
            return
        if sid not in source_ids:
            status.object = (
                f"<span style='color:#c0392b'>{sid} is not in this release.</span>"
            )
            return
        status.object = ""
        selector.value = sid

    def on_reload(_event) -> None:
        state.reset_source()
        render()

    def on_copy(_event) -> None:
        pn.state.notifications.info(f"source_id {state.source_id}")

    def on_state_source_id(event) -> None:
        """Keep the sidebar in step when the source changes from elsewhere.

        The URL parameter writes straight to the state, so without this the
        dropdown would keep showing whatever was selected before.
        """
        sid = int(event.new)
        if sid not in source_ids:
            status.object = (
                f"<span style='color:#c0392b'>{sid} is not in this release; "
                "showing the previous source.</span>"
            )
            state.source_id = int(event.old)
            return
        if selector.value != sid:
            selector.value = sid   # triggers on_select, which re-renders
        else:
            render()

    selector.param.watch(on_select, "value")
    id_input.param.watch(on_id_input, "value")
    state.param.watch(on_state_source_id, "source_id")
    reload_button.on_click(on_reload)
    copy_button.on_click(on_copy)

    # A source_id in the URL makes a view shareable.  Sync before the first
    # render so a deep link does not paint the default source first.
    if pn.state.location is not None:
        pn.state.location.sync(state, {"source_id": "source_id"})

    if selector.value != state.source_id and state.source_id in source_ids:
        selector.value = state.source_id
    render()

    template = pn.template.FastListTemplate(
        title=TITLE,
        header=[header_label, release_badge(state.release)],
        sidebar=[
            pn.pane.HTML("<b>Data source</b><br><span style='font-size:12px;color:#555'>"
                         "Gaia DR4 prerelease (June 2026)</span>"),
            selector, id_input, pn.Row(reload_button, copy_button), status,
            pn.pane.HTML("<hr style='border:none;border-top:1px solid #eee'>"),
            _cache_status(provider),
            _plugin_status(plugins),
        ],
        main=[body],
        sidebar_width=330,
        accent="#2c3e50",
    )
    return template


def _static_fit_notice(context) -> pn.Column:
    from gaia_dr4_explorer.data.catalog import reference_fits

    values = reference_fits().get(context.source_id, {})
    rows = "".join(
        f"<tr><td style='padding-right:14px;color:#555'>{k}</td>"
        f"<td style='font-family:monospace'>{v:,.6f}</td></tr>"
        for k, v in values.items()
    )
    return pn.Column(
        pn.pane.HTML(
            "<div style='font-size:12px;color:#555'>This build cannot run gaiasupdate, so "
            "the values below are the precomputed reference results shipped with the "
            "application.</div>"
            f"<table style='font-size:12px;margin-top:8px'>{rows}</table>"
        )
    )


def _cache_status(provider) -> pn.pane.HTML:
    try:
        archive = provider.zip_path
        size = archive.stat().st_size if archive.exists() else 0
        text = f"{archive.name}<br>{size:,} bytes" if size else "archive not cached"
    except Exception:
        text = "unknown"
    return pn.pane.HTML(f"<b>Cache</b><br><span style='font-size:11px;color:#555'>{text}</span>")


def _plugin_status(plugins) -> pn.pane.HTML:
    items = "".join(
        f"<li>{p.title} <span style='color:#888'>({p.release})</span></li>" for p in plugins
    )
    return pn.pane.HTML(
        f"<b>Products</b><ul style='font-size:11px;color:#555;margin:4px 0 0 16px;padding:0'>"
        f"{items}</ul>"
        "<div style='font-size:11px;color:#888;margin-top:6px'>Photometry, spectra and "
        "external context are not yet implemented.</div>"
    )
