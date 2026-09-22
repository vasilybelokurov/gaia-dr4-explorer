"""Entry point for the interactive application."""

from __future__ import annotations

from gaia_dr4_explorer.config import AppConfig


def build(config: AppConfig, *, allow_fit: bool = True):
    """Construct the Panel application for *config*."""
    from gaia_dr4_explorer.data import PreReleaseProvider
    from gaia_dr4_explorer.ui.shell import build_app

    provider = PreReleaseProvider(config)
    provider.load_table()  # fail fast, with a clear message, before serving
    return build_app(provider, allow_fit=allow_fit)


def serve(config: AppConfig, *, port: int = 5006, show: bool = False) -> None:
    """Serve the application on *port*."""
    import panel as pn

    pn.serve(lambda: build(config), port=port, show=show, title="Gaia DR4 Object Explorer")
