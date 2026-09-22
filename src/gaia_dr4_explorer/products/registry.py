"""Registry of available product plugins.

The application builds its navigation from this, so adding a product means
registering a plugin and nothing else.
"""

from __future__ import annotations

from gaia_dr4_explorer.products.base import ProductPlugin

_REGISTRY: dict[str, ProductPlugin] = {}


def register(plugin: ProductPlugin) -> ProductPlugin:
    """Register *plugin*, replacing any earlier plugin with the same key."""
    if not plugin.key:
        raise ValueError(f"{type(plugin).__name__} has no key")
    _REGISTRY[plugin.key] = plugin
    return plugin


def get(key: str) -> ProductPlugin:
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(f"no product plugin {key!r}; registered: {sorted(_REGISTRY)}") from None


def all_plugins(*, static_only: bool = False) -> list[ProductPlugin]:
    """Registered plugins, in registration order."""
    plugins = list(_REGISTRY.values())
    if static_only:
        plugins = [p for p in plugins if p.static_safe]
    return plugins


def clear() -> None:
    """Drop every registration.  For tests."""
    _REGISTRY.clear()


def load_builtin_plugins() -> list[ProductPlugin]:
    """Import and register the plugins that ship with the application."""
    from gaia_dr4_explorer.products.astrometry.plugin import EpochAstrometryPlugin

    if "epoch_astrometry" not in _REGISTRY:
        register(EpochAstrometryPlugin())
    return all_plugins()
