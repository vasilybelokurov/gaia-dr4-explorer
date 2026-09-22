import pytest

from gaia_dr4_explorer.domain import ProductState, SourceContext, SourceKey
from gaia_dr4_explorer.products import registry
from gaia_dr4_explorer.products.base import ProductPlugin


class Dummy(ProductPlugin):
    key = "dummy"
    title = "Dummy"
    release = "Gaia DR3"
    static_safe = False

    def discover(self, context):
        return self.unavailable("nothing here")

    def load(self, context):
        raise AssertionError("load must not be called by discovery")

    def build_view(self, context, payload):
        return None


@pytest.fixture(autouse=True)
def clean_registry():
    registry.clear()
    yield
    registry.clear()


def test_register_and_get():
    plugin = registry.register(Dummy())
    assert registry.get("dummy") is plugin


def test_unknown_key_lists_what_exists():
    registry.register(Dummy())
    with pytest.raises(KeyError, match="dummy"):
        registry.get("nope")


def test_static_filter_excludes_network_plugins():
    registry.register(Dummy())
    assert registry.all_plugins() != []
    assert registry.all_plugins(static_only=True) == []


def test_discovery_does_not_load():
    plugin = Dummy()
    ctx = SourceContext(key=SourceKey("Gaia DR3", 1))
    descriptor = plugin.discover(ctx)
    assert descriptor.state is ProductState.UNAVAILABLE
    assert descriptor.detail == "nothing here"


def test_builtin_plugins_register_once():
    first = registry.load_builtin_plugins()
    second = registry.load_builtin_plugins()
    assert [p.key for p in first] == [p.key for p in second] == [
        "epoch_astrometry", "epoch_photometry",
    ]


def test_photometry_is_not_static_safe():
    """It needs a live archive request, so a static build must exclude it."""
    registry.load_builtin_plugins()
    static = [p.key for p in registry.all_plugins(static_only=True)]
    assert "epoch_astrometry" in static
    assert "epoch_photometry" not in static
