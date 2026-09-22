"""State transitions, not pixels."""

import pytest
from tests.conftest import PRERELEASE_ZIP

from gaia_dr4_explorer.config import AppConfig
from gaia_dr4_explorer.data import PreReleaseProvider
from gaia_dr4_explorer.products import registry
from gaia_dr4_explorer.ui.state import AppState

pytestmark = pytest.mark.integration

BH3 = 4318465066420528000
HD183633 = 4181040337841125632


@pytest.fixture(scope="module")
def provider(tmp_path_factory):
    cfg = AppConfig(
        cache_dir=tmp_path_factory.mktemp("cache"),
        local_prerelease_zip=PRERELEASE_ZIP,
        allow_network=False,
    )
    p = PreReleaseProvider(cfg)
    p.load_table()
    return p


@pytest.fixture
def plugin(provider):
    registry.clear()
    registry.load_builtin_plugins()
    return registry.get("epoch_astrometry").bind(provider)


@pytest.fixture
def state(provider):
    return AppState(provider=provider, source_id=BH3, release=provider.release())


def test_context_separates_page_metadata_from_measurements(state):
    ctx = state.context()
    assert ctx.common_name == "Gaia BH3"
    assert ctx.sample_category == "orbit"
    assert ctx.page_metadata["parallax_mas"] == 2.0
    assert ctx.measured == {}, "page metadata must not leak into measured values"
    assert ctx.is_release_candidate


def test_payload_is_memoised_per_source(state, plugin):
    first = state.payload(plugin)
    assert state.payload(plugin) is first
    state.source_id = HD183633
    assert state.payload(plugin) is not first


def test_reload_drops_cached_products(state, plugin):
    first = state.payload(plugin)
    state.reset_source()
    assert state.payload(plugin) is not first


def test_fit_is_not_run_until_asked(state):
    assert not state.has_fit()
    result = state.fit()
    assert state.has_fit()
    assert state.fit() is result


def test_discover_reports_absent_source(state, plugin):
    state.source_id = BH3
    assert plugin.discover(state.context()).state.value == "available"
    state.source_id = 999999999999
    assert plugin.discover(state.context()).state.value == "unavailable"


def test_selecting_an_absent_source_is_rejected_not_crashed(state, plugin):
    """The shell's guard: an out-of-release id must not blank the view."""
    good = state.source_id
    assert plugin.discover(state.context()).state.value == "available"
    state.source_id = 1
    assert plugin.discover(state.context()).state.value == "unavailable"
    state.source_id = good
    assert plugin.discover(state.context()).state.value == "available"
