import pytest

from gaia_dr4_explorer.config import CacheLayout, _slug
from gaia_dr4_explorer.domain import SourceKey


def test_release_is_part_of_every_cache_path(tmp_path):
    layout = CacheLayout(tmp_path)
    dr3 = layout.raw("Gaia DR3", 4318465066420528000)
    dr4 = layout.raw("Gaia DR4_RC3", 4318465066420528000)
    assert dr3 != dr4, "the same integer under two releases must not collide"
    assert "4318465066420528000" in str(dr3)


def test_slug_makes_release_path_safe():
    assert _slug("Gaia DR4_RC3") == "Gaia-DR4_RC3"
    assert "/" not in _slug("a/b")


def test_source_key_rejects_nonsense():
    with pytest.raises(ValueError):
        SourceKey(release="", source_id=1)
    with pytest.raises(ValueError):
        SourceKey(release="Gaia DR4_RC3", source_id=0)


def test_source_key_distinguishes_releases():
    a = SourceKey("Gaia DR3", 4318465066420528000)
    b = SourceKey("Gaia DR4_RC3", 4318465066420528000)
    assert a != b
    assert a.cache_token != b.cache_token
    assert hash(a) != hash(b)
