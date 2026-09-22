import zipfile

import pytest

from gaia_dr4_explorer.data import PreReleaseError, PreReleaseProvider


def test_missing_archive_without_network_is_an_error(offline_config):
    provider = PreReleaseProvider(offline_config)
    with pytest.raises(PreReleaseError, match="network access is disabled"):
        provider.ensure_archive()


def test_configured_local_archive_that_does_not_exist(tmp_path, offline_config):
    from dataclasses import replace

    cfg = replace(offline_config, local_prerelease_zip=tmp_path / "nope.zip")
    with pytest.raises(PreReleaseError, match="does not exist"):
        PreReleaseProvider(cfg).ensure_archive()


def test_corrupt_zip_is_reported(tmp_path, offline_config):
    from dataclasses import replace

    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"this is not a zip file")
    cfg = replace(offline_config, local_prerelease_zip=bad, cache_dir=tmp_path / "cache")
    with pytest.raises(PreReleaseError, match="not a readable ZIP"):
        PreReleaseProvider(cfg).ensure_extracted()


def test_member_missing_from_zip(tmp_path, offline_config):
    from dataclasses import replace

    z = tmp_path / "empty.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "nothing here")
    cfg = replace(offline_config, local_prerelease_zip=z, cache_dir=tmp_path / "cache")
    with pytest.raises(PreReleaseError, match="not found in"):
        PreReleaseProvider(cfg).ensure_extracted()


def test_malformed_votable_is_reported(tmp_path, offline_config):
    from dataclasses import replace

    from gaia_dr4_explorer.config import PRERELEASE_MEMBER

    z = tmp_path / "bad_xml.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr(PRERELEASE_MEMBER, "<VOTABLE><not-really></VOTABLE>")
    cfg = replace(offline_config, local_prerelease_zip=z, cache_dir=tmp_path / "cache")
    with pytest.raises(PreReleaseError):
        PreReleaseProvider(cfg).load_table()


def test_unknown_source_id_raises_keyerror(config):
    provider = PreReleaseProvider(config)
    with pytest.raises(KeyError, match="not in the prerelease"):
        provider.raw_table_for(123456789)
