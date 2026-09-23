"""The "Spectra in other archives" section and the snapshot it shows."""

import panel as pn
import pytest

from gaia_dr4_explorer.data import external_spectra as xs
from gaia_dr4_explorer.data.catalog import reference_fits
from gaia_dr4_explorer.ui.external_spectra import (
    SEARCH_LABEL,
    external_spectra_section,
    records_frame,
    report_panel,
)

HD114762 = 3937211745905473024
BH3 = 4318465066420528000


@pytest.fixture(scope="module")
def shipped():
    return xs.ExternalSpectraProvider(allow_network=False)


def test_snapshot_covers_every_prerelease_source(shipped):
    for sid in reference_fits():
        report = shipped.bundled(sid)
        assert report is not None, sid
        assert {r.archive for r in report.results} == {
            "MAST", "ESO", "CADC", "CfA TDC", "FEROS", "PolarBase", "BeSS", "ELODIE"}
        # No archive failed or timed out when the snapshot was made; the only
        # gaps allowed are ELODIE searches skipped for want of a name.
        for r in report.results:
            assert r.status is not xs.SearchStatus.FAILED, (sid, r.archive, r.error)
            assert r.status is not xs.SearchStatus.TIMEOUT, (sid, r.archive)
            if r.status is xs.SearchStatus.SKIPPED:
                assert r.archive == "ELODIE"


def test_snapshot_finds_the_well_observed_stars(shipped):
    hd = {r.archive: r.n_spectra for r in shipped.bundled(HD114762).results}
    assert hd["CfA TDC"] >= 100 and hd["CADC"] >= 50 and hd["ESO"] >= 5
    bh3 = {r.archive: r.n_spectra for r in shipped.bundled(BH3).results}
    assert bh3["ESO"] >= 10, "Gaia BH3's UVES follow-up should be listed"


def test_records_table_keeps_missing_values_empty(shipped):
    frame = records_frame(shipped.bundled(HD114762), "ESO")
    assert len(frame) == 8
    assert frame["link"].str.startswith("https://").all()
    assert (frame["sep. [″]"] < 2).all(), "ESO matches lie on the star's path"


def test_panel_builds_and_flags_an_incomplete_search(shipped):
    report = shipped.bundled(2309425390592896)          # ELODIE skipped: no name
    html = report_panel(report, origin="Shipped search, run").objects[0].object
    assert "Incomplete" in html and "ELODIE" in html
    complete = shipped.bundled(HD114762)
    cards = [o for o in report_panel(complete, origin="x").objects if isinstance(o, pn.Card)]
    assert {c.title.split(" —")[0] for c in cards} == {
        "MAST", "ESO", "CADC", "CfA TDC", "PolarBase", "ELODIE"}
    assert all(c.collapsed for c in cards)


def test_browser_section_has_no_live_button(shipped):
    section = external_spectra_section(shipped.bundled(HD114762))
    assert not any(isinstance(o, pn.widgets.Button) for o in section.objects)


def test_live_button_replaces_the_shipped_result(shipped):
    calls = []
    fresh = shipped.bundled(BH3)

    def live():
        calls.append(1)
        return fresh

    section = external_spectra_section(shipped.bundled(HD114762), live)
    button = next(o for o in section.objects if isinstance(o, pn.widgets.Button))
    button.clicks += 1
    assert calls == [1]
    out = section.objects[-1]
    assert "Live search" in out.objects[0].objects[0].object
    assert button.name == f"{SEARCH_LABEL} (search again)"


def test_without_snapshot_or_network_it_says_so():
    section = external_spectra_section(None, None)
    assert "cannot query archives" in section.objects[-1].objects[0].object


def test_nothing_is_searched_until_the_button_is_pressed():
    calls = []
    section = external_spectra_section(None, lambda: calls.append(1))
    button = next(o for o in section.objects if isinstance(o, pn.widgets.Button))
    assert button.name == SEARCH_LABEL
    assert calls == [], "building the section must not start a search"
    assert "Not searched yet" in section.objects[-1].objects[0].object


def test_a_live_search_is_saved_and_shown_next_time(tmp_path, shipped):
    fresh = shipped.bundled(BH3)
    calls = []

    def fake_search(position, **kwargs):
        calls.append(kwargs)
        return fresh

    archives = xs.ExternalSpectraProvider(cache_dir=tmp_path, search=fake_search,
                                          snapshot={"sources": {}})
    assert archives.latest(123) == (None, "")
    pos = xs.SkyPosition(1.0, 2.0, 2017.5)
    archives.search(pos, source_id=123)
    report, origin = archives.latest(123)
    assert origin == "saved" and report.results[1].n_spectra == fresh.results[1].n_spectra
    # The cache key carries the release (CLAUDE.md invariant 1).
    assert (tmp_path / "external_spectra" / "Gaia-DR4_RC3" / "123.json").is_file()


def test_saved_search_wins_over_the_shipped_one(tmp_path, shipped):
    archives = xs.ExternalSpectraProvider(cache_dir=tmp_path, search=lambda p, **k: shipped.bundled(BH3))
    assert archives.latest(HD114762)[1] == "shipped"
    archives.search(xs.SkyPosition(1.0, 2.0, 2017.5), source_id=HD114762)
    assert archives.latest(HD114762)[1] == "saved"


def test_nothing_is_saved_with_the_network_off(tmp_path):
    archives = xs.ExternalSpectraProvider(cache_dir=tmp_path, allow_network=False,
                                          snapshot={"sources": {}})
    report = archives.search(xs.SkyPosition(1.0, 2.0, 2017.5), source_id=5,
                             searches=[xs.MastSearch(xs.Transport())])
    assert report.results[0].status is xs.SearchStatus.SKIPPED
    assert archives.saved(5) is None


def test_an_unreadable_saved_search_is_treated_as_absent(tmp_path):
    archives = xs.ExternalSpectraProvider(cache_dir=tmp_path, snapshot={"sources": {}})
    path = tmp_path / "external_spectra" / "Gaia-DR4_RC3" / "7.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert archives.saved(7) is None
