"""Live checks of every spectral archive (run with ``pytest -m network``).

HD 114762 is used because it is bright, nearby and has been observed for
decades: each archive below held spectra of it when the search was built
(2026-09-23). A failure here means an archive changed or is down, not
necessarily a bug -- the status says which.
"""

import pytest

from gaia_dr4_explorer.data import external_spectra as xs

pytestmark = pytest.mark.network

HD114762 = xs.SkyPosition(198.0792922757309, 17.517123799636206, 2017.5,
                          pmra_masyr=-582.03, pmdec_masyr=-0.71)

#: Archives known to hold spectra of HD 114762, and a minimum count seen live.
EXPECTED = {"MAST": 5, "ESO": 5, "CADC": 50, "CfA TDC": 100, "PolarBase": 5}


@pytest.fixture(scope="module")
def report():
    return xs.search_external_spectra(HD114762, timeout_s=180)


@pytest.mark.parametrize("archive,minimum", sorted(EXPECTED.items()))
def test_archive_finds_hd114762(report, archive, minimum):
    res = {r.archive: r for r in report.results}[archive]
    assert res.status is xs.SearchStatus.FOUND, f"{archive}: {res.status} {res.error}"
    assert res.n_spectra >= minimum


def test_every_archive_answers(report):
    bad = {r.archive: (r.status.value, r.error) for r in report.results
           if r.status in (xs.SearchStatus.FAILED, xs.SearchStatus.TIMEOUT)}
    assert not bad, bad


def test_matches_lie_on_the_path_of_the_star(report):
    """Every archive's matches lie close to the star's 1975-2027 path, whether
    it stores positions at the epoch of observation or at J2000 (measured
    2026-09-23: per-archive medians 0.01-0.3 arcsec)."""
    seps = [r.separation_arcsec for r in report.records()
            if r.separation_arcsec == r.separation_arcsec]
    assert seps, "no record had a position"
    assert sorted(seps)[len(seps) // 2] < 1.0
    by_archive = {}
    for r in report.records():
        if r.separation_arcsec == r.separation_arcsec:
            by_archive.setdefault(r.archive, []).append(r.separation_arcsec)
    for archive, s in by_archive.items():
        assert sorted(s)[len(s) // 2] < 1.0, f"{archive}: median {sorted(s)[len(s) // 2]:.2f}"


def test_an_empty_polarbase_answer_is_none_not_failed():
    """HD 183633 has no PolarBase spectra; the service says OK with no table."""
    hd183633 = xs.SkyPosition(292.77959920112164, -16.70246576342942, 2017.5,
                              pmra_masyr=2.90, pmdec_masyr=-0.29)
    transport = xs.LiveTransport(timeout_s=60)
    rep = xs.search_external_spectra(
        hd183633, [xs.SsaSearch(transport, name="PolarBase", ivoid=xs.SSA_IVOIDS["PolarBase"])],
        timeout_s=90)
    assert rep.results[0].status is xs.SearchStatus.NONE, rep.results[0].error
