"""ZTF light curves, offline. The CSV has the columns IRSA returned on 2026-09-24."""

import json
import math

import numpy as np
import pytest

from gaia_dr4_explorer.data import external_spectra as xs
from gaia_dr4_explorer.data import ztf

HEADER = ("oid,expid,hjd,mjd,mag,magerr,catflags,filtercode,ra,dec,chi,sharp,filefracday,"
          "field,ccdid,qid,limitmag,magzp,magzprms,clrcoeff,clrcounc,exptime,airmass,programid")

QSO = xs.SkyPosition(44.8591, 4.4937, 2017.5, pmra_masyr=0.0, pmdec_masyr=0.0)
HD114762 = xs.SkyPosition(198.0792922757309, 17.517123799636206, 2017.5,
                          pmra_masyr=-582.03, pmdec_masyr=-0.71)


def row(oid, mjd, mag, err, flags, band, ra=44.8591, dec=4.4937):
    return (f"{oid},1,0,{mjd},{mag},{err},{flags},{band},{ra},{dec},1,0,0,"
            f"400,1,1,21.0,26,0.01,0,0,30,1.2,1")


def csv(*rows):
    return HEADER + "\n" + "\n".join(rows) + "\n"


class Irsa(xs.Transport):
    def __init__(self, text="", fail=False):
        self.text, self.fail, self.calls = text, fail, []

    def get_text(self, url, params):
        self.calls.append((url, params))
        if self.fail:
            raise TimeoutError("IRSA did not answer")
        return self.text


# ---------------------------------------------------------------- query


def test_query_is_centred_mid_ztf_with_a_proper_motion_radius():
    params, ra, dec, radius = ztf.query_params(HD114762)
    mid = 0.5 * sum(ztf.ZTF_SPAN_JYEAR)
    assert (ra, dec) == pytest.approx(HD114762.at(mid))
    half = 0.5 * (ztf.ZTF_SPAN_JYEAR[1] - ztf.ZTF_SPAN_JYEAR[0])
    assert radius == pytest.approx(2.0 + math.hypot(582.03, 0.71) * half / 1000)
    assert params["POS"] == f"CIRCLE {ra:.6f} {dec:.6f} {radius / 3600:.6f}"
    assert params["COLLECTION"] == "ztf_dr24" and params["FORMAT"] == "CSV"
    assert params["BANDNAME"] == "g,r,i"


def test_unknown_proper_motion_uses_the_base_radius_only():
    _, ra, dec, radius = ztf.query_params(xs.SkyPosition(10.0, 20.0, 2017.5))
    assert radius == 2.0 and (ra, dec) == (10.0, 20.0)


# ---------------------------------------------------------------- parse


def test_no_rows_is_an_empty_light_curve_not_an_error():
    for text in ("", "   ", "No rows returned"):
        assert len(ztf.parse_ztf_csv(text, QSO)) == 0


def test_xml_is_an_error():
    with pytest.raises(ztf.ZtfError, match="XML"):
        ztf.parse_ztf_csv("<?xml version='1.0'?><error/>", QSO)


def test_bands_clean_flags_and_separation():
    f = ztf.parse_ztf_csv(csv(
        row(1, 58500.1, 19.6, 0.10, 0, "zg"),
        row(1, 58400.1, 19.5, 0.10, 32768, "zg"),
        row(2, 58450.1, 19.2, 0.08, 0, "zr", dec=4.4937 + 1 / 3600),
    ), QSO)
    assert f["mjd"].is_monotonic_increasing
    assert f["band"].tolist() == ["g", "r", "g"]
    assert f["clean"].tolist() == [False, True, True]
    assert f.loc[f["oid"] == 2, "sep_arcsec"].iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_missing_magnitudes_stay_missing():
    f = ztf.parse_ztf_csv(csv(row(1, 58500.1, "", "", 0, "zg")), QSO)
    assert np.isnan(f["mag"].iloc[0])


def test_summary_uses_clean_points_only():
    rows = [row(1, 58500 + i, 19.0 + 0.1 * (i % 2), 0.05, 0, "zr") for i in range(10)]
    rows.append(row(1, 58600, 12.0, 0.05, 256, "zr"))           # flagged outlier
    lc = ztf.ZtfLightCurve(ztf.parse_ztf_csv(csv(*rows), QSO), 0, 0, 2, "ztf_dr24", "t")
    r = lc.summary().set_index("band").loc["r"]
    assert r["points"] == 11 and r["clean"] == 10
    assert r["median mag"] == pytest.approx(19.05)
    assert r["χ²/dof"] == pytest.approx(np.sum((np.array([0, 0.1] * 5) - 0.05) ** 2 / 0.05**2) / 9)


# ------------------------------------------------------------- provider


def test_fetch_is_saved_time_stamped_and_shown_next_time(tmp_path):
    irsa = Irsa(csv(row(1, 58500.1, 19.6, 0.1, 0, "zg")))
    p = ztf.ZtfProvider(irsa, cache_dir=tmp_path, snapshot={"sources": {}})
    assert p.latest(7) == (None, "")
    lc = p.fetch(QSO, source_id=7)
    assert lc.n_points == 1 and irsa.calls[0][0] == ztf.ZTF_API_URL
    again, origin = p.latest(7, QSO)
    assert origin == "saved" and again.n_points == 1 and again.collection == "ztf_dr24"
    saved = list((tmp_path / "ztf" / "Gaia-DR4_RC3" / "7").glob("*.json"))
    assert len(saved) == 1, "one immutable file per fetch, keyed by release and source_id"


def test_saved_wins_over_shipped(tmp_path):
    shipped_lc = ztf.ZtfLightCurve(ztf.parse_ztf_csv(csv(row(1, 58500, 19, 0.1, 0, "zg"))),
                                   1, 2, 2, "ztf_dr24", "2026-09-24T00:00:00+00:00")
    snap = {"sources": {"7": shipped_lc.to_dict()}}
    p = ztf.ZtfProvider(Irsa(csv(row(1, 58500, 19, 0.1, 0, "zg"), row(1, 58501, 19, 0.1, 0, "zg"))),
                        cache_dir=tmp_path, snapshot=snap)
    assert p.latest(7)[1] == "shipped"
    p.fetch(QSO, source_id=7)
    lc, origin = p.latest(7)
    assert origin == "saved" and lc.n_points == 2


def test_network_off_refuses_to_query():
    with pytest.raises(ztf.ZtfError, match="network"):
        ztf.ZtfProvider(Irsa(), allow_network=False).fetch(QSO)


def test_round_trip_keeps_points_and_provenance():
    lc = ztf.ZtfProvider(Irsa(csv(row(1, 58500.1, 19.6, 0.1, 0, "zg"),
                                  row(2, 58501.1, 19.7, 0.1, 256, "zr")))).fetch(QSO)
    back = ztf.ZtfLightCurve.from_dict(json.loads(json.dumps(lc.to_dict())), QSO)
    assert back.n_points == 2 and back.n_clean == 1
    assert back.params["COLLECTION"] == "ztf_dr24"


def test_snapshot_records_failures_instead_of_dropping_them():
    p = ztf.ZtfProvider(Irsa(fail=True))
    snap = ztf.build_snapshot({1: QSO}, p, attempts=2, pause_s=0, log=lambda *_: None)
    assert snap["sources"] == {} and "TimeoutError" in snap["failed"]["1"]


# ----------------------------------------------------------------- plot/UI


@pytest.fixture
def bokeh():
    import holoviews as hv

    hv.extension("bokeh")


def test_plot_draws_clean_and_optionally_flagged_points(bokeh):
    import holoviews as hv

    from gaia_dr4_explorer.products.photometry.plots import ztf_light_curve

    f = ztf.parse_ztf_csv(csv(row(1, 58500.1, 19.6, 0.1, 0, "zg"),
                              row(1, 58501.1, 12.0, 0.1, 256, "zg")), QSO)
    labels = lambda o: [e.label for e in o if isinstance(e, hv.Scatter)]  # noqa: E731
    assert labels(ztf_light_curve(f)) == ["ZTF g (1)", "ZTF g flagged (1)"]
    assert labels(ztf_light_curve(f, show_flagged=False)) == ["ZTF g (1)"]
    assert isinstance(ztf_light_curve(f.iloc[0:0]), hv.Text)


def test_section_never_queries_until_pressed_and_browser_has_no_button(bokeh):
    import panel as pn

    from gaia_dr4_explorer.ui.ztf import FETCH_LABEL, ztf_section

    calls = []
    section = ztf_section(None, lambda: calls.append(1))
    button = next(o for o in section.objects if isinstance(o, pn.widgets.Button))
    assert button.name == FETCH_LABEL and calls == []
    browser = ztf_section(None, None)
    assert not any(isinstance(o, pn.widgets.Button) for o in browser.objects)
    assert "cannot query IRSA" in browser.objects[-1].objects[0].object
