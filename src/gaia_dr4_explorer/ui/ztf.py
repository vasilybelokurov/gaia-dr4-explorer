"""The ZTF section of the Photometry tab.

Shown for every source, whether or not Gaia published photometry for it.
The query runs in :mod:`gaia_dr4_explorer.data.ztf`; this module does no I/O.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd
import panel as pn
import param

from gaia_dr4_explorer.data.ztf import ZtfLightCurve
from gaia_dr4_explorer.products.photometry import plots

EXPLANATION = (
    "ZTF DR light curves from IRSA (PSF-fit magnitudes in ZTF g, r, i). "
    "<b>Clean</b> points have <code>catflags</code> = 0; flagged points are kept and "
    "can be shown. The search is centred on the star moved to mid-ZTF, with a radius "
    "of 2″ plus its proper motion over half the ZTF span; <b>sep.</b> in the hover is "
    "each point's closest approach to the star's path, so a neighbour stands out. "
    "Stars brighter than about G ≈ 12 saturate in ZTF: expect no points, or flagged "
    "ones at wrong magnitudes. ZTF and Gaia magnitudes are in different systems."
)

_ORIGINS = {"shipped": "Light curve shipped with the app, fetched",
            "saved": "Saved light curve, fetched", "live": "Fetched"}

FETCH_LABEL = "Fetch ZTF light curve"


class ZtfView(param.Parameterized):
    """One ZTF light curve: summary, plot and provenance."""

    show_flagged = param.Boolean(default=True, label="Show flagged points (catflags ≠ 0)")

    def __init__(self, lc: ZtfLightCurve, origin: str, **params) -> None:
        super().__init__(**params)
        self.lc = lc
        self.origin = origin

    @param.depends("show_flagged")
    def plot(self):
        return pn.pane.HoloViews(plots.ztf_light_curve(self.lc.frame, show_flagged=self.show_flagged),
                                 sizing_mode="stretch_width")

    def panel(self) -> pn.Column:
        lc = self.lc
        when = lc.retrieved_at.replace("T", " ").replace("+00:00", " UTC")
        n_oid = lc.frame["oid"].nunique() if lc.n_points else 0
        head = (f"<div style='font-size:12px;color:#444;margin:4px 0'><b>{lc.n_points}</b> "
                f"ZTF points (<b>{lc.n_clean}</b> clean) from {n_oid} ZTF object"
                f"{'' if n_oid == 1 else 's'}; {lc.collection}. "
                f"{_ORIGINS.get(self.origin, self.origin)} {when}. Searched "
                f"{lc.radius_arcsec:.1f}″ around ({lc.ra_deg:.5f}, {lc.dec_deg:+.5f}).</div>")
        items = [pn.pane.HTML(head)]
        if lc.n_points:
            items.append(pn.widgets.Tabulator(
                _summary(lc.summary()), disabled=True, show_index=False, width=620,
                layout="fit_data"))
            items.append(pn.widgets.Checkbox.from_param(self.param.show_flagged, width=320))
            items.append(self.plot)
        return pn.Column(*items, sizing_mode="stretch_width")


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ("median mag", "rms [mag]", "median σ [mag]", "χ²/dof"):
        out[col] = [None if (v is None or (isinstance(v, float) and math.isnan(v)))
                    else round(float(v), 3) for v in out[col]]
    return out


def ztf_section(latest: ZtfLightCurve | None, fetch: Callable[[], ZtfLightCurve] | None = None,
                *, origin: str = "shipped") -> pn.Column:
    """The ZTF section. Nothing is queried until the button is pressed.

    Parameters
    ----------
    latest : ZtfLightCurve or None
        The newest light curve available without querying (saved or shipped).
    fetch : callable, optional
        Queries IRSA now (desktop app). Absent in the browser build: IRSA sends
        no CORS header, so a web page cannot query it.
    """
    out = pn.Column(sizing_mode="stretch_width")
    if latest is not None:
        out.objects = [ZtfView(latest, origin).panel()]
    elif fetch is None:
        out.objects = [pn.pane.HTML(
            "<div style='font-size:12px;color:#666'>No ZTF light curve is shipped for this "
            "source, and a web page cannot query IRSA. Run the app locally to fetch one.</div>")]
    else:
        out.objects = [pn.pane.HTML(
            "<div style='font-size:12px;color:#666'>Not fetched yet. IRSA usually answers in "
            "5–90 s, occasionally longer.</div>")]
    items = [
        pn.pane.HTML("<h3 style='margin:18px 0 4px 0'>ZTF light curve</h3>"),
        pn.pane.HTML(f"<div style='font-size:11px;color:#555;max-width:900px'>{EXPLANATION}</div>"),
    ]
    if fetch is not None:
        button = pn.widgets.Button(
            name=FETCH_LABEL if latest is None else f"{FETCH_LABEL} (fetch again)",
            button_type="primary" if latest is None else "default", width=300)

        def run(_event) -> None:
            button.loading = True
            try:
                out.objects = [ZtfView(fetch(), "live").panel()]
            except Exception as exc:
                out.objects = [pn.pane.HTML(
                    f"<div style='color:#c0392b'>ZTF query failed: "
                    f"{type(exc).__name__}: {str(exc)[:300]}</div>")]
            finally:
                button.loading = False

        button.on_click(run)
        items.append(button)
    items.append(out)
    return pn.Column(*items, sizing_mode="stretch_width")
