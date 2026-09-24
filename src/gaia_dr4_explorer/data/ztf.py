"""ZTF light curves from the IRSA API, for any source, with or without Gaia photometry.

The query follows the pattern of the author's existing ZTF scripts (e.g.
``qso_agn_dwarfs/scripts/download_ztf_api_lightcurves.py``,
``gaia_euclid_lens/scripts/fetch_ztf_lightcurves.py``):

* endpoint ``https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves``;
* ``POS=CIRCLE ra dec radius_deg``, ``BANDNAME=g,r,i``, ``FORMAT=CSV``,
  ``COLLECTION=ztf_dr24`` (pinned, so a result says which release it is);
* an empty answer or ``No rows returned`` means no data; XML means an error;
* clean points are ``catflags == 0``. Flagged points are kept and shown, not
  deleted.

What is different here: the star moves. ZTF positions are at each epoch
(2018-2025), so the search is centred on the source moved to mid-ZTF, with a
radius of 2 arcsec plus the proper motion over half the ZTF span, and every
point carries its closest approach to the star's path.

Observed when this was built (2026-09-24): IRSA sends no CORS header, so a
web page cannot query it; queries take 4-90 s and occasionally over 120 s.
Bright stars saturate: Gaia BH3 (G = 11.2) and HD 183633 (G = 9.0) return
nothing, and HD 114762 (G = 7.2) only ten g-band points 2024-2025, all
flagged, 4-6 arcsec off its path, at 10.8-11.5 mag.
"""

from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gaia_dr4_explorer.data.external_spectra import SkyPosition, Transport

ZTF_API_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"
DEFAULT_COLLECTION = "ztf_dr24"
DEFAULT_BANDS = "g,r,i"
#: Positional tolerance before proper motion (the author's scripts use 1.5-2).
BASE_RADIUS_ARCSEC = 2.0
#: Span of ZTF data in DR24, Julian years (MJD 58194-60861 seen in queries).
ZTF_SPAN_JYEAR = (2018.2, 2025.5)
BANDS = {"zg": "g", "zr": "r", "zi": "i"}

#: Columns kept: enough to plot, judge and trace every point.
KEEP_COLUMNS = ("oid", "mjd", "mag", "magerr", "catflags", "filtercode", "ra", "dec",
                "limitmag", "field", "ccdid", "qid", "exptime", "airmass")


class ZtfError(RuntimeError):
    """IRSA answered, but not with a light curve."""


@dataclass(frozen=True)
class ZtfLightCurve:
    """All ZTF points near a source, with how they were obtained."""

    frame: pd.DataFrame
    ra_deg: float
    dec_deg: float
    radius_arcsec: float
    collection: str
    retrieved_at: str
    params: dict[str, str] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return len(self.frame)

    @property
    def n_clean(self) -> int:
        return int(self.frame["clean"].sum()) if len(self.frame) else 0

    def summary(self) -> pd.DataFrame:
        """Per band, on clean points: count, median, rms and reduced chi^2."""
        rows = []
        for band in ("g", "r", "i"):
            d = self.frame[self.frame["band"] == band] if len(self.frame) else self.frame
            clean = d[d["clean"]] if len(d) else d
            m = clean["mag"].to_numpy(float) if len(clean) else np.array([])
            e = clean["magerr"].to_numpy(float) if len(clean) else np.array([])
            ok = np.isfinite(m) & np.isfinite(e) & (e > 0)
            m, e = m[ok], e[ok]
            row = {"band": band, "points": len(d), "clean": int(m.size),
                   "median mag": np.nan, "rms [mag]": np.nan, "median σ [mag]": np.nan,
                   "χ²/dof": np.nan}
            if m.size >= 2:
                w = 1 / e**2
                mean = float(np.sum(w * m) / np.sum(w))
                row.update({"median mag": float(np.median(m)), "rms [mag]": float(np.std(m, ddof=1)),
                            "median σ [mag]": float(np.median(e)),
                            "χ²/dof": float(np.sum(((m - mean) / e) ** 2) / (m.size - 1))})
            rows.append(row)
        return pd.DataFrame(rows)

    # -------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {"ra_deg": self.ra_deg, "dec_deg": self.dec_deg,
                "radius_arcsec": self.radius_arcsec, "collection": self.collection,
                "retrieved_at": self.retrieved_at, "params": self.params,
                "csv": self.frame[[c for c in KEEP_COLUMNS if c in self.frame]].to_csv(index=False)}

    @classmethod
    def from_dict(cls, d: dict[str, Any], position: SkyPosition | None = None) -> ZtfLightCurve:
        frame = parse_ztf_csv(d["csv"], position)
        return cls(frame, d["ra_deg"], d["dec_deg"], d["radius_arcsec"], d["collection"],
                   d["retrieved_at"], d.get("params", {}))


def query_params(position: SkyPosition, *, collection: str = DEFAULT_COLLECTION,
                 bands: str = DEFAULT_BANDS) -> tuple[dict[str, str], float, float, float]:
    """IRSA parameters for this source; also returns (ra, dec, radius) used."""
    mid = 0.5 * (ZTF_SPAN_JYEAR[0] + ZTF_SPAN_JYEAR[1])
    ra, dec = position.at(mid)
    pm = position.total_pm_masyr
    half = 0.5 * (ZTF_SPAN_JYEAR[1] - ZTF_SPAN_JYEAR[0])
    radius = BASE_RADIUS_ARCSEC + (pm * half / 1000.0 if np.isfinite(pm) else 0.0)
    params = {"POS": f"CIRCLE {ra:.6f} {dec:.6f} {radius / 3600.0:.6f}",
              "BANDNAME": bands, "FORMAT": "CSV"}
    if collection:
        params["COLLECTION"] = collection
    return params, ra, dec, radius


def parse_ztf_csv(text: str, position: SkyPosition | None = None) -> pd.DataFrame:
    """IRSA's CSV as a frame with ``band``, ``clean`` and ``sep_arcsec`` added.

    ``sep_arcsec`` is each point's closest approach to the source's path;
    NaN when no position is given. Missing values stay NaN.
    """
    stripped = text.strip()
    columns = list(KEEP_COLUMNS) + ["band", "clean", "sep_arcsec"]
    if not stripped or "No rows returned" in stripped:
        return pd.DataFrame(columns=columns)
    if stripped.startswith("<"):
        raise ZtfError(f"IRSA returned XML, not a light curve: {stripped[:200]!r}")
    frame = pd.read_csv(io.StringIO(text))
    if "filtercode" not in frame.columns or "mjd" not in frame.columns:
        raise ZtfError(f"unexpected columns: {', '.join(frame.columns[:12])}")
    frame = frame[[c for c in KEEP_COLUMNS if c in frame.columns]].copy()
    frame["band"] = frame["filtercode"].map(BANDS)
    frame["clean"] = frame["catflags"].fillna(-1).astype("int64") == 0
    if position is not None:
        frame["sep_arcsec"] = [position.separation_arcsec(a, b)
                               for a, b in zip(frame["ra"], frame["dec"], strict=True)]
    else:
        frame["sep_arcsec"] = np.nan
    frame["oid"] = frame["oid"].astype("int64")
    return frame.sort_values("mjd", kind="stable").reset_index(drop=True)


class ZtfProvider:
    """Fetches ZTF light curves and keeps them.

    Parameters
    ----------
    cache_dir : Path, optional
        Where fetched light curves are kept: one immutable, time-stamped file
        per fetch, keyed by release and source_id; the newest is shown.
    snapshot : dict, optional
        Shipped light curves for the prerelease sources (browser build).
    """

    SNAPSHOT_RESOURCE = "ztf_lightcurves.json"

    def __init__(self, transport: Transport | None = None, *, cache_dir: Path | None = None,
                 release: str = "Gaia DR4_RC3", allow_network: bool = True,
                 snapshot: dict | None = None, collection: str = DEFAULT_COLLECTION) -> None:
        self.transport = transport
        self.cache_dir = cache_dir
        self.release = release
        self.allow_network = allow_network
        self.collection = collection
        self._snapshot = snapshot

    # -------------------------------------------------------------- query

    def fetch(self, position: SkyPosition, *, source_id: int | None = None) -> ZtfLightCurve:
        """Query IRSA now; keep the answer when ``source_id`` is given."""
        if not self.allow_network or self.transport is None:
            raise ZtfError("network access is off")
        params, ra, dec, radius = query_params(position, collection=self.collection)
        text = self.transport.get_text(ZTF_API_URL, params)
        lc = ZtfLightCurve(parse_ztf_csv(text, position), ra, dec, radius, self.collection,
                           datetime.now(UTC).isoformat(timespec="seconds"), params)
        if source_id is not None and self.cache_dir is not None:
            self._save(source_id, lc)
        return lc

    # ------------------------------------------------------------ results

    def latest(self, source_id: int, position: SkyPosition | None = None
               ) -> tuple[ZtfLightCurve | None, str]:
        """The newest light curve available without querying, and its origin."""
        saved = self._newest_saved(source_id)
        if saved is not None:
            return ZtfLightCurve.from_dict(json.loads(saved.read_text()), position), "saved"
        shipped = self.snapshot().get("sources", {}).get(str(int(source_id)))
        if shipped:
            return ZtfLightCurve.from_dict(shipped, position), "shipped"
        return None, ""

    def snapshot(self) -> dict:
        if self._snapshot is None:
            from importlib.resources import files

            res = files("gaia_dr4_explorer.resources") / self.SNAPSHOT_RESOURCE
            self._snapshot = json.loads(res.read_text()) if res.is_file() else {"sources": {}}
        return self._snapshot

    def _dir(self, source_id: int) -> Path:
        from gaia_dr4_explorer.config import _slug

        return Path(self.cache_dir) / "ztf" / _slug(self.release) / str(int(source_id))

    def _save(self, source_id: int, lc: ZtfLightCurve) -> None:
        d = self._dir(source_id)
        d.mkdir(parents=True, exist_ok=True)
        stamp = lc.retrieved_at.replace(":", "").replace("-", "")
        path = d / f"{stamp}.json"
        tmp = path.with_suffix(".part")
        tmp.write_text(json.dumps(lc.to_dict(), allow_nan=False))
        tmp.replace(path)

    def _newest_saved(self, source_id: int) -> Path | None:
        if self.cache_dir is None:
            return None
        d = self._dir(source_id)
        files = sorted(d.glob("*.json")) if d.is_dir() else []
        return files[-1] if files else None


def build_snapshot(positions: dict[int, SkyPosition], provider: ZtfProvider, *,
                   attempts: int = 3, pause_s: float = 5.0, max_workers: int = 4,
                   log=print) -> dict:
    """Fetch every source, four at a time as the author's scripts do, with
    retries (IRSA can take over two minutes). Failures are recorded."""
    from concurrent.futures import ThreadPoolExecutor

    def one(item):
        sid, pos = item
        for attempt in range(attempts):
            try:
                lc = provider.fetch(pos)
                log(f"{sid}: {lc.n_points} points ({lc.n_clean} clean), "
                    f"radius {lc.radius_arcsec:.1f} arcsec")
                return sid, lc.to_dict(), None
            except Exception as exc:  # recorded, never silently dropped
                if attempt == attempts - 1:
                    log(f"{sid}: FAILED {exc}")
                    return sid, None, f"{type(exc).__name__}: {exc}"[:300]
                time.sleep(pause_s * (attempt + 1))
        return sid, None, "no attempt made"

    out = {"created_at": datetime.now(UTC).isoformat(timespec="seconds"),
           "collection": provider.collection,
           "note": "ZTF light curves from IRSA; regenerate with "
                   "`gaia-dr4-explorer snapshot-ztf`.",
           "sources": {}, "failed": {}}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for sid, data, error in pool.map(one, sorted(positions.items())):
            if data is not None:
                out["sources"][str(sid)] = data
            else:
                out["failed"][str(sid)] = error
    return out
