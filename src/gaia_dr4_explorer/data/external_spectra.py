"""Discovery of spectra of one object in external archives.

This module only *finds* spectra: it asks each archive what it holds near a
position and returns metadata. Nothing is downloaded here; downloads happen
later, and only for archives the user selects.

Every archive answers with an explicit status -- found, none, failed, timed
out or skipped -- so a failed search is never reported as "no spectra".

Archives, and how each is reached (all public, no credentials):

=============  =========================================================
MAST           Mashup ``Mast.Caom.Filtered.Position``: every public
               spectrum at MAST (HST, IUE, FUSE, HLSP, SDSS/APOGEE, ...)
ESO            TAP, ``ivoa.ObsCore``
CADC           TAP, ``caom2`` tables (``ivoa.ObsCore`` there timed out
               at 60 s in testing; the ``caom2`` footprint search returns
               in ~2 s)
CfA TDC        TAP, ``ivoa.ObsCore`` (plain ``http://``)
FEROS (GAVO)   SSA, endpoint resolved from the VO registry
PolarBase      SSA, endpoint resolved from the VO registry
BeSS           SSA, endpoint resolved from the VO registry
ELODIE         name search only; skipped when no name is known
=============  =========================================================

The queries follow ``../hst_archive`` (``hst_archive/optical.py``,
``hst_archive/api.py``), widened from its optical and HST-only selections to
every spectrum each archive holds.

Positions and proper motion
---------------------------
A fast-moving star is far from its Gaia position in old data: HD 114762
(-582 mas/yr) has moved ~16 arcsec since its 1989 IUE spectrum, which MAST
records 38 arcsec from the Gaia J2017.5 position. So the search radius grows
with proper motion over the span archives cover.

Archives do not agree on *which* epoch a position refers to. Checked live for
HD 114762: MAST and ESO give the position on the night (median 0.2 and 0.5
arcsec once the star is moved to that date); CfA TDC and PolarBase give the
J2000 catalogue position (0.01-0.03 arcsec from where the star was in 2000,
but 6-8 arcsec from where it was on the night); CADC mixes both. A separation
at the observation date would therefore be wrong for half the archives. Each
record's separation is instead its closest approach to the source's path over
1975-2027, which is small for a genuine match whichever convention the archive
uses (medians 0.01-0.3 arcsec across all five archives).
"""

from __future__ import annotations

import enum
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import Any

import numpy as np

# ------------------------------------------------------------------ endpoints

MAST_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
MAST_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
ESO_TAP = "https://archive.eso.org/tap_obs"
CADC_TAP = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/argus"
CADC_DATALINK = "https://ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/caom2ops/datalink?ID="
CFA_TAP = "http://oirsa.cfa.harvard.edu:8080/tap"
ELODIE_SEARCH = "http://atlas.obs-hp.fr/elodie/fE.cgi"
ELODIE_GET = (
    "http://atlas.obs-hp.fr/elodie/E.cgi?&c=i&o=elodie:{dataset}/{imanum}"
    "&z=s1d&a=mime:application/x-fits"
)
SSA_IVOIDS = {
    "FEROS": "ivo://org.gavo.dc/feros/q/ssa",
    "PolarBase": "ivo://ov-gso/ssap/polarbase",
    "BeSS": "ivo://vopdc.obspm/lesia/bestars/bess",
}

#: Span of epochs archive spectra can come from: the earliest digitised plate
#: and IUE archives start in the late 1970s. Used only to size search radii.
ARCHIVE_EPOCH_SPAN_JYEAR = (1975.0, 2027.0)



# --------------------------------------------------------------- domain types


@dataclass(frozen=True)
class SkyPosition:
    """Where to look, with enough astrometry to follow the source in time.

    Parameters
    ----------
    ra_deg, dec_deg : float
        ICRS position at ``epoch_jyear``.
    epoch_jyear : float
        Reference epoch of the position, Julian years (DR4: 2017.5).
    pmra_masyr, pmdec_masyr : float
        Proper motion; ``pmra`` includes cos(dec). NaN when unknown, which
        disables the proper-motion allowance rather than assuming zero.
    name : str
        Optional object name, for archives that can only search by name.
    """

    ra_deg: float
    dec_deg: float
    epoch_jyear: float
    pmra_masyr: float = float("nan")
    pmdec_masyr: float = float("nan")
    name: str = ""

    @property
    def total_pm_masyr(self) -> float:
        return math.hypot(self.pmra_masyr, self.pmdec_masyr)

    def search_radius_arcsec(
        self, base_arcsec: float, span: tuple[float, float] = ARCHIVE_EPOCH_SPAN_JYEAR
    ) -> float:
        """``base`` plus the largest proper-motion displacement over ``span``."""
        pm = self.total_pm_masyr
        if not np.isfinite(pm):
            return float(base_arcsec)
        dt = max(abs(span[0] - self.epoch_jyear), abs(span[1] - self.epoch_jyear))
        return float(base_arcsec + pm * dt / 1000.0)

    def at(self, jyear: float) -> tuple[float, float]:
        """Position at ``jyear`` by linear proper motion (arcsec-scale accuracy).

        Returns the reference position when the proper motion is unknown.
        """
        if not (np.isfinite(self.pmra_masyr) and np.isfinite(self.pmdec_masyr)):
            return self.ra_deg, self.dec_deg
        dt = jyear - self.epoch_jyear
        cosd = math.cos(math.radians(self.dec_deg))
        dra = self.pmra_masyr * dt / 3.6e6 / max(cosd, 1e-9)
        ddec = self.pmdec_masyr * dt / 3.6e6
        return (self.ra_deg + dra) % 360.0, self.dec_deg + ddec

    def separation_arcsec(
        self, ra_deg: float, dec_deg: float,
        span: tuple[float, float] = ARCHIVE_EPOCH_SPAN_JYEAR,
    ) -> float:
        """Closest approach of (ra, dec) to the source's path over ``span``.

        Archives record either the position on the night or a catalogue (often
        J2000) position; the closest approach is right for both. With unknown
        proper motion it is the plain distance to the reference position.
        Computed in the tangent plane at the reference position, which is exact
        to far better than an arcsec over the arcminute scales involved.
        """
        if not (np.isfinite(ra_deg) and np.isfinite(dec_deg)):
            return float("nan")
        cosd = math.cos(math.radians(self.dec_deg))
        dra = ((ra_deg - self.ra_deg + 180.0) % 360.0 - 180.0) * cosd * 3.6e6   # mas
        ddec = (dec_deg - self.dec_deg) * 3.6e6
        vx, vy = self.pmra_masyr, self.pmdec_masyr
        if not (np.isfinite(vx) and np.isfinite(vy)) or (vx == 0 and vy == 0):
            return _angular_distance_arcsec(self.ra_deg, self.dec_deg, ra_deg, dec_deg)
        t = (dra * vx + ddec * vy) / (vx * vx + vy * vy)            # yr from epoch
        t = min(max(t, span[0] - self.epoch_jyear), span[1] - self.epoch_jyear)
        return math.hypot(dra - vx * t, ddec - vy * t) / 1000.0


@dataclass(frozen=True)
class SpectrumRecord:
    """One spectrum an archive says it holds. Missing values stay NaN or ''.

    ``wl_min_nm``/``wl_max_nm`` in nm, ``mjd`` in MJD, ``exptime_s`` in s.
    ``size_bytes`` is the archive's estimate (``None`` when not given).
    ``separation_arcsec`` is the closest approach to the source's path over
    1975-2027 (see :meth:`SkyPosition.separation_arcsec`).
    """

    archive: str
    collection: str = ""
    instrument: str = ""
    obs_id: str = ""
    target_name: str = ""
    ra_deg: float = float("nan")
    dec_deg: float = float("nan")
    mjd: float = float("nan")
    exptime_s: float = float("nan")
    wl_min_nm: float = float("nan")
    wl_max_nm: float = float("nan")
    resolving_power: float = float("nan")
    snr: float = float("nan")
    access_url: str = ""
    access_format: str = ""
    size_bytes: int | None = None
    separation_arcsec: float = float("nan")
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SearchStatus(enum.StrEnum):
    FOUND = "found"
    NONE = "none"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ArchiveResult:
    """The outcome of asking one archive."""

    archive: str
    status: SearchStatus
    records: tuple[SpectrumRecord, ...] = ()
    radius_arcsec: float = float("nan")
    query: str = ""
    error: str = ""
    elapsed_s: float = float("nan")
    searched_at: str = ""

    @property
    def n_spectra(self) -> int:
        return len(self.records)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass(frozen=True)
class SearchReport:
    """Every archive's answer for one position."""

    position: SkyPosition
    results: tuple[ArchiveResult, ...]
    searched_at: str

    def records(self) -> list[SpectrumRecord]:
        return [r for res in self.results for r in res.records]

    def complete(self) -> bool:
        """True only if every archive answered (found or none)."""
        return all(r.status in (SearchStatus.FOUND, SearchStatus.NONE) for r in self.results)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, with missing values (NaN) as None."""
        return _nan_to_none({"position": asdict(self.position), "searched_at": self.searched_at,
                             "results": [r.as_dict() for r in self.results]})

    def to_json(self) -> str:
        """Strict JSON: missing values become null, never ``NaN``."""
        return json.dumps(self.to_dict(), default=_json_default, indent=1, allow_nan=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SearchReport:
        """Inverse of :meth:`to_dict`; nulls become NaN (or None for sizes) again."""
        pos = SkyPosition(**{k: _nan_if_none(v) if k != "name" else (v or "")
                             for k, v in d["position"].items()})
        results = []
        for r in d["results"]:
            records = tuple(
                SpectrumRecord(**{k: (v if k == "size_bytes" else _nan_if_none(v))
                                  for k, v in rec.items()})
                for rec in r.get("records", [])
            )
            results.append(ArchiveResult(
                archive=r["archive"], status=SearchStatus(r["status"]), records=records,
                radius_arcsec=_nan_if_none(r.get("radius_arcsec")), query=r.get("query", ""),
                error=r.get("error", ""), elapsed_s=_nan_if_none(r.get("elapsed_s")),
                searched_at=r.get("searched_at", ""),
            ))
        return cls(pos, tuple(results), d.get("searched_at", ""))


# ------------------------------------------------------------------ transport


class Transport:
    """Everything that touches the network, so tests can replace it whole.

    Methods raise on failure; the runner turns exceptions into statuses.
    """

    def tap(self, url: str, adql: str):  # -> astropy Table
        raise NotImplementedError

    def ssa(self, url: str, ra: float, dec: float, diameter_deg: float):
        """Return ``(fields, rows)``: fields as dicts with name/utype/ucd/unit."""
        raise NotImplementedError

    def ssa_url(self, ivoid: str) -> str:
        raise NotImplementedError

    def mast(self, request: dict) -> dict:
        raise NotImplementedError

    def get_text(self, url: str, params: dict) -> str:
        raise NotImplementedError


class LiveTransport(Transport):
    """Real network access, with a socket timeout on every request.

    Synchronous TAP is used throughout: ESO's asynchronous endpoint hung for
    more than seven minutes in testing, while the same query ran in 11 s sync.
    Nothing is imported or opened until a method is called.
    """

    def __init__(self, timeout_s: float = 60.0) -> None:
        self.timeout_s = float(timeout_s)
        self._session = None

    def session(self):
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter

            timeout = (10.0, self.timeout_s)

            class _Timeout(HTTPAdapter):
                def send(self, request, **kwargs):
                    kwargs["timeout"] = kwargs.get("timeout") or timeout
                    return super().send(request, **kwargs)

            s = requests.Session()
            s.mount("http://", _Timeout())
            s.mount("https://", _Timeout())
            self._session = s
        return self._session

    def tap(self, url, adql):
        import pyvo

        return pyvo.dal.TAPService(url, session=self.session()).run_sync(adql).to_table()

    def ssa_url(self, ivoid):
        import pyvo

        found = pyvo.registry.search(ivoid=ivoid)
        if not len(found):
            raise LookupError(f"{ivoid} is not in the VO registry")
        return found[0].get_service("ssa").baseurl

    def ssa(self, url, ra, dec, diameter_deg):
        import astropy.units as u
        import pyvo
        from astropy.coordinates import SkyCoord

        try:
            res = pyvo.dal.SSAService(url, session=self.session()).search(
                pos=SkyCoord(ra, dec, unit="deg"), diameter=diameter_deg * u.deg, maxrec=100000
            )
        except pyvo.dal.DALFormatError:
            # PolarBase answers an empty search with QUERY_STATUS="OK" and no
            # TABLE, which pyvo rejects as malformed. Re-read the raw answer and
            # call it empty only if the service itself said OK.
            raw = self.session().get(url, params={
                "REQUEST": "queryData", "POS": f"{ra},{dec}", "SIZE": f"{diameter_deg}"}).text
            if ssa_says_empty(raw):
                return [], []
            raise
        fields = []
        for name in res.fieldnames:
            d = res.getdesc(name)
            fields.append({"name": name, "utype": d.utype or "", "ucd": d.ucd or "",
                           "unit": str(d.unit or "")})
        rows = [{f["name"]: rec[f["name"]] for f in fields} for rec in res]
        return fields, rows

    def mast(self, request):
        deadline = time.monotonic() + self.timeout_s
        while True:
            response = self.session().post(MAST_INVOKE, data={"request": json.dumps(request)})
            response.raise_for_status()
            result = response.json()
            status = result.get("status")
            if status == "COMPLETE":
                return result
            if status not in ("EXECUTING", "QUEUED"):
                raise RuntimeError(f"MAST: {result.get('msg') or status}")
            if time.monotonic() > deadline:
                raise TimeoutError("MAST did not complete in time")
            time.sleep(2)

    def get_text(self, url, params):
        response = self.session().get(url, params=params)
        response.raise_for_status()
        return response.text


# --------------------------------------------------------------- the archives


class ArchiveSearch:
    """One archive. Subclasses implement :meth:`search`."""

    name: str = ""
    #: Positional tolerance before proper motion, arcsec.
    base_radius_arcsec: float = 20.0

    def __init__(self, transport: Transport) -> None:
        self.transport = transport

    def query(self, position: SkyPosition, radius_arcsec: float) -> str:
        """Human-readable description of the query, kept with the result."""
        return (f"{self.name}: cone {radius_arcsec:.1f} arcsec around "
                f"({position.ra_deg:.6f}, {position.dec_deg:.6f})")

    def search(self, position: SkyPosition, radius_arcsec: float) -> list[SpectrumRecord]:
        raise NotImplementedError


class MastSearch(ArchiveSearch):
    """Every public spectrum MAST holds, all collections."""

    name = "MAST"
    #: IUE positions are commanded pointings, good to tens of arcsec.
    base_radius_arcsec = 40.0
    COLUMNS = ("obs_collection,instrument_name,obs_id,target_name,s_ra,s_dec,t_min,"
               "t_exptime,em_min,em_max,dataURL,dataproduct_type,dataRights")

    def request(self, position, radius_arcsec) -> dict:
        return {
            "service": "Mast.Caom.Filtered.Position", "format": "json",
            "page": 1, "pagesize": 50000,
            "params": {
                "columns": self.COLUMNS,
                "filters": [{"paramName": "dataproduct_type", "values": ["spectrum"]},
                            {"paramName": "dataRights", "values": ["PUBLIC"]}],
                "position": f"{position.ra_deg}, {position.dec_deg}, {radius_arcsec / 3600.0}",
            },
        }

    def query(self, position, radius_arcsec):
        return json.dumps(self.request(position, radius_arcsec)["params"])

    def search(self, position, radius_arcsec):
        return parse_mast(self.transport.mast(self.request(position, radius_arcsec)))


class ObsCoreSearch(ArchiveSearch):
    """Any TAP service publishing ``ivoa.ObsCore``."""

    COLUMNS = ("obs_collection, instrument_name, obs_id, target_name, s_ra, s_dec, t_min, "
               "t_exptime, em_min, em_max, em_res_power, access_url, access_format, "
               "access_estsize")

    def __init__(self, transport, *, name: str, url: str, where: str = "") -> None:
        super().__init__(transport)
        self.name = name
        self.url = url
        self.where = where

    def adql(self, position, radius_arcsec) -> str:
        extra = f" AND {self.where}" if self.where else ""
        return (
            f"SELECT {self.COLUMNS} FROM ivoa.ObsCore WHERE dataproduct_type='spectrum'"
            f"{extra} AND 1=CONTAINS(POINT('ICRS', s_ra, s_dec), "
            f"CIRCLE('ICRS', {position.ra_deg}, {position.dec_deg}, {radius_arcsec / 3600.0}))"
        )

    def query(self, position, radius_arcsec):
        return f"{self.url}: {self.adql(position, radius_arcsec)}"

    def search(self, position, radius_arcsec):
        return parse_obscore(self.transport.tap(self.url, self.adql(position, radius_arcsec)),
                             archive=self.name)


class CadcSearch(ArchiveSearch):
    """CADC through its ``caom2`` tables, every collection, public planes only."""

    name = "CADC"

    def adql(self, position, radius_arcsec) -> str:
        return (
            "SELECT o.collection, o.instrument_name, o.observationID, o.target_name, "
            "o.targetPosition_coordinates_cval1 AS ra, o.targetPosition_coordinates_cval2 AS dec, "
            "p.productID, p.time_bounds_lower, p.time_exposure, p.energy_bounds_lower, "
            "p.energy_bounds_upper, p.energy_resolvingPower, p.dataRelease, p.publisherID "
            "FROM caom2.Observation o JOIN caom2.Plane p ON o.obsID = p.obsID "
            "WHERE p.dataProductType = 'spectrum' AND 1=INTERSECTS(CIRCLE('ICRS', "
            f"{position.ra_deg}, {position.dec_deg}, {radius_arcsec / 3600.0}), p.position_bounds)"
        )

    def query(self, position, radius_arcsec):
        return f"{CADC_TAP}: {self.adql(position, radius_arcsec)}"

    def search(self, position, radius_arcsec):
        return parse_caom2(self.transport.tap(CADC_TAP, self.adql(position, radius_arcsec)))


class SsaSearch(ArchiveSearch):
    """A registered SSA service, located through the VO registry."""

    def __init__(self, transport, *, name: str, ivoid: str) -> None:
        super().__init__(transport)
        self.name = name
        self.ivoid = ivoid

    def query(self, position, radius_arcsec):
        return f"{self.ivoid}: SSA cone, diameter {2 * radius_arcsec:.1f} arcsec"

    def search(self, position, radius_arcsec):
        url = self.transport.ssa_url(self.ivoid)
        fields, rows = self.transport.ssa(url, position.ra_deg, position.dec_deg,
                                          2 * radius_arcsec / 3600.0)
        return parse_ssa(fields, rows, archive=self.name)


class ElodieSearch(ArchiveSearch):
    """ELODIE (OHP), searchable by object name only."""

    name = "ELODIE"

    def query(self, position, radius_arcsec):
        return f"{ELODIE_SEARCH}: name {position.name!r}"

    def search(self, position, radius_arcsec):
        text = self.transport.get_text(
            ELODIE_SEARCH, {"n": "e500", "c": "o", "o": position.name, "a": "csv"})
        return parse_elodie(text)


def default_searches(transport: Transport) -> list[ArchiveSearch]:
    """The archives ``../hst_archive`` covers, widened to all their spectra."""
    return [
        MastSearch(transport),
        ObsCoreSearch(transport, name="ESO", url=ESO_TAP),
        CadcSearch(transport),
        ObsCoreSearch(transport, name="CfA TDC", url=CFA_TAP),
        *(SsaSearch(transport, name=n, ivoid=i) for n, i in SSA_IVOIDS.items()),
        ElodieSearch(transport),
    ]


# ------------------------------------------------------------------- the runner


def search_external_spectra(
    position: SkyPosition,
    searches: Sequence[ArchiveSearch] | None = None,
    *,
    allow_network: bool = True,
    timeout_s: float = 120.0,
    max_workers: int = 8,
    clock: Callable[[], float] = time.monotonic,
) -> SearchReport:
    """Ask every archive at once; never raise for a single archive's failure.

    Parameters
    ----------
    position : SkyPosition
    searches : sequence of ArchiveSearch, optional
        Defaults to :func:`default_searches` over a :class:`LiveTransport`.
    allow_network : bool
        When False every archive is reported as skipped and nothing is sent.
    timeout_s : float
        Wall-clock budget for the whole search. Archives still running when it
        expires are reported as timed out.
    """
    started = _utcnow()
    if searches is None:
        searches = default_searches(LiveTransport(timeout_s=min(timeout_s, 90.0)))
    radii = {s.name: position.search_radius_arcsec(s.base_radius_arcsec) for s in searches}

    if not allow_network:
        results = tuple(
            ArchiveResult(s.name, SearchStatus.SKIPPED, radius_arcsec=radii[s.name],
                          query=s.query(position, radii[s.name]),
                          error="network access is disabled", searched_at=started)
            for s in searches
        )
        return SearchReport(position, results, started)

    def run(s: ArchiveSearch) -> ArchiveResult:
        radius = radii[s.name]
        query = s.query(position, radius)
        if isinstance(s, ElodieSearch) and not position.name:
            return ArchiveResult(s.name, SearchStatus.SKIPPED, radius_arcsec=radius,
                                 query=query, error="ELODIE searches by name; none known",
                                 searched_at=_utcnow())
        t0 = clock()
        try:
            found = s.search(position, radius)
        except Exception as exc:
            return ArchiveResult(s.name, SearchStatus.FAILED, radius_arcsec=radius, query=query,
                                 error=f"{type(exc).__name__}: {exc}"[:500],
                                 elapsed_s=clock() - t0, searched_at=_utcnow())
        records = tuple(_with_separation(found, position))
        return ArchiveResult(
            s.name, SearchStatus.FOUND if records else SearchStatus.NONE, records=records,
            radius_arcsec=radius, query=query, elapsed_s=clock() - t0, searched_at=_utcnow(),
        )

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {pool.submit(run, s): s for s in searches}
        done, _ = wait(futures, timeout=timeout_s)
        results = []
        for fut, s in futures.items():
            if fut in done:
                results.append(fut.result())
            else:
                results.append(ArchiveResult(
                    s.name, SearchStatus.TIMEOUT, radius_arcsec=radii[s.name],
                    query=s.query(position, radii[s.name]),
                    error=f"no answer within {timeout_s:.0f} s", elapsed_s=timeout_s,
                    searched_at=_utcnow()))
    finally:
        # Do not wait for a hung archive; its socket timeout will end the thread.
        pool.shutdown(wait=False, cancel_futures=True)
    return SearchReport(position, tuple(results), started)


# ------------------------------------------------------ snapshot and provider

#: Search results for the prerelease sources, shipped for the browser build
#: (which cannot query archives) and as a starting point for the desktop app.
SNAPSHOT_RESOURCE = "external_spectra.json"


def position_from_reference(values: dict[str, float], *, name: str = "") -> SkyPosition:
    """Search position from a row of the prerelease reference table.

    Uses the transit reference point (ra0, dec0) at J2017.5 and the refitted
    proper motion. These are recomputed values, used here only to size the
    search and to follow the star's path -- never shown as catalogue values.
    """
    return SkyPosition(
        ra_deg=values["ra0_deg"], dec_deg=values["dec0_deg"], epoch_jyear=2017.5,
        pmra_masyr=values.get("fit_pmra_mas_yr", float("nan")),
        pmdec_masyr=values.get("fit_pmdec_mas_yr", float("nan")),
        name=name,
    )


def build_snapshot(
    positions: dict[int, SkyPosition],
    search: Callable[[SkyPosition], SearchReport] = search_external_spectra,
) -> dict[str, Any]:
    """Run ``search`` for every source; return the JSON-ready snapshot."""
    return {
        "created_at": _utcnow(),
        "note": "Metadata only: what each archive listed near each source. "
                "Regenerate with `gaia-dr4-explorer snapshot-spectra`.",
        "sources": {str(sid): search(pos).to_dict() for sid, pos in sorted(positions.items())},
    }


class ExternalSpectraProvider:
    """Serves archive-search results: a saved search, the shipped snapshot, or
    a live search. UI code calls this; it never queries an archive itself
    (CLAUDE.md invariant 8).

    Parameters
    ----------
    cache_dir : Path, optional
        Where live searches are saved, so a source searched once shows its
        result again without a new search. Keyed by release and source_id
        (CLAUDE.md invariant 1). Without it nothing is saved.
    release : str
        Gaia release the source_ids belong to.
    """

    def __init__(self, *, allow_network: bool = True, snapshot: dict | None = None,
                 search: Callable[..., SearchReport] = search_external_spectra,
                 cache_dir=None, release: str = "Gaia DR4_RC3") -> None:
        self.allow_network = allow_network
        self._snapshot = snapshot
        self._search = search
        self._cache_dir = cache_dir
        self.release = release

    def snapshot(self) -> dict:
        if self._snapshot is None:
            from importlib.resources import files

            res = files("gaia_dr4_explorer.resources") / SNAPSHOT_RESOURCE
            self._snapshot = json.loads(res.read_text()) if res.is_file() else {"sources": {}}
        return self._snapshot

    def bundled(self, source_id: int) -> SearchReport | None:
        """The shipped search for this source, or None when there is none."""
        d = self.snapshot().get("sources", {}).get(str(int(source_id)))
        return SearchReport.from_dict(d) if d else None

    def _path(self, source_id: int):
        if self._cache_dir is None:
            return None
        from gaia_dr4_explorer.config import CacheLayout

        return CacheLayout(self._cache_dir).external_spectra(self.release, source_id)

    def saved(self, source_id: int) -> SearchReport | None:
        """The last live search saved for this source, if any."""
        path = self._path(source_id)
        if path is None or not path.is_file():
            return None
        try:
            return SearchReport.from_dict(json.loads(path.read_text()))
        except (ValueError, KeyError, TypeError):
            return None           # an unreadable cache entry is just absent

    def latest(self, source_id: int) -> tuple[SearchReport | None, str]:
        """The newest result available without searching, and where it came from."""
        saved = self.saved(source_id)
        if saved is not None:
            return saved, "saved"
        bundled = self.bundled(source_id)
        return (bundled, "shipped") if bundled is not None else (None, "")

    def search(self, position: SkyPosition, *, source_id: int | None = None,
               **kwargs) -> SearchReport:
        """A live search now, saved for next time when ``source_id`` is given.

        Reported as skipped (and not saved) when the network is off.
        """
        report = self._search(position, allow_network=self.allow_network, **kwargs)
        path = self._path(source_id) if source_id is not None else None
        if path is not None and self.allow_network:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(report.to_json())
            tmp.replace(path)
        return report


# -------------------------------------------------------------------- parsers


def parse_mast(response: dict) -> list[SpectrumRecord]:
    """MAST CAOM rows. Wavelengths arrive in nm, times in MJD (checked live)."""
    out = []
    for d in response.get("data", []):
        if str(d.get("dataRights", "PUBLIC")).upper() != "PUBLIC":
            continue
        url = str(d.get("dataURL") or "")
        if url.startswith("mast:"):
            url = MAST_DOWNLOAD + url
        out.append(SpectrumRecord(
            archive="MAST", collection=_s(d.get("obs_collection")),
            instrument=_s(d.get("instrument_name")), obs_id=_s(d.get("obs_id")),
            target_name=_s(d.get("target_name")), ra_deg=_f(d.get("s_ra")),
            dec_deg=_f(d.get("s_dec")), mjd=_f(d.get("t_min")), exptime_s=_f(d.get("t_exptime")),
            wl_min_nm=_f(d.get("em_min")), wl_max_nm=_f(d.get("em_max")),
            access_url=url, access_format="application/fits" if url else "",
        ))
    return out


def parse_obscore(table, *, archive: str) -> list[SpectrumRecord]:
    """ObsCore rows, converting with the units the service declares.

    ObsCore defaults are applied only where a column carries no unit:
    em_* in m, t_* in MJD, access_estsize in kB.
    """
    em_lo = _as(table, "em_min", "nm", default_unit="m")
    em_hi = _as(table, "em_max", "nm", default_unit="m")
    size_b = _as(table, "access_estsize", "byte", default_unit="kbyte")
    out = []
    for i, row in enumerate(table):
        lo, hi = em_lo[i], em_hi[i]
        out.append(SpectrumRecord(
            archive=archive, collection=_s(_get(row, "obs_collection")),
            instrument=_s(_get(row, "instrument_name")), obs_id=_s(_get(row, "obs_id")),
            target_name=_s(_get(row, "target_name")), ra_deg=_f(_get(row, "s_ra")),
            dec_deg=_f(_get(row, "s_dec")), mjd=_f(_get(row, "t_min")),
            exptime_s=_f(_get(row, "t_exptime")), wl_min_nm=lo, wl_max_nm=hi,
            resolving_power=resolving_power(_f(_get(row, "em_res_power")), lo, hi),
            access_url=_s(_get(row, "access_url")), access_format=_s(_get(row, "access_format")),
            size_bytes=int(size_b[i]) if np.isfinite(size_b[i]) else None,
        ))
    return out


def parse_caom2(table, *, now: datetime | None = None) -> list[SpectrumRecord]:
    """CADC ``caom2`` planes; planes still proprietary (dataRelease ahead) are dropped."""
    now = now or datetime.now(UTC)
    em_lo = _as(table, "energy_bounds_lower", "nm", default_unit="m")
    em_hi = _as(table, "energy_bounds_upper", "nm", default_unit="m")
    out = []
    for i, row in enumerate(table):
        release = _s(_get(row, "dataRelease"))
        if not _is_public(release, now):
            continue
        pid = _s(_get(row, "publisherID"))
        out.append(SpectrumRecord(
            archive="CADC", collection=_s(_get(row, "collection")),
            instrument=_s(_get(row, "instrument_name")), obs_id=_s(_get(row, "productID")),
            target_name=_s(_get(row, "target_name")), ra_deg=_f(_get(row, "ra")),
            dec_deg=_f(_get(row, "dec")), mjd=_f(_get(row, "time_bounds_lower")),
            exptime_s=_f(_get(row, "time_exposure")), wl_min_nm=em_lo[i], wl_max_nm=em_hi[i],
            resolving_power=_f(_get(row, "energy_resolvingPower")),
            access_url=CADC_DATALINK + pid if pid else "",
            access_format="application/x-votable+xml;content=datalink" if pid else "",
        ))
    return out


#: SSA fields located by utype suffix (case-insensitive), then UCD.
_SSA_KEYS = {
    "title": (("dataid.title",), ("meta.title",)),
    "instrument": (("dataid.instrument",), ("instr",)),
    "collection": (("dataid.collection",), ()),
    "pos": (("char.spatialaxis.coverage.location.value", "target.pos"), ("pos.eq",)),
    "time": (("char.timeaxis.coverage.location.value",), ("time.epoch",)),
    "exptime": (("char.timeaxis.coverage.bounds.extent",), ("time.duration",)),
    "wl_start": (("char.spectralaxis.coverage.bounds.start",), ()),
    "wl_stop": (("char.spectralaxis.coverage.bounds.stop",), ()),
    "wl_loc": (("char.spectralaxis.coverage.location.value",), ()),
    "wl_ext": (("char.spectralaxis.coverage.bounds.extent",), ()),
    "respower": (("char.spectralaxis.respower",), ()),
    "resolution": (("char.spectralaxis.resolution",), ()),
    "snr": (("derived.snr",), ()),
    "url": (("access.reference",), ("meta.ref.url",)),
    "format": (("access.format",), ("meta.code.mime",)),
    "size": (("access.size",), ()),
}


def parse_ssa(fields: Sequence[dict], rows: Iterable[dict], *, archive: str) -> list[SpectrumRecord]:
    """SSA rows, matched by utype so one parser serves every service.

    Services differ: PolarBase gives the spectral range as a centre plus an
    extent in nm, others give start/stop in m. Both are handled, using each
    field's declared unit.
    """
    col = {k: _find_field(fields, u, c) for k, (u, c) in _SSA_KEYS.items()}
    unit = {f["name"]: f.get("unit", "") for f in fields}

    def val(row, key):
        name = col[key]
        return None if name is None else row.get(name)

    def wl(row, key):
        return _to_nm(_f(val(row, key)), unit.get(col[key] or "", "") or "m")

    out = []
    for row in rows:
        ra, dec = _pair(val(row, "pos"))
        lo, hi = wl(row, "wl_start"), wl(row, "wl_stop")
        if not (np.isfinite(lo) and np.isfinite(hi)):
            mid, ext = wl(row, "wl_loc"), wl(row, "wl_ext")
            if np.isfinite(mid) and np.isfinite(ext):
                lo, hi = mid - ext / 2.0, mid + ext / 2.0
        rp = _f(val(row, "respower"))
        if not np.isfinite(rp):
            fwhm = wl(row, "resolution")
            if np.isfinite(fwhm) and fwhm > 0 and np.isfinite(lo) and np.isfinite(hi):
                rp = 0.5 * (lo + hi) / fwhm
        size = _f(val(row, "size"))
        out.append(SpectrumRecord(
            archive=archive, collection=_s(val(row, "collection")),
            instrument=_s(val(row, "instrument")), obs_id=_s(val(row, "title")),
            ra_deg=ra, dec_deg=dec,
            mjd=_to_mjd(val(row, "time"), unit.get(col["time"] or "", "")),
            exptime_s=_f(val(row, "exptime")), wl_min_nm=lo, wl_max_nm=hi,
            resolving_power=rp, snr=_f(val(row, "snr")), access_url=_s(val(row, "url")),
            access_format=_s(val(row, "format")),
            size_bytes=int(size) if np.isfinite(size) and size > 0 else None,
        ))
    return out


def parse_elodie(content: str) -> list[SpectrumRecord]:
    """The ELODIE tab-separated listing (format as parsed in ``hst_archive``)."""
    out = []
    for line in content.splitlines():
        if not line or line.startswith(("#", "$")):
            continue
        f = line.split("\t")
        if len(f) >= 9 and f[4].isdigit() and f[5].isdigit():
            out.append(SpectrumRecord(
                archive="ELODIE", instrument="ELODIE", obs_id=f"{f[4]}/{f[5]}",
                exptime_s=_f(f[7]), snr=_f(f[8]), wl_min_nm=400.0, wl_max_nm=680.0,
                resolving_power=42000.0,
                access_url=ELODIE_GET.format(dataset=f[4], imanum=f[5]),
                access_format="application/fits",
                note="range and R nominal; S/N in order 47; epoch in the file header",
            ))
    return out


def ssa_says_empty(votable_text: str) -> bool:
    """True for an SSA answer with QUERY_STATUS OK and no result table."""
    import re

    ok = re.search(r'name="QUERY_STATUS"\s+value="OK"', votable_text) is not None
    return ok and "<TABLE" not in votable_text.upper()


def resolving_power(value: float, lo_nm: float, hi_nm: float) -> float:
    """Dimensionless R. Some services (CfA TDC) fill ``em_res_power`` with a
    resolution element in metres (~1e-11); such values become lambda_mid / dlambda."""
    if not np.isfinite(value) or value <= 0:
        return float("nan")
    if value < 1e-6:
        if np.isfinite(lo_nm) and np.isfinite(hi_nm):
            return 0.5 * (lo_nm + hi_nm) / (value * 1e9)
        return float("nan")
    return value if value >= 1 else float("nan")


# -------------------------------------------------------------------- helpers


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _angular_distance_arcsec(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    s = (math.sin((d2 - d1) / 2) ** 2
         + math.cos(d1) * math.cos(d2) * math.sin((r2 - r1) / 2) ** 2)
    return math.degrees(2 * math.asin(min(1.0, math.sqrt(s)))) * 3600.0


def _with_separation(records, position):
    for r in records:
        yield replace(r, separation_arcsec=position.separation_arcsec(r.ra_deg, r.dec_deg))


def _f(x) -> float:
    try:
        if x is None or np.ma.is_masked(x):
            return float("nan")
        v = float(np.ravel(np.ma.filled(np.ma.asarray(x, dtype=float), np.nan))[0])
        return v
    except (TypeError, ValueError, IndexError):
        return float("nan")


def _s(x) -> str:
    if x is None or np.ma.is_masked(x):
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    return str(x)


def _get(row, name):
    try:
        return row[name]
    except (KeyError, IndexError, ValueError):
        return None


def _as(table, name, target, *, default_unit):
    """Column ``name`` in ``target`` units as float64, NaN where masked/missing."""
    import astropy.units as u

    if name not in table.colnames:
        return np.full(len(table), np.nan)
    col = table[name]
    values = np.asarray(np.ma.filled(np.ma.asarray(col, dtype=float), np.nan), dtype=float)
    unit = col.unit if col.unit is not None else u.Unit(default_unit)
    try:
        return (values * unit).to_value(target)
    except u.UnitConversionError:
        return (values * u.Unit(default_unit)).to_value(target)


def _to_nm(value: float, unit: str) -> float:
    import astropy.units as u

    if not np.isfinite(value):
        return float("nan")
    try:
        return float((value * u.Unit(unit or "m")).to_value(u.nm, equivalencies=u.spectral()))
    except (ValueError, u.UnitsError):
        return float("nan")


def _to_mjd(value, unit: str) -> float:
    """SSA times come as MJD, JD or ISO strings, depending on the service."""
    if value is None or (not isinstance(value, str) and np.ma.is_masked(value)):
        return float("nan")
    if isinstance(value, bytes | str):
        text = value.decode() if isinstance(value, bytes) else value
        try:
            return float(text)
        except ValueError:
            from astropy.time import Time

            try:
                return float(Time(text).mjd)
            except ValueError:
                return float("nan")
    v = _f(value)
    if np.isfinite(v) and v > 2_400_000:
        return v - 2_400_000.5      # JD -> MJD
    return v


def _pair(value) -> tuple[float, float]:
    try:
        a = np.ravel(np.ma.filled(np.ma.asarray(value, dtype=float), np.nan))
        return (float(a[0]), float(a[1])) if a.size >= 2 else (float("nan"), float("nan"))
    except (TypeError, ValueError):
        return float("nan"), float("nan")


def _find_field(fields, utype_suffixes, ucds) -> str | None:
    for suffix in utype_suffixes:
        for f in fields:
            if str(f.get("utype", "")).lower().endswith(suffix):
                return f["name"]
    for ucd in ucds:
        for f in fields:
            if str(f.get("ucd", "")).lower().split(";")[0] == ucd:
                return f["name"]
    return None


def _is_public(release: str, now: datetime) -> bool:
    if not release:
        return True
    try:
        when = datetime.fromisoformat(release.replace("Z", "+00:00"))
    except ValueError:
        return True
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when <= now


def _nan_to_none(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_nan_to_none(v) for v in obj]
    return obj


def _nan_if_none(v):
    return float("nan") if v is None else v


def _json_default(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, enum.Enum):
        return obj.value
    raise TypeError(type(obj).__name__)


__all__ = [
    "ArchiveResult", "ArchiveSearch", "CadcSearch", "ElodieSearch", "ExternalSpectraProvider",
    "LiveTransport", "SNAPSHOT_RESOURCE", "build_snapshot", "position_from_reference",
    "MastSearch", "ObsCoreSearch", "SearchReport", "SearchStatus", "SkyPosition",
    "SpectrumRecord", "SsaSearch", "Transport", "default_searches", "parse_caom2",
    "parse_elodie", "parse_mast", "parse_obscore", "parse_ssa", "resolving_power",
    "ssa_says_empty",
    "search_external_spectra",
]

