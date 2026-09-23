"""Download the spectrum a user picked, and keep it.

A search record says where an archive lists a spectrum; this module turns
that into files on disk and a :class:`~.spectrum_reader.Spectrum`. The link
an archive gives is not always the file (checked 2026-09-23):

==============  ============================================================
ESO             DataLink page -> the direct Phase 3 file, derived from the ID
CADC            DataLink page -> the ``#this`` file listed in it
MAST APOGEE     MAST's download service returns 404 -> the same file on the
                SDSS science archive server
MAST HST GHRS   ``_c0f`` holds wavelengths only -> also fetch ``_c1f`` (flux)
MAST HST        pick from MAST's product list: the extracted 1-D spectrum
                (``X1D`` first); the observation-level link can be raw
CADC raw        DAO and GRIF frames (calibration level 1) have no wavelength
                axis; the reader refuses them and says why
others          the link is the FITS file
==============  ============================================================

Downloaded files are immutable (CLAUDE.md invariant 2): written once,
atomically, never modified, with a sidecar recording URL, size, SHA-256 and
time. A file already on disk is reused without a request.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from gaia_dr4_explorer.data.external_spectra import (
    MAST_DOWNLOAD,
    SpectrumRecord,
    Transport,
)
from gaia_dr4_explorer.data.spectrum_reader import Spectrum, read_spectrum

ESO_FILE = "https://dataportal.eso.org/dataportal_new/file/"
SDSS_APOGEE_STARS = "https://data.sdss.org/sas/dr17/apogee/spectro/redux/dr17/stars/"

#: Largest file fetched without the user raising the limit.
DEFAULT_MAX_BYTES = 300_000_000

#: MAST HST products that hold an extracted 1-D spectrum, best first.
MAST_SPECTRUM_PRODUCTS = ("X1D", "SX1", "X1DSUM", "C1F")

_FITS_MAGIC = b"SIMPLE  ="
_GZIP_MAGIC = b"\x1f\x8b"


class FileResolutionError(RuntimeError):
    """The archive lists the spectrum but offers no file we can fetch."""


def resolve_file_urls(record: SpectrumRecord, transport: Transport) -> list[str]:
    """The file URL(s) that hold this record's spectrum.

    Two URLs mean a wavelength file and a flux file (HST GHRS).
    """
    url = record.access_url
    if record.archive == "ESO" and "eso.org/ID?" in url:
        return [ESO_FILE + url.rsplit("?", 1)[1]]
    if "datalink" in record.access_format.lower():
        return [datalink_file(transport.get_bytes(url, 5_000_000))]
    if record.archive == "MAST":
        if "mast:SDSS/apogee/" in url:
            return [apogee_sas_url(url)]
        if record.collection == "HST" and record.archive_key:
            # The observation-level link can be a raw file (GHRS _d1f gave 3
            # "pixels" at 19.8 nm); the product list names the extracted spectrum.
            return mast_product_urls(record, transport)
        if not url:
            return mast_product_urls(record, transport)
        if url.endswith("_c0f.fits"):
            return [url, url[: -len("_c0f.fits")] + "_c1f.fits"]
    if not url:
        raise FileResolutionError(f"{record.archive} lists this spectrum without a file")
    return [url]


def apogee_sas_url(url: str) -> str:
    """``mast:SDSS/apogee/<tel>/<field>/<id>/apStar-...fits`` on the SDSS server.

    MAST lists these but its download service answers 404; the SAS keeps
    apStar files at ``stars/<tel>/<field>/``, without the per-star directory.
    """
    path = url.split("mast:SDSS/apogee/", 1)[1]
    parts = path.split("/")
    if len(parts) < 4:
        raise FileResolutionError(f"unexpected APOGEE path {path!r}")
    telescope, field, filename = parts[0], "/".join(parts[1:-2]), parts[-1]
    return f"{SDSS_APOGEE_STARS}{telescope}/{field}/{filename}"


def datalink_file(votable_bytes: bytes) -> str:
    """The science file a DataLink response lists as ``#this`` (FITS preferred)."""
    from astropy.io.votable import parse_single_table

    table = parse_single_table(io.BytesIO(votable_bytes)).to_table()
    rows = [r for r in table if str(r["semantics"]).strip() == "#this"
            and str(r["access_url"]).strip() not in ("", "--")]
    if not rows:
        raise FileResolutionError("the DataLink response lists no #this file")
    fits_rows = [r for r in rows if "fits" in str(r["content_type"]).lower()]
    return str((fits_rows or rows)[0]["access_url"]).strip()


def mast_product_urls(record: SpectrumRecord, transport: Transport) -> list[str]:
    """Pick the 1-D spectrum from MAST's product list for an HST observation."""
    if not record.archive_key:
        raise FileResolutionError("MAST gives no file and no obsid for this observation")
    result = transport.mast({"service": "Mast.Caom.Products", "format": "json",
                             "params": {"obsid": record.archive_key}})
    science = {}
    for p in result.get("data", []):
        if str(p.get("productType")) == "SCIENCE" and str(p.get("dataRights")) == "PUBLIC":
            science.setdefault(str(p.get("productSubGroupDescription")), p)
    for group in MAST_SPECTRUM_PRODUCTS:
        if group in science:
            uri = science[group]["dataURI"]
            urls = [MAST_DOWNLOAD + uri]
            if group == "C1F" and "C0F" in science:        # GHRS: wavelengths first
                urls = [MAST_DOWNLOAD + science["C0F"]["dataURI"], urls[0]]
            return urls
    raise FileResolutionError(
        f"MAST has no extracted 1-D spectrum for {record.obs_id} "
        f"(science products: {', '.join(sorted(science)) or 'none'})")


@dataclass(frozen=True)
class LoadedSpectrum:
    record: SpectrumRecord
    spectrum: Spectrum
    paths: tuple[Path, ...]
    downloaded: bool          # False when every file came from the cache


class SpectrumFileStore:
    """Fetch, keep and read spectra the user picked."""

    def __init__(self, cache_dir: Path, transport: Transport, *,
                 max_bytes: int = DEFAULT_MAX_BYTES, allow_network: bool = True) -> None:
        self.root = Path(cache_dir) / "external_spectra" / "files"
        self.transport = transport
        self.max_bytes = int(max_bytes)
        self.allow_network = allow_network

    def path_for(self, url: str) -> Path:
        """Cache path: a hash of the URL keeps it unique, the name keeps it readable."""
        digest = hashlib.sha256(url.encode()).hexdigest()[:16]
        name = re.sub(r"[^A-Za-z0-9._+-]", "_", url.rstrip("/").split("/")[-1].split("?")[-1])
        return self.root / f"{digest}-{name[-80:] or 'spectrum'}"

    def fetch(self, record: SpectrumRecord) -> tuple[list[Path], bool]:
        """Files for this record, downloading what is not cached yet."""
        urls = resolve_file_urls(record, self.transport)
        paths, downloaded = [], False
        for url in urls:
            path = self.path_for(url)
            if not path.is_file():
                if not self.allow_network:
                    raise FileResolutionError("not downloaded yet, and network access is off")
                self._download(url, path, record)
                downloaded = True
            paths.append(path)
        return paths, downloaded

    def load(self, record: SpectrumRecord) -> LoadedSpectrum:
        """Fetch (or reuse) and read the spectrum."""
        # Not refused on calibration level: CfA TDC labels its extracted,
        # wavelength-calibrated echelle orders level 1 (ObsCore: "instrumental
        # data in a standard format"). The reader decides, and says why a raw
        # frame (pixel axis, no wavelength solution) is not a spectrum.
        paths, downloaded = self.fetch(record)
        spectrum = read_spectrum(paths if len(paths) > 1 else paths[0])
        return LoadedSpectrum(record, spectrum, tuple(paths), downloaded)

    def _download(self, url: str, path: Path, record: SpectrumRecord) -> None:
        data = self.transport.get_bytes(url, self.max_bytes)
        if not (data.startswith(_FITS_MAGIC) or data.startswith(_GZIP_MAGIC)):
            head = data[:80].decode("latin-1", "replace")
            raise FileResolutionError(f"{url} did not return a FITS file: {head!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        sidecar = {
            "url": url, "archive": record.archive, "obs_id": record.obs_id,
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        path.with_name(path.name + ".json").write_text(json.dumps(sidecar, indent=1))
