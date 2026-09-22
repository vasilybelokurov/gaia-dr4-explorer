"""Gaia archive access.

Product modules and UI modules never talk to the archive directly; they go
through this provider, so archive access can be mocked in tests and disabled
entirely in a build that has no network.

Only DataLink retrieval is implemented so far. TAP queries are deferred until
the public DR4 archive exists and its schema is known.
"""

from __future__ import annotations

import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from astropy.io.votable import parse_single_table
from astropy.table import Table

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.config import AppConfig, CacheLayout
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord

#: Gaia DataLink endpoint.
DATALINK_URL = "https://gea.esac.esa.int/data-server/data"

#: Retrieval types this application knows how to parse.  Anything not in this
#: set is refused rather than interpolated into a cache path.
EPOCH_PHOTOMETRY = "EPOCH_PHOTOMETRY"
XP_SAMPLED = "XP_SAMPLED"
XP_CONTINUOUS = "XP_CONTINUOUS"

KNOWN_RETRIEVAL_TYPES = frozenset({EPOCH_PHOTOMETRY, XP_SAMPLED, XP_CONTINUOUS})


class ArchiveError(RuntimeError):
    """Raised when an archive request fails or returns nothing usable."""


class ArchiveUnavailable(ArchiveError):
    """The product does not exist for this source.  A normal outcome."""


class GaiaArchiveProvider:
    """Retrieves DataLink products, caching each one immutably.

    Parameters
    ----------
    config : AppConfig
        Supplies the cache root and whether network access is permitted.
    timeout : float
        Per-request timeout in seconds.
    """

    def __init__(self, config: AppConfig, *, timeout: float = 90.0) -> None:
        self._config = config
        self._layout = CacheLayout(config.cache_dir)
        self._timeout = timeout

    def cache_path(self, source_id: int, retrieval_type: str, release: str) -> Path:
        _check_retrieval_type(retrieval_type)
        directory = self._layout.raw(release, int(source_id))
        return directory / f"{retrieval_type.lower()}.vot"

    def get_datalink_product(
        self,
        source_id: int,
        retrieval_type: str,
        *,
        release: str = "Gaia DR3",
        refresh: bool = False,
    ) -> tuple[Table, ProvenanceRecord]:
        """Fetch one DataLink product for one source.

        Never called during discovery: a product is retrieved only after the
        user asks for it.

        Raises
        ------
        ArchiveUnavailable
            The archive answered, but with no data for this source.
        ArchiveError
            The request failed, or network access is disabled and nothing is
            cached.
        """
        _check_retrieval_type(retrieval_type)
        path = self.cache_path(source_id, retrieval_type, release)
        if path.exists() and not refresh:
            try:
                return self._read(path, source_id, retrieval_type, release, cached=True)
            except ArchiveUnavailable:
                raise
            except ArchiveError:
                # A cached file that no longer parses, or that holds the wrong
                # source, is discarded rather than displayed.
                path.unlink(missing_ok=True)
                if not self._config.allow_network:
                    raise

        if not self._config.allow_network:
            raise ArchiveError(
                f"{retrieval_type} for {source_id} is not cached and network "
                "access is disabled"
            )

        url = self._url(source_id, retrieval_type, release)
        self._layout.ensure(path.parent)
        # A per-process temporary name: a shared ".part" lets concurrent
        # downloads truncate and interleave each other's bytes.
        tmp = path.with_suffix(f".part.{os.getpid()}")
        try:
            with urllib.request.urlopen(url, timeout=self._timeout) as response:
                with tmp.open("wb") as fh:
                    shutil.copyfileobj(response, fh)
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            raise ArchiveError(f"could not retrieve {retrieval_type} for {source_id}: {exc}") from exc

        if tmp.stat().st_size == 0:
            # The archive answers 200 with an empty body for a product a source
            # does not have.  That is an absence, not a failure.
            tmp.unlink(missing_ok=True)
            raise ArchiveUnavailable(
                f"{release} has no {retrieval_type} for source {source_id}"
            )
        # Parse before promoting to the cache, so a truncated or wrong-source
        # response is never left behind for the next call to trust.
        try:
            result = self._read(tmp, source_id, retrieval_type, release, cached=False)
        except ArchiveError:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        return result

    def _url(self, source_id: int, retrieval_type: str, release: str) -> str:
        query = urllib.parse.urlencode(
            {
                "RETRIEVAL_TYPE": retrieval_type,
                "ID": f"{release} {int(source_id)}",
                "FORMAT": "votable",
                "DATA_STRUCTURE": "INDIVIDUAL",
                "RELEASE": release,
            }
        )
        return f"{DATALINK_URL}?{query}"

    def _read(
        self, path: Path, source_id: int, retrieval_type: str, release: str, *, cached: bool
    ) -> tuple[Table, ProvenanceRecord]:
        try:
            votable = parse_single_table(path)
            table = votable.to_table(use_names_over_ids=True)
        except Exception as exc:
            raise ArchiveError(f"could not parse {path}: {exc}") from exc
        if len(table) == 0:
            raise ArchiveUnavailable(
                f"{release} has no {retrieval_type} for source {source_id}"
            )
        table.meta["votable_params"] = {
            p.name: p.value for p in votable.params if p.name is not None
        }
        _check_source_id(table, source_id, release, path)
        record = ProvenanceRecord(
            kind=f"datalink-{retrieval_type.lower()}",
            release=release,
            source_url=self._url(source_id, retrieval_type, release),
            source_id=int(source_id),
            steps=(
                f"retrieved {retrieval_type} via Gaia DataLink"
                + (" (served from cache)" if cached else ""),
            ),
            software={"gaia_dr4_explorer": __version__},
            notes={"n_rows": len(table), "local_file": str(path)},
        )
        return table, record


def _check_retrieval_type(retrieval_type: str) -> None:
    if retrieval_type not in KNOWN_RETRIEVAL_TYPES:
        raise ArchiveError(
            f"unknown retrieval type {retrieval_type!r}; "
            f"known: {sorted(KNOWN_RETRIEVAL_TYPES)}"
        )


def _check_source_id(table: Table, source_id: int, release: str, path: Path) -> None:
    """Refuse a product that is not for the source or release we asked about.

    A cache file is just a path on disk; nothing else guarantees its contents
    match the identifier in its name.  DataLink products carry their identity
    in the VOTable PARAMs rather than in a column -- ``EPOCH_PHOTOMETRY`` has
    no ``source_id`` column at all -- so both are checked.
    """
    import numpy as np

    params = table.meta.get("votable_params") or {}
    declared = params.get("source_id")
    if declared is not None:
        if isinstance(declared, bytes):
            declared = declared.decode("utf-8", "replace")
        try:
            declared_id = int(str(declared).strip())
        except ValueError:
            declared_id = None
        if declared_id is not None and declared_id != int(source_id):
            raise ArchiveError(
                f"{path} declares source_id {declared_id}, not {source_id}; "
                "refusing to use it"
            )

    declared_release = params.get("release")
    if declared_release is not None:
        if isinstance(declared_release, bytes):
            declared_release = declared_release.decode("utf-8", "replace")
        if str(declared_release).strip() != release:
            raise ArchiveError(
                f"{path} declares release {declared_release!r}, not {release!r}; "
                "refusing to use it"
            )

    if "source_id" in table.colnames:
        found = np.unique(np.asarray(table["source_id"]))
        if found.size != 1 or int(found[0]) != int(source_id):
            raise ArchiveError(
                f"{path} holds source_id {found.tolist()}, not {source_id}; "
                "refusing to use it"
            )
