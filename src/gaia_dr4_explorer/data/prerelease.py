"""Acquisition of the official June-2026 Gaia DR4 prerelease archive.

The archive holds epoch astrometry for 12 sources.  It is downloaded once,
checksummed, and never re-fetched while the local copy still matches.
"""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

import numpy as np
from astropy.io.votable import parse_single_table
from astropy.io.votable.exceptions import VOTableSpecError
from astropy.table import Table

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.config import (
    PRERELEASE_MEMBER,
    PRERELEASE_N_SOURCES,
    AppConfig,
    CacheLayout,
)
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord


class PreReleaseError(RuntimeError):
    """Raised when the prerelease archive cannot be obtained or parsed."""


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Return the hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


class PreReleaseProvider:
    """Fetch, verify, extract and read the prerelease epoch-astrometry table.

    Parameters
    ----------
    config : AppConfig
        Resolved configuration.  Set ``local_prerelease_zip`` to use a fixture
        instead of the network, and ``allow_network=False`` to forbid downloads.

    Notes
    -----
    The table is read lazily: nothing is downloaded or parsed until one of the
    accessors is called.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._layout = CacheLayout(config.cache_dir)
        self._table: Table | None = None
        self._provenance: ProvenanceRecord | None = None

    # ------------------------------------------------------------------ paths

    @property
    def config(self) -> AppConfig:
        """The configuration this provider was built with (read-only)."""
        return self._config

    @property
    def zip_path(self) -> Path:
        """Location of the cached archive."""
        if self._config.local_prerelease_zip is not None:
            return Path(self._config.local_prerelease_zip)
        name = self._config.prerelease_url.rsplit("/", 1)[-1]
        return self._layout.prerelease / name

    @property
    def xml_path(self) -> Path:
        """Location the VOTable is extracted to."""
        return self._layout.prerelease / PRERELEASE_MEMBER

    # --------------------------------------------------------------- fetching

    def ensure_archive(self, *, force: bool = False) -> Path:
        """Return a path to a verified local copy of the archive.

        An unchanged local copy is never re-downloaded.  If the checksum does
        not match the expected value the file is rejected rather than used.
        """
        path = self.zip_path
        if path.exists() and not force:
            digest = sha256_of(path)
            if digest == self._config.prerelease_sha256:
                return path
            if self._config.local_prerelease_zip is not None:
                # A caller-supplied fixture is allowed to differ from the
                # official archive; that is the point of the override.
                return path
            raise PreReleaseError(
                f"checksum mismatch for {path}: expected "
                f"{self._config.prerelease_sha256}, got {digest}. "
                "Delete the file to re-download."
            )
        if self._config.local_prerelease_zip is not None:
            raise PreReleaseError(f"configured local archive does not exist: {path}")
        if not self._config.allow_network:
            raise PreReleaseError(
                f"{path} is missing and network access is disabled; "
                "run 'gaia-dr4-explorer fetch-prerelease' with network enabled"
            )
        return self._download(path)

    def _download(self, dest: Path) -> Path:
        import urllib.request

        self._layout.ensure(dest.parent)
        tmp = dest.with_suffix(dest.suffix + ".part")
        try:
            with urllib.request.urlopen(self._config.prerelease_url, timeout=120) as resp:
                with tmp.open("wb") as fh:
                    shutil.copyfileobj(resp, fh)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise PreReleaseError(
                f"could not download {self._config.prerelease_url}: {exc}"
            ) from exc
        digest = sha256_of(tmp)
        if digest != self._config.prerelease_sha256:
            tmp.unlink(missing_ok=True)
            raise PreReleaseError(
                f"downloaded archive has checksum {digest}, expected "
                f"{self._config.prerelease_sha256}"
            )
        tmp.replace(dest)
        return dest

    def ensure_extracted(self) -> Path:
        """Extract the VOTable member from the archive and return its path."""
        xml = self.xml_path
        if xml.exists():
            return xml
        archive = self.ensure_archive()
        self._layout.ensure(xml.parent)
        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                if PRERELEASE_MEMBER not in names:
                    raise PreReleaseError(
                        f"{PRERELEASE_MEMBER} not found in {archive}; members: {names}"
                    )
                member = zf.getinfo(PRERELEASE_MEMBER)
                # Refuse absolute or traversing member names before extracting.
                if Path(member.filename).is_absolute() or ".." in Path(member.filename).parts:
                    raise PreReleaseError(f"unsafe member name in archive: {member.filename!r}")
                with zf.open(member) as src, xml.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile as exc:
            raise PreReleaseError(f"{archive} is not a readable ZIP archive: {exc}") from exc
        return xml

    # ---------------------------------------------------------------- reading

    def load_table(self, *, reload: bool = False) -> Table:
        """Read the full prerelease VOTable.

        Returns
        -------
        astropy.table.Table
            The raw table exactly as published, with units and masks intact.
        """
        if self._table is not None and not reload:
            return self._table
        xml = self.ensure_extracted()
        try:
            # parse_single_table, not Table.read: the latter discards the
            # VOTable PARAMs, and one of them carries the release string that
            # provenance is required to record.
            votable = parse_single_table(xml)
            table = votable.to_table(use_names_over_ids=True)
        except (VOTableSpecError, ValueError, IndexError, OSError) as exc:
            raise PreReleaseError(f"could not parse {xml} as a VOTable: {exc}") from exc
        table.meta["votable_params"] = {
            p.name: p.value for p in votable.params if p.name is not None
        }
        _validate(table)
        self._table = table
        self._provenance = self._make_provenance(table)
        return table

    def _make_provenance(self, table: Table) -> ProvenanceRecord:
        archive = self.zip_path
        solution_ids = np.unique(np.asarray(table["solution_id"]))
        return ProvenanceRecord(
            kind="prerelease-archive",
            release=str(_scalar_param(table, "release", "unknown")),
            source_url=self._config.prerelease_url,
            sha256=sha256_of(archive) if archive.exists() else None,
            solution_id=int(solution_ids[0]) if solution_ids.size == 1 else None,
            steps=(f"read {PRERELEASE_MEMBER} with astropy Table.read(format='votable')",),
            software={"gaia_dr4_explorer": __version__},
            notes={
                "n_rows": len(table),
                "n_columns": len(table.colnames),
                "n_sources": int(np.unique(np.asarray(table["source_id"])).size),
                "local_archive": str(archive),
            },
        )

    @property
    def provenance(self) -> ProvenanceRecord:
        """Provenance of the loaded archive."""
        if self._provenance is None:
            self.load_table()
        assert self._provenance is not None
        return self._provenance

    def source_ids(self) -> list[int]:
        """Sorted unique source identifiers present in the archive."""
        table = self.load_table()
        return sorted(int(v) for v in np.unique(np.asarray(table["source_id"])))

    def raw_table_for(self, source_id: int) -> Table:
        """Return the raw rows belonging to one source.

        Raises
        ------
        KeyError
            If the identifier is not present.
        """
        table = self.load_table()
        mask = np.asarray(table["source_id"]) == int(source_id)
        if not mask.any():
            raise KeyError(
                f"source_id {source_id} is not in the prerelease; "
                f"available: {self.source_ids()}"
            )
        return table[mask]

    def release(self) -> str:
        """Release string declared by the VOTable, e.g. ``'Gaia DR4_RC3'``."""
        return str(_scalar_param(self.load_table(), "release", "unknown"))


def _scalar_param(table: Table, name: str, default: object = None) -> object:
    """Read a VOTable PARAM by name, returning *default* when absent.

    The parameters are stashed under ``table.meta["votable_params"]`` by
    :meth:`PreReleaseProvider.load_table`.
    """
    params = table.meta.get("votable_params") or {}
    value = params.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _validate(table: Table) -> None:
    """Check structural properties that must hold for any epoch-astrometry table.

    Deliberately permissive: unknown or additional columns must not cause a
    failure, and exact row counts are not asserted.  Only the invariants the
    normalizer depends on are enforced.
    """
    required = {"source_id", "transit_id", "obs_time_tcb", "centroid_pos_al"}
    missing = sorted(required - set(table.colnames))
    if missing:
        raise PreReleaseError(f"VOTable is missing required columns: {missing}")
    if len(table) == 0:
        raise PreReleaseError("VOTable contains no rows")
    n_sources = int(np.unique(np.asarray(table["source_id"])).size)
    if n_sources != PRERELEASE_N_SOURCES:
        # A warning-level fact, not fatal: DR4 proper will have more.
        table.meta.setdefault("gaia_dr4_explorer", {})["n_sources_unexpected"] = n_sources
