"""Gaia DR3 products bundled with the package.

The Gaia archive sends no ``Access-Control-Allow-Origin`` header (verified
2026-09-22), so a browser can never fetch DataLink products directly. For the
twelve prerelease sources the products are therefore shipped with the package,
which also makes the desktop application work with no network at all.

This provider exposes the same surface as :class:`GaiaArchiveProvider`, so a
product plugin cannot tell them apart.
"""

from __future__ import annotations

import io
import json
import zipfile
from functools import lru_cache
from importlib.resources import files

from astropy.io.votable import parse_single_table
from astropy.table import Table

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.data.archive import (
    EPOCH_PHOTOMETRY,
    XP_SAMPLED,
    ArchiveError,
    ArchiveUnavailable,
)
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord

BUNDLE = files("gaia_dr4_explorer.resources") / "dr3_products_bundle.zip"

#: Retrieval type to the directory holding it inside the bundle.
_MEMBERS = {EPOCH_PHOTOMETRY: "photometry", XP_SAMPLED: "xp"}

#: Release the bundled products come from, and when they were retrieved.
RELEASE = "Gaia DR3"
RETRIEVED_ON = "2026-09-22"


@lru_cache(maxsize=1)
def _archive_bytes() -> bytes:
    if not BUNDLE.is_file():
        return b""
    return BUNDLE.read_bytes()


@lru_cache(maxsize=1)
def _names() -> frozenset[str]:
    data = _archive_bytes()
    if not data:
        return frozenset()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        return frozenset(z.namelist())


def _read_member(name: str) -> bytes:
    with zipfile.ZipFile(io.BytesIO(_archive_bytes())) as z:
        return z.read(name)


class BundledProductProvider:
    """Serves the shipped DR3 products, and nothing else."""

    release = RELEASE

    def has(self, source_id: int, retrieval_type: str) -> bool:
        """True when this product is bundled for this source. No I/O beyond the zip."""
        member = _MEMBERS.get(retrieval_type)
        return bool(member) and f"{member}/{int(source_id)}.vot" in _names()

    def get_datalink_product(
        self,
        source_id: int,
        retrieval_type: str,
        *,
        release: str = RELEASE,
        refresh: bool = False,
    ) -> tuple[Table, ProvenanceRecord]:
        """Return a bundled product, with the same signature as the archive provider."""
        del refresh  # a bundled product is fixed; there is nothing to refresh
        member = _MEMBERS.get(retrieval_type)
        if member is None:
            raise ArchiveError(
                f"{retrieval_type} is not bundled; known: {sorted(_MEMBERS)}"
            )
        name = f"{member}/{int(source_id)}.vot"
        if name not in _names():
            raise ArchiveUnavailable(
                f"{RELEASE} has no {retrieval_type} bundled for source {source_id}"
            )
        try:
            votable = parse_single_table(io.BytesIO(_read_member(name)))
            table = votable.to_table(use_names_over_ids=True)
        except Exception as exc:
            raise ArchiveError(f"could not parse bundled {name}: {exc}") from exc
        table.meta["votable_params"] = {
            p.name: p.value for p in votable.params if p.name is not None
        }
        record = ProvenanceRecord(
            kind=f"bundled-{retrieval_type.lower()}",
            release=RELEASE,
            source_id=int(source_id),
            steps=(
                f"read {name} from the DR3 product bundle shipped with this package, "
                f"retrieved from the Gaia archive on {RETRIEVED_ON}",
            ),
            software={"gaia_dr4_explorer": __version__},
            notes={"n_rows": len(table), "bundled": True},
        )
        return table, record

    # ------------------------------------------------------------- context

    @lru_cache(maxsize=1)  # noqa: B019 - one instance per process, bounded by the bundle
    def simbad(self) -> dict[int, dict]:
        """SIMBAD identity for the sources SIMBAD knows, keyed by source_id."""
        if "simbad.json" not in _names():
            return {}
        return {int(k): v for k, v in json.loads(_read_member("simbad.json")).items()}

    @lru_cache(maxsize=1)  # noqa: B019
    def ads(self) -> dict[int, dict]:
        """ADS bibliography, keyed by source_id. Public results only."""
        if "ads.json" not in _names():
            return {}
        return {int(k): v for k, v in json.loads(_read_member("ads.json")).items()}
