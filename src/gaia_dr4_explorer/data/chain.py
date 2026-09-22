"""Serve a product from the bundle when it is there, otherwise the archive.

The twelve prerelease sources ship with their DR3 products, so the desktop
application answers instantly and works offline; anything else still goes to the
archive. A browser build is given the bundled provider alone, because the Gaia
archive sends no CORS header and cannot be reached from a page.
"""

from __future__ import annotations

from astropy.table import Table

from gaia_dr4_explorer.data.archive import ArchiveError, ArchiveUnavailable
from gaia_dr4_explorer.data.bundled import BundledProductProvider
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord


class ProductChain:
    """Tries each provider in turn, preferring the bundle."""

    def __init__(self, bundled: BundledProductProvider, archive=None) -> None:
        self._bundled = bundled
        self._archive = archive

    def get_datalink_product(
        self, source_id: int, retrieval_type: str, *, release: str = "Gaia DR3",
        refresh: bool = False,
    ) -> tuple[Table, ProvenanceRecord]:
        if not refresh and self._bundled.has(source_id, retrieval_type):
            return self._bundled.get_datalink_product(
                source_id, retrieval_type, release=release
            )
        if self._archive is None:
            raise ArchiveUnavailable(
                f"{retrieval_type} for {source_id} is not bundled and no archive "
                "provider is available"
            )
        try:
            return self._archive.get_datalink_product(
                source_id, retrieval_type, release=release, refresh=refresh
            )
        except ArchiveError:
            raise

    def simbad(self) -> dict:
        return self._bundled.simbad()

    def ads(self) -> dict:
        return self._bundled.ads()
