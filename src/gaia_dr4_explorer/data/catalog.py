"""Metadata for the 12 prerelease sources.

Three kinds of number are kept apart on purpose, because the interface must
always be able to say which is which:

* ``page_*``   -- quoted on the ESA prerelease page, rounded, not measured here;
* ``measured`` -- computed by us from the VOTable;
* ``fit_*``    -- computed by us with ``gaiasupdate``.

The reference table also serves the static build, which cannot run the fit.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

#: Shipped alongside the package docs.
REFERENCE_CSV = Path(__file__).resolve().parents[3] / "docs" / "prerelease_reference.csv"

#: Category labels used on the ESA prerelease page.
CATEGORY_LABELS = {
    "parallax": "Parallax example",
    "magnitude": "Magnitude example",
    "qso": "QSO",
    "orbit": "Orbit / binary",
}


@dataclass(frozen=True)
class CatalogEntry:
    """One prerelease source, as described by the release page."""

    source_id: int
    common_name: str
    sample_category: str
    page_g_mag: float
    page_parallax_mas: float

    @property
    def label(self) -> str:
        """Dropdown label: name if it has one, else the category and G."""
        if self.common_name:
            return f"{self.common_name}  (G={self.page_g_mag:.1f})"
        return f"{self.category_label}  G={self.page_g_mag:.1f}  ({self.source_id})"

    @property
    def category_label(self) -> str:
        return CATEGORY_LABELS.get(self.sample_category, self.sample_category)


@lru_cache(maxsize=1)
def _rows() -> tuple[dict[str, str], ...]:
    if not REFERENCE_CSV.exists():
        return ()
    with REFERENCE_CSV.open() as fh:
        return tuple(csv.DictReader(fh))


@lru_cache(maxsize=1)
def catalog() -> dict[int, CatalogEntry]:
    """Release-page metadata, keyed by source identifier."""
    out: dict[int, CatalogEntry] = {}
    for row in _rows():
        sid = int(row["source_id"])
        out[sid] = CatalogEntry(
            source_id=sid,
            common_name=row.get("common_name", "") or "",
            sample_category=row.get("sample_category", "") or "",
            page_g_mag=float(row["g_mag_page"]),
            page_parallax_mas=float(row["parallax_page_mas"]),
        )
    return out


@lru_cache(maxsize=1)
def reference_fits() -> dict[int, dict[str, float]]:
    """Precomputed fit results, for builds that cannot run ``gaiasupdate``."""
    out: dict[int, dict[str, float]] = {}
    for row in _rows():
        out[int(row["source_id"])] = {
            k: float(v)
            for k, v in row.items()
            if k.startswith("fit_") and v not in ("", None)
        }
    return out


def entry(source_id: int) -> CatalogEntry | None:
    return catalog().get(int(source_id))
