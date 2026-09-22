"""Which DR3 products exist for the prerelease sources.

Shipped as data so a plugin can answer "is this available?" without a network
call.  Queried from ``gaiadr3.gaia_source`` on 2026-09-22; the flags are
properties of DR3 and do not change.

Important: a DR3 flag says nothing about DR4.  Epoch photometry exists for
every source in DR4, but DR4 is not public until 2026-12-02.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DR3_PRODUCTS_CSV = Path(__file__).resolve().parents[3] / "docs" / "dr3_products.csv"

#: The release these flags describe.
RELEASE = "Gaia DR3"

#: When the flags were read from the archive.
CHECKED_ON = "2026-09-22"


@dataclass(frozen=True)
class Dr3Availability:
    """DR3 product flags and mean photometry for one source."""

    source_id: int
    has_epoch_photometry: bool
    has_xp_continuous: bool
    has_xp_sampled: bool
    has_rvs: bool
    phot_variable_flag: str
    g_mag: float
    bp_mag: float
    rp_mag: float

    @property
    def bp_rp(self) -> float:
        return self.bp_mag - self.rp_mag


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


@lru_cache(maxsize=1)
def dr3_availability() -> dict[int, Dr3Availability]:
    """Flags keyed by source identifier, empty if the table is missing."""
    if not DR3_PRODUCTS_CSV.exists():
        return {}
    out: dict[int, Dr3Availability] = {}
    with DR3_PRODUCTS_CSV.open() as fh:
        for row in csv.DictReader(fh):
            sid = int(row["source_id"])
            out[sid] = Dr3Availability(
                source_id=sid,
                has_epoch_photometry=_as_bool(row["has_epoch_photometry"]),
                has_xp_continuous=_as_bool(row["has_xp_continuous"]),
                has_xp_sampled=_as_bool(row["has_xp_sampled"]),
                has_rvs=_as_bool(row["has_rvs"]),
                phot_variable_flag=row["phot_variable_flag"],
                g_mag=float(row["phot_g_mean_mag"]),
                bp_mag=float(row["phot_bp_mean_mag"]),
                rp_mag=float(row["phot_rp_mean_mag"]),
            )
    return out


def for_source(source_id: int) -> Dr3Availability | None:
    return dr3_availability().get(int(source_id))
