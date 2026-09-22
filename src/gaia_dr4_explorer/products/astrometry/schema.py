"""Field registry for Gaia DR4 epoch astrometry.

Every column of the prerelease VOTable is described here: its original unit as
published, the unit we canonicalise to, a description, and whether it varies per
source, per FoV transit, or per CCD observation.  Plotting code reads units from
this registry; it never converts angles implicitly.

Verified against ``GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml`` on 2026-09-22
(release ``Gaia DR4_RC3``, 1008 rows, 37 columns).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Level(StrEnum):
    """Granularity at which a field varies."""

    SOURCE = "source"
    TRANSIT = "transit"
    CCD = "ccd"


@dataclass(frozen=True)
class FieldSpec:
    """Description of one published field.

    Parameters
    ----------
    name : str
        Column name exactly as published.
    level : Level
        Whether the value varies per source, per transit, or per CCD.
    unit : str or None
        Unit as declared in the VOTable, or None if dimensionless.
    canonical_unit : str or None
        Unit the application displays it in.  Equal to *unit* unless a
        conversion is applied, and a conversion is only ever applied in a
        derived column that sits alongside the original.
    description : str
        Short description, taken from the VOTable where one exists.
    is_bitmask : bool
        True for signed integer columns that carry unsigned bit flags.  These
        must be cast to an unsigned dtype before any bit test.
    may_be_empty : bool
        True for array columns published as zero-length in this release.
    """

    name: str
    level: Level
    unit: str | None
    canonical_unit: str | None
    description: str
    is_bitmask: bool = False
    may_be_empty: bool = False


def _f(name, level, unit, description, *, canonical=..., bitmask=False, empty=False):
    return FieldSpec(
        name=name,
        level=level,
        unit=unit,
        canonical_unit=unit if canonical is ... else canonical,
        description=description,
        is_bitmask=bitmask,
        may_be_empty=empty,
    )


_S, _T, _C = Level.SOURCE, Level.TRANSIT, Level.CCD

#: All 37 published fields, in VOTable order.
FIELDS: dict[str, FieldSpec] = {
    spec.name: spec
    for spec in (
        _f("solution_id", _S, None, "Solution identifier"),
        _f("source_id", _S, None, "Unique source identifier, specific to this release"),
        _f("transit_id", _T, None, "Transit identifier; encodes the field of view"),
        _f("ra0", _T, "deg", "Right ascension of reference point"),
        _f("dec0", _T, "deg", "Declination of reference point"),
        _f("agis_source_excess_noise", _S, "mas", "Unscaled AGIS source excess noise"),
        _f(
            "obs_time_tcb", _C, "ns",
            "Effective observing time as TCB **at Gaia**, from the time origin; "
            "add obs_time_bary_corr for barycentric time",
        ),
        _f(
            "obs_time_bary_corr", _T, "ns",
            "Barycentric correction, in the sense TCB(barycentric) - TCB(at Gaia), "
            "computed at the AF4 CCD",
        ),
        _f("scan_pos_angle", _C, "deg", "Position angle of the scan"),
        _f("zeta", _T, "deg", "Across-scan field angle zeta of the reference point"),
        _f("parallax_factor_al", _T, None, "Parallax factor along scan"),
        _f("parallax_factor_ac", _T, None, "Parallax factor across scan"),
        _f("colour_factor_al", _C, None, "Colour factor along scan"),
        _f("colour_factor_ac", _C, None, "Colour factor across scan"),
        _f("nu_eff_used_in_astrometry", _T, "1 / um", "Effective wavenumber used in AGIS and IPD"),
        _f("nu_eff_error", _T, "1 / um", "Standard error of nu_eff_used_in_astrometry"),
        _f("centroid_pos_al", _C, "mas", "Local plane coordinate (w) of image centroid AL"),
        _f("centroid_pos_ac", _C, "mas", "Local plane coordinate (z) of image centroid AC", empty=True),
        _f("calculated_pos_ac", _C, "mas", "Calculated local plane coordinate (z)"),
        _f("centroid_pos_error_al", _C, "mas", "Standard error of centroid_pos_al"),
        _f("centroid_pos_error_ac", _C, "mas", "Standard error of centroid_pos_ac", empty=True),
        _f("used_by_agis_al", _C, None, "Whether the CCD transit was used in AL by AGIS-DR4"),
        _f("used_by_agis_ac", _C, None, "Whether the CCD transit was used in AC by AGIS-DR4"),
        _f("transit_acq_flags", _T, None, "On-board acquisition information", bitmask=True),
        _f("transit_proc_flags", _T, None, "Transit-level processing information", bitmask=True),
        _f("ccd_proc_flags", _C, None, "CCD-level processing information", bitmask=True),
        _f("multipeak", _T, None, "Multiple peaks detected in the window"),
        _f("blended", _T, None, "Window blended with another source"),
        _f("ipd_error_al", _C, "mas", "Image parameter determination error AL"),
        _f("ipd_error_ac", _C, "mas", "Image parameter determination error AC"),
        _f("g_mag", _T, "mag", "G magnitude"),
        _f("g_class", _T, None, "Window class"),
        _f("gates", _C, None, "Gate used for the observation", bitmask=True),
        _f("source_dist_to_last_ci", _C, "pix", "Distance to the last charge injection"),
        _f("ac_rate", _T, "pix / s", "Across-scan rate"),
        _f("sub_pixel_coord", _C, "pix", "Sub-pixel coordinate"),
        _f("mu", _C, "pix", "Window reference pixel coordinate"),
    )
}

#: Columns that vary per CCD observation.
CCD_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELDS.items() if s.level is Level.CCD)

#: Columns constant within a transit, broadcast to CCD level when flattening.
TRANSIT_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELDS.items() if s.level is Level.TRANSIT)

#: Columns constant for a source.
SOURCE_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELDS.items() if s.level is Level.SOURCE)

#: Signed integer columns carrying unsigned bitmasks.
BITMASK_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELDS.items() if s.is_bitmask)

#: Array columns published as zero-length in Gaia DR4_RC3.
MAY_BE_EMPTY_FIELDS: tuple[str, ...] = tuple(n for n, s in FIELDS.items() if s.may_be_empty)


def describe(name: str) -> FieldSpec | None:
    """Return the registry entry for *name*, or None for an unknown column.

    Unknown columns are expected: the parser must not fail on fields added
    after this registry was written.
    """
    return FIELDS.get(name)
