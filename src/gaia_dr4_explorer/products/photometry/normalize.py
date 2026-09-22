"""Normalization of Gaia epoch photometry.

DataLink serves epoch photometry in a wide layout: one row per FoV transit with
separate G, BP and RP columns.  Light curves are easier to reason about in a
long layout, one row per band measurement, so that is what this produces --
while keeping the raw wide table untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.table import Column, MaskedColumn, Table, vstack

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord

#: Bands and the columns DataLink serves for each.
#:
#: Verified against a real DR3 EPOCH_PHOTOMETRY response on 2026-09-22.  The
#: per-band rejection flags are ``variability_flag_<band>_reject``; there is no
#: ``bp_rejected_by_photometry`` or ``rp_rejected_by_photometry``.
#: ``rejected_by_photometry`` is transit-level and is carried separately.
BANDS: dict[str, dict[str, str]] = {
    "G": {
        "time": "g_transit_time",
        "mag": "g_transit_mag",
        "flux": "g_transit_flux",
        "flux_error": "g_transit_flux_error",
        "flux_over_error": "g_transit_flux_over_error",
        "rejected": "variability_flag_g_reject",
        "other_flags": "g_other_flags",
    },
    "BP": {
        "time": "bp_obs_time",
        "mag": "bp_mag",
        "flux": "bp_flux",
        "flux_error": "bp_flux_error",
        "flux_over_error": "bp_flux_over_error",
        "rejected": "variability_flag_bp_reject",
        "other_flags": "bp_other_flags",
    },
    "RP": {
        "time": "rp_obs_time",
        "mag": "rp_mag",
        "flux": "rp_flux",
        "flux_error": "rp_flux_error",
        "flux_over_error": "rp_flux_over_error",
        "rejected": "variability_flag_rp_reject",
        "other_flags": "rp_other_flags",
    },
}

#: Transit-level rejection flag, applying to the transit rather than one band.
TRANSIT_REJECTED_COLUMN = "rejected_by_photometry"

#: Gaia photometric time is barycentric JD offset by this, in days.
JD_OFFSET = 2455197.5


class PhotometryNormalizationError(ValueError):
    """The served table does not have the structure this parser expects."""


@dataclass
class NormalizedPhotometry:
    """Wide raw table plus the long per-band representation."""

    raw: Table
    epochs: Table
    provenance: ProvenanceRecord
    warnings: list[str] = field(default_factory=list)

    @property
    def bands_present(self) -> list[str]:
        if len(self.epochs) == 0:
            return []
        return sorted(set(np.asarray(self.epochs["band"]).tolist()))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {"n_transits": len(self.raw), "n_measurements": len(self.epochs)}
        if len(self.epochs) == 0:
            return out
        band = np.asarray(self.epochs["band"])
        rejected = np.asarray(np.ma.filled(self.epochs["rejected"], False), dtype=bool)
        mag = np.asarray(np.ma.filled(self.epochs["mag"], np.nan), dtype="float64")
        time = np.asarray(np.ma.filled(self.epochs["time_jd_tcb"], np.nan), dtype="float64")
        for b in sorted(set(band.tolist())):
            m = (band == b) & np.isfinite(mag)
            if not m.any():
                continue
            out[f"n_{b}"] = int(m.sum())
            out[f"n_{b}_rejected"] = int((m & rejected).sum())
            out[f"median_{b}_mag"] = float(np.nanmedian(mag[m]))
            out[f"ptp_{b}_mag"] = float(np.nanmax(mag[m]) - np.nanmin(mag[m]))
        finite_t = time[np.isfinite(time)]
        if finite_t.size:
            out["span_yr"] = float((finite_t.max() - finite_t.min()) / 365.25)
        return out


def normalize_epoch_photometry(
    raw: Table, *, provenance: ProvenanceRecord | None = None
) -> NormalizedPhotometry:
    """Turn the wide DataLink table into one row per band measurement.

    Parameters
    ----------
    raw : astropy.table.Table
        The served table.  Not modified.
    provenance : ProvenanceRecord, optional
        Provenance of *raw*.

    Returns
    -------
    NormalizedPhotometry
    """
    if len(raw) == 0:
        raise PhotometryNormalizationError("epoch photometry table is empty")

    warnings: list[str] = []
    blocks: list[Table] = []

    for band, cols in BANDS.items():
        if cols["time"] not in raw.colnames or cols["mag"] not in raw.colnames:
            warnings.append(f"{band}: not served for this source")
            continue
        n = len(raw)
        block = Table()
        block["band"] = Column(np.full(n, band), description="Photometric band")
        block["transit_id"] = _carry(raw, "transit_id", n)
        block["source_id"] = _carry(raw, "source_id", n)
        block["time_bjd_offset"] = _carry(raw, cols["time"], n, unit="d")
        block["time_jd_tcb"] = Column(
            np.asarray(np.ma.filled(raw[cols["time"]], np.nan), dtype="float64") + JD_OFFSET,
            unit="d",
            description=(
                f"Derived from {cols['time']} by adding the Gaia time origin "
                f"({JD_OFFSET}); the original column is kept unchanged"
            ),
        )
        block["mag"] = _carry(raw, cols["mag"], n, unit="mag")
        for key in ("flux", "flux_error", "flux_over_error", "other_flags"):
            name = cols.get(key)
            block[key] = (
                _carry(raw, name, n) if name and name in raw.colnames else _missing(n)
            )
        rej = cols.get("rejected")
        if rej and rej in raw.colnames:
            block["rejected"] = Column(
                np.asarray(np.ma.filled(raw[rej], False), dtype=bool),
                description=f"Per-band rejection flag, from {rej}",
            )
        else:
            # Never quietly present unknown-quality data as accepted.
            block["rejected"] = MaskedColumn(
                np.zeros(n, dtype=bool), mask=np.ones(n, dtype=bool),
                description="No per-band rejection flag served; quality unknown",
            )
            warnings.append(
                f"{band}: {rej!r} was not served, so per-band rejection is unknown "
                "and is shown as such rather than as accepted"
            )
        if TRANSIT_REJECTED_COLUMN in raw.colnames:
            block["transit_rejected"] = Column(
                np.asarray(np.ma.filled(raw[TRANSIT_REJECTED_COLUMN], False), dtype=bool),
                description=f"Transit-level flag, from {TRANSIT_REJECTED_COLUMN}",
            )
        else:
            block["transit_rejected"] = MaskedColumn(
                np.zeros(n, dtype=bool), mask=np.ones(n, dtype=bool),
                description="Transit-level rejection flag not served",
            )
        blocks.append(block)

    if not blocks:
        raise PhotometryNormalizationError(
            f"no recognised photometric band columns in {sorted(raw.colnames)}"
        )

    epochs = vstack(blocks, metadata_conflicts="silent")
    finite = np.isfinite(np.asarray(np.ma.filled(epochs["mag"], np.nan), dtype="float64"))
    dropped = int((~finite).sum())
    epochs = epochs[finite]
    if dropped:
        warnings.append(
            f"{dropped} band-epoch slots had no magnitude and are not plotted; "
            "the raw table keeps them"
        )
    epochs.sort(["band", "time_jd_tcb"])

    parent = provenance or ProvenanceRecord(kind="unknown-raw", release="unknown")
    derived = parent.derive(
        "epoch-photometry-normalized",
        "reshaped the wide per-transit table to one row per band measurement",
        f"added time_jd_tcb = {BANDS['G']['time']} + {JD_OFFSET}",
        "dropped band-epoch slots with no magnitude from the long table only",
        n_measurements=len(epochs),
    )
    derived.software["gaia_dr4_explorer"] = __version__
    return NormalizedPhotometry(raw=raw, epochs=epochs, provenance=derived, warnings=warnings)


def _carry(raw: Table, name: str, n: int, *, unit: str | None = None):
    if name not in raw.colnames:
        return _missing(n)
    col = raw[name].copy()
    if unit is not None and col.unit is None:
        col.unit = unit
    return col


def _missing(n: int) -> MaskedColumn:
    return MaskedColumn(np.zeros(n, dtype="float64"), mask=np.ones(n, dtype=bool))
