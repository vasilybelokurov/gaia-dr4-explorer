"""Normalization of Gaia DR4 epoch astrometry.

Turns the published transit-level VOTable -- in which the per-CCD quantities are
variable-length array cells -- into a flat CCD-level table, while keeping the
raw table untouched.

Design rules enforced here:

* every array column's length is validated explicitly before flattening; there
  is no blanket ``DataFrame.explode``;
* a missing value stays missing, never becomes zero;
* signed integer bitmask columns are cast to unsigned before anything can test
  a bit on them;
* the published timestamp is preserved and derived time columns sit beside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from astropy.table import Column, MaskedColumn, Table

from gaia_dr4_explorer import __version__
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord
from gaia_dr4_explorer.products.astrometry import times
from gaia_dr4_explorer.products.astrometry.schema import FIELDS

#: Focal-plane strips, in published array order.
CCD_NAMES: tuple[str, ...] = ("SM", "AF1", "AF2", "AF3", "AF4", "AF5", "AF6", "AF7", "AF8", "AF9")

#: Number of CCD observations per FoV transit in Gaia DR4_RC3.
N_CCD = len(CCD_NAMES)

_UNSIGNED_FOR = {"int8": "uint8", "int16": "uint16", "int32": "uint32", "int64": "uint64"}


class NormalizationError(ValueError):
    """Raised when the published table violates an invariant we depend on."""


@dataclass
class NormalizedAstrometry:
    """The three representations of one source's epoch astrometry.

    Attributes
    ----------
    raw : astropy.table.Table
        The published rows for this source, untouched.
    transits : astropy.table.Table
        One row per FoV transit: the scalar columns plus per-transit summaries.
    ccd : astropy.table.Table
        One row per CCD observation, with transit-level values broadcast.
    provenance : ProvenanceRecord
        How the derived tables were produced.
    warnings : list of str
        Non-fatal observations, e.g. columns that are constant or empty.
    """

    raw: Table
    transits: Table
    ccd: Table
    provenance: ProvenanceRecord
    warnings: list[str] = field(default_factory=list)

    @property
    def source_id(self) -> int:
        return int(np.asarray(self.raw["source_id"])[0])

    def summary(self) -> dict[str, Any]:
        """Counts and spans computed from the data, not quoted from anywhere."""
        used = np.asarray(_filled(self.ccd["used_by_agis_al"], False), dtype=bool)
        al = np.asarray(_filled(self.ccd["centroid_pos_al"], np.nan), dtype="float64")
        err = np.asarray(_filled(self.ccd["centroid_pos_error_al"], np.nan), dtype="float64")
        jd = np.asarray(self.ccd["obs_time_jd_tcb"], dtype="float64")
        finite = np.isfinite(al)
        return {
            "source_id": self.source_id,
            "n_transits": len(self.transits),
            "n_ccd_slots": len(self.ccd),
            "n_ccd_finite": int(finite.sum()),
            "n_used_by_agis_al": int(used.sum()),
            "frac_used": float(used.sum() / len(self.ccd)) if len(self.ccd) else float("nan"),
            "t_start_jd_tcb": float(np.nanmin(jd)) if jd.size else float("nan"),
            "t_end_jd_tcb": float(np.nanmax(jd)) if jd.size else float("nan"),
            "span_yr": float((np.nanmax(jd) - np.nanmin(jd)) / 365.25) if jd.size else float("nan"),
            "median_sigma_al_used_mas": (
                float(np.nanmedian(err[used])) if used.any() else float("nan")
            ),
            "agis_source_excess_noise_mas": float(
                np.asarray(self.raw["agis_source_excess_noise"])[0]
            ),
        }


def normalize_epoch_astrometry(
    raw: Table,
    *,
    provenance: ProvenanceRecord | None = None,
    expected_n_ccd: int | None = N_CCD,
) -> NormalizedAstrometry:
    """Flatten one source's epoch astrometry to CCD level.

    Parameters
    ----------
    raw : astropy.table.Table
        Rows for a single source, as published.  Not modified.
    provenance : ProvenanceRecord, optional
        Provenance of *raw*, recorded as the parent of the derived tables.
    expected_n_ccd : int or None
        Array length every per-CCD column must have.  ``None`` infers it from
        the first non-empty column instead of asserting a value.

    Returns
    -------
    NormalizedAstrometry

    Raises
    ------
    NormalizationError
        If more than one source is present, or if array lengths are
        inconsistent between columns or between rows.
    """
    if len(raw) == 0:
        raise NormalizationError("cannot normalize an empty table")
    source_ids = np.unique(np.asarray(raw["source_id"]))
    if source_ids.size != 1:
        raise NormalizationError(
            f"expected exactly one source, found {source_ids.size}: {source_ids.tolist()}"
        )

    warnings: list[str] = []
    array_cols, empty_cols, scalar_cols = _classify_columns(raw, warnings)
    n_ccd = _resolve_n_ccd(raw, array_cols, expected_n_ccd)
    n_transits = len(raw)

    transits = _build_transit_table(raw, scalar_cols, array_cols, n_ccd)
    ccd = _build_ccd_table(raw, array_cols, scalar_cols, empty_cols, n_ccd, warnings)

    for name in empty_cols:
        warnings.append(
            f"{name!r} is published as a zero-length array in every row of this release; "
            "the column is retained and fully masked"
        )
    _warn_constant(raw, ("multipeak", "blended"), warnings)

    parent = provenance or ProvenanceRecord(kind="unknown-raw", release="unknown")
    derived = parent.derive(
        "epoch-astrometry-normalized",
        f"selected {n_transits} transit rows for source_id {int(source_ids[0])}",
        f"validated every per-CCD array column has length {n_ccd}",
        f"flattened to {n_transits * n_ccd} CCD observations, preserving array order",
        "added ccd_index and ccd_name, and fov decoded from transit_id",
        "cast signed bitmask columns to unsigned",
        "added derived time columns beside the original obs_time_tcb",
        n_transits=n_transits,
        n_ccd_per_transit=n_ccd,
        empty_columns=list(empty_cols),
    )
    derived.software["gaia_dr4_explorer"] = __version__

    return NormalizedAstrometry(
        raw=raw, transits=transits, ccd=ccd, provenance=derived, warnings=warnings
    )


# --------------------------------------------------------------------- helpers


def _classify_columns(
    raw: Table, warnings: list[str]
) -> tuple[list[str], list[str], list[str]]:
    """Split columns into per-CCD arrays, empty arrays, and scalars."""
    array_cols: list[str] = []
    empty_cols: list[str] = []
    scalar_cols: list[str] = []
    for name in raw.colnames:
        lengths = _cell_lengths(raw[name])
        if lengths is None:
            scalar_cols.append(name)
        elif set(lengths.tolist()) <= {0}:
            empty_cols.append(name)
        else:
            array_cols.append(name)
        spec = FIELDS.get(name)
        if spec is None:
            warnings.append(f"{name!r} is not in the field registry; carried through unchanged")
    return array_cols, empty_cols, scalar_cols


def _cell_lengths(col: Column | MaskedColumn) -> np.ndarray | None:
    """Return per-row cell lengths for a variable-length column, else None."""
    if col.ndim > 1:
        return np.full(len(col), col.shape[1], dtype=int)
    if col.dtype != np.dtype(object):
        return None
    lengths = np.empty(len(col), dtype=int)
    for i, value in enumerate(col):
        if value is None or value is np.ma.masked:
            lengths[i] = 0
        else:
            lengths[i] = np.size(np.atleast_1d(value))
    return lengths


def _resolve_n_ccd(raw: Table, array_cols: list[str], expected: int | None) -> int:
    """Validate that every per-CCD column has one consistent length."""
    if not array_cols:
        raise NormalizationError("no per-CCD array columns found")
    observed: dict[str, set[int]] = {}
    for name in array_cols:
        lengths = _cell_lengths(raw[name])
        assert lengths is not None
        observed[name] = set(lengths.tolist())
    inconsistent = {n: sorted(v) for n, v in observed.items() if len(v) != 1}
    if inconsistent:
        raise NormalizationError(
            "per-CCD array columns have inconsistent lengths between rows: "
            f"{inconsistent}. Refusing to flatten, because the CCD identity of each "
            "value could not be preserved."
        )
    lengths = {n: next(iter(v)) for n, v in observed.items()}
    distinct = set(lengths.values())
    if len(distinct) != 1:
        raise NormalizationError(
            f"per-CCD array columns disagree on length: {lengths}. "
            "Refusing to flatten."
        )
    n_ccd = distinct.pop()
    if expected is not None and n_ccd != expected:
        raise NormalizationError(
            f"expected {expected} CCD observations per transit, found {n_ccd}. "
            "Pass expected_n_ccd=None to accept a different focal-plane layout."
        )
    return n_ccd


def _build_transit_table(
    raw: Table, scalar_cols: list[str], array_cols: list[str], n_ccd: int
) -> Table:
    """One row per transit: scalars, plus per-transit counts and the FoV."""
    out = Table()
    for name in scalar_cols:
        out[name] = _copy_column(raw[name], name)
    out["fov"] = Column(
        [fov_from_transit_id(int(t)) for t in np.asarray(raw["transit_id"])],
        name="fov",
        description="Gaia field of view (1 or 2), decoded from transit_id",
    )
    if "used_by_agis_al" in array_cols:
        used = _stack(raw["used_by_agis_al"], n_ccd, dtype=bool, fill=False)
        out["n_ccd_used_by_agis_al"] = Column(
            used.sum(axis=1), description="Number of CCD observations AGIS used in AL"
        )
    if "centroid_pos_al" in array_cols:
        al = _stack(raw["centroid_pos_al"], n_ccd, dtype="float64", fill=np.nan)
        out["n_ccd_finite_al"] = Column(
            np.isfinite(al).sum(axis=1), description="Number of finite AL centroids"
        )
    if "obs_time_tcb" in array_cols:
        t = _stack(raw["obs_time_tcb"], n_ccd, dtype="int64", fill=0)
        out["transit_time_jd_tcb"] = Column(
            times.tcb_ns_to_jd(t[:, 0]),
            unit="d",
            description="JD (TCB) of the first CCD observation of the transit",
        )
    return out


def _build_ccd_table(
    raw: Table,
    array_cols: list[str],
    scalar_cols: list[str],
    empty_cols: list[str],
    n_ccd: int,
    warnings: list[str],
) -> Table:
    """One row per CCD observation, array order preserved."""
    n_transits = len(raw)
    total = n_transits * n_ccd
    out = Table()

    out["ccd_index"] = Column(
        np.tile(np.arange(n_ccd, dtype="int8"), n_transits),
        description="Position in the published per-CCD array (0 = SM, 1..9 = AF1..AF9)",
    )
    names = np.array(CCD_NAMES[:n_ccd] if n_ccd <= len(CCD_NAMES) else
                     [f"CCD{i}" for i in range(n_ccd)])
    out["ccd_name"] = Column(np.tile(names, n_transits), description="Focal-plane strip")

    for name in array_cols:
        out[name] = _flatten_array_column(raw[name], n_ccd, name)

    for name in scalar_cols:
        out[name] = _broadcast_column(raw[name], n_ccd, name)

    # An empty array column carries no values, but dropping it would hide the
    # fact that the release published it at all.  Keep it, fully masked.
    for name in empty_cols:
        out[name] = MaskedColumn(
            np.zeros(total, dtype="float64"),
            mask=np.ones(total, dtype=bool),
            unit=_unit(raw[name]),
            description=_desc(raw[name], name),
        )

    out["fov"] = Column(
        np.repeat([fov_from_transit_id(int(t)) for t in np.asarray(raw["transit_id"])], n_ccd),
        description="Gaia field of view (1 or 2), decoded from transit_id",
    )

    if "obs_time_tcb" in out.colnames:
        ns = np.asarray(_filled(out["obs_time_tcb"], 0), dtype="int64")
        out["obs_time_jd_tcb"] = Column(
            times.tcb_ns_to_jd(ns),
            unit="d",
            description="Derived from obs_time_tcb; the original column is unchanged",
        )
        out["obs_time_jyear_tcb"] = Column(
            times.tcb_ns_to_jyear(ns),
            unit="yr",
            description="Derived from obs_time_tcb; the original column is unchanged",
        )
        if "obs_time_bary_corr" in out.colnames:
            corr_col = out["obs_time_bary_corr"]
            corr = np.asarray(_filled(corr_col, np.nan), dtype="float64")
            bary_ns = times.barycentric_ns(ns, corr)
            mask = ~np.isfinite(bary_ns)
            jd = np.where(mask, np.nan, times.tcb_ns_to_jd(np.nan_to_num(bary_ns)))
            rel = np.where(
                mask, np.nan,
                times.relative_time_year(ns, np.nan_to_num(corr)),
            )
            out["obs_time_jd_tcb_barycentric"] = MaskedColumn(
                jd, mask=mask, unit="d",
                description=(
                    "obs_time_tcb + obs_time_bary_corr, i.e. TCB at the solar-system "
                    "barycentre; the correction is computed at AF4, so it is exact at "
                    "mid-transit and approximate for the other CCDs"
                ),
            )
            out["relative_time_year"] = MaskedColumn(
                rel, mask=mask, unit="yr",
                description=(
                    "Barycentric TCB years relative to the DR4 reference epoch "
                    f"J{times.DR4_REFERENCE_EPOCH_JYEAR}"
                ),
            )

    if len(out) != total:
        raise NormalizationError(f"expected {total} CCD rows, built {len(out)}")
    _check_nan_not_used(out, warnings)
    return out


def _flatten_array_column(col: Column | MaskedColumn, n_ccd: int, name: str) -> Column:
    """Flatten a variable-length array column, preserving order and masks."""
    spec = FIELDS.get(name)
    sample = _first_cell(col)
    if sample is None:
        dtype = np.dtype(object)
    else:
        dtype = np.asarray(sample).dtype
        if dtype == np.dtype(object):
            # Nested object cells: infer the element type from the values.
            dtype = np.asarray(np.asarray(sample).tolist()).dtype
    if np.issubdtype(dtype, np.integer) and spec is not None and spec.is_bitmask:
        dtype = np.dtype(_UNSIGNED_FOR.get(dtype.name, dtype.name))
    stacked = _stack(col, n_ccd, dtype=dtype, fill=_fill_for(dtype))
    mask = _stack_mask(col, n_ccd)
    flat = stacked.reshape(-1)
    if mask is not None and mask.any():
        return MaskedColumn(
            flat, mask=mask.reshape(-1), unit=_unit(col), description=_desc(col, name)
        )
    return Column(flat, unit=_unit(col), description=_desc(col, name))


def _broadcast_column(col: Column | MaskedColumn, n_ccd: int, name: str) -> Column:
    """Repeat a per-transit scalar across the CCDs of its transit."""
    spec = FIELDS.get(name)
    values = np.asarray(_filled(col, _fill_for(col.dtype)))
    if spec is not None and spec.is_bitmask and np.issubdtype(values.dtype, np.integer):
        values = values.astype(_UNSIGNED_FOR.get(values.dtype.name, values.dtype.name))
    repeated = np.repeat(values, n_ccd)
    mask = getattr(col, "mask", None)
    if mask is not None and np.any(mask):
        return MaskedColumn(
            repeated, mask=np.repeat(np.asarray(mask), n_ccd),
            unit=_unit(col), description=_desc(col, name),
        )
    return Column(repeated, unit=_unit(col), description=_desc(col, name))


def _stack(col, n_ccd: int, *, dtype, fill) -> np.ndarray:
    """Stack a variable-length column into an (n_rows, n_ccd) array."""
    out = np.full((len(col), n_ccd), fill, dtype=dtype)
    for i, cell in enumerate(col):
        if cell is None or cell is np.ma.masked:
            continue
        arr = np.atleast_1d(np.ma.getdata(cell))
        if arr.size == 0:
            continue
        out[i, : min(arr.size, n_ccd)] = arr[:n_ccd]
    return out


def _stack_mask(col, n_ccd: int) -> np.ndarray | None:
    """Per-element mask for a variable-length column, or None if nothing masked."""
    mask = np.zeros((len(col), n_ccd), dtype=bool)
    any_masked = False
    for i, cell in enumerate(col):
        if cell is None or cell is np.ma.masked:
            mask[i, :] = True
            any_masked = True
            continue
        arr = np.atleast_1d(cell)
        if arr.size == 0:
            mask[i, :] = True
            any_masked = True
            continue
        cell_mask = np.ma.getmaskarray(arr)
        if cell_mask.any():
            mask[i, : min(arr.size, n_ccd)] = cell_mask[:n_ccd]
            any_masked = True
        if arr.size < n_ccd:
            mask[i, arr.size :] = True
            any_masked = True
    return mask if any_masked else None


def _first_cell(col):
    for cell in col:
        if cell is None or cell is np.ma.masked:
            continue
        arr = np.atleast_1d(np.ma.getdata(cell))
        if arr.size:
            return arr
    return None


def _fill_for(dtype) -> Any:
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.floating):
        return np.nan
    if np.issubdtype(dtype, np.bool_):
        return False
    if np.issubdtype(dtype, np.integer):
        return 0
    return ""


def _filled(col, fill):
    return np.ma.filled(np.ma.getdata(col) if not hasattr(col, "filled") else col, fill)


def _copy_column(col, name: str):
    new = col.copy()
    new.name = name
    return new


def _unit(col):
    return getattr(col, "unit", None)


def _desc(col, name: str) -> str | None:
    desc = getattr(col, "description", None)
    if desc:
        return str(desc)
    spec = FIELDS.get(name)
    return spec.description if spec else None


def _warn_constant(raw: Table, names: tuple[str, ...], warnings: list[str]) -> None:
    for name in names:
        if name not in raw.colnames:
            continue
        values = np.unique(np.asarray(_filled(raw[name], False)))
        if values.size == 1:
            warnings.append(
                f"{name!r} is {values[0]!r} for every transit of this source; "
                "a filter on it cannot do anything"
            )


def _check_nan_not_used(ccd: Table, warnings: list[str]) -> None:
    """Assert the invariant that AGIS never used a missing AL centroid."""
    if "centroid_pos_al" not in ccd.colnames or "used_by_agis_al" not in ccd.colnames:
        return
    al = np.asarray(_filled(ccd["centroid_pos_al"], np.nan), dtype="float64")
    used = np.asarray(_filled(ccd["used_by_agis_al"], False), dtype=bool)
    bad = int((~np.isfinite(al) & used).sum())
    if bad:
        warnings.append(
            f"{bad} CCD observations are flagged used_by_agis_al with a missing AL centroid; "
            "this violates the invariant observed in Gaia DR4_RC3"
        )


def fov_from_transit_id(transit_id: int) -> int:
    """Return the Gaia field of view (1 or 2) encoded in a transit identifier.

    See GAIA-C3-TN-UB-JP-011 p.9.

    Notes
    -----
    Reimplemented locally because ``gaiasupdate.utils.fov_from_transit_id`` in
    version 0.1.2 computes ``np.byte(transit_id >> 15) & 0x03``, which raises
    ``OverflowError`` under NumPy 2.x for every real transit identifier.
    """
    return (int(transit_id) >> 15) & 0x03
