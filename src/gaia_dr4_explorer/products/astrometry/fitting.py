"""DR4-like single-source astrometric fit, delegated to ``gaiasupdate``.

``gaiasupdate`` is treated as an external scientific dependency: none of its
algorithms are reimplemented here.  This module only adapts our data to its
expected input, isolates its mutating behaviour, and lifts its untyped result
dict into a domain object.

Important properties of the upstream package, verified against 0.1.2:

* the supported entry point for snake-case (archive-format) data is
  ``GaiaEpochAstrometryArchive.supdate``; it applies an official column mapping
  and a ``colourFactorAl *= -1e3`` conversion that a hand-rolled rename would
  miss, materially changing faint and red sources;
* its objects are single-use -- fitting mutates ``epoch_data`` in place -- so a
  fresh object is built from a deep copy for every call;
* the reported ``chi2``/``F2`` are computed against the measurement variance
  *excluding* the AGIS source excess noise, while the fit itself weights by the
  total variance including it.  Both are exposed here, separately labelled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd
from astropy.table import Table

from gaia_dr4_explorer.domain.provenance import ProvenanceRecord

#: Model used by the DR4-like configuration.
DR4_LIKE_MODEL = "6p_constrained_colour"

#: Astrometric meaning of each design column, in order.
PARAMETER_NAMES: tuple[str, ...] = (
    "delta_alpha_star",
    "delta_delta",
    "parallax",
    "pmra_star",
    "pmdec",
    "pseudocolour_term",
)

PARAMETER_UNITS: tuple[str, ...] = ("mas", "mas", "mas", "mas / yr", "mas / yr", "1 / nm")

#: Design columns ``gaiasupdate`` builds, in the same order.
DESIGN_COLUMNS: tuple[str, ...] = (
    "sin_theta",
    "cos_theta",
    "parallax_factor_al",
    "sin_theta_time",
    "cos_theta_time",
    "colour_factor_al",
)


class FitError(RuntimeError):
    """Raised when a source update cannot be computed."""


@dataclass
class SourceUpdateResult:
    """Outcome of a DR4-like source update.

    These are values *recomputed from epoch data*, not official catalogue
    values.  Attributes
    ----------
    model : str
        ``gaiasupdate`` model name.
    parameter_names : tuple of str
        Astrometric meaning of each fitted parameter, in order.
    parameters : ndarray
        Fitted parameters.
    parameter_errors : ndarray
        Formal uncertainties.
    covariance : ndarray
        Formal parameter covariance matrix.
    residuals : ndarray
        Post-fit AL residuals of the retained observations, in mas.
    n_measurements : int
        Number of observations that entered the fit.
    n_outliers : int
        Always 0 on the DR4-like path, which uses fixed external variance and
        performs no robust downweighting.
    chi2_measurement_variance, f2_measurement_variance : float
        As reported by ``gaiasupdate``: computed *excluding* the AGIS source
        excess noise.  Not comparable to catalogue ``astrometric_gof_al``.
    chi2_total_variance, f2_total_variance : float
        Recomputed including the AGIS source excess noise, i.e. against the
        same variance the fit was weighted by.
    excess_noise_input_mas : float
        The AGIS source excess noise taken from the data.  An input to the fit,
        not a result of it.
    """

    model: str
    parameter_names: tuple[str, ...]
    parameter_units: tuple[str, ...]
    parameters: np.ndarray
    parameter_errors: np.ndarray
    covariance: np.ndarray
    residuals: np.ndarray
    n_measurements: int
    n_outliers: int
    n_degrees_of_freedom: int
    chi2_measurement_variance: float
    f2_measurement_variance: float
    chi2_total_variance: float
    f2_total_variance: float
    excess_noise_input_mas: float
    provenance: ProvenanceRecord | None = None
    raw_result: dict[str, Any] = field(default_factory=dict, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Parameter name to (value, error, unit)."""
        return {
            name: (float(v), float(e), unit)
            for name, v, e, unit in zip(
                self.parameter_names, self.parameters, self.parameter_errors,
                self.parameter_units, strict=True,
            )
        }

    @property
    def caveat(self) -> str:
        return (
            "Recomputed from epoch astrometry with gaiasupdate; not an official "
            "Gaia catalogue solution."
        )


def gaiasupdate_version() -> str:
    try:
        return version("gaiasupdate")
    except PackageNotFoundError:  # pragma: no cover - only when not installed
        return "not installed"


def fit_dr4_like_single_source(
    raw: Table | pd.DataFrame,
    source_id: int,
    *,
    provenance: ProvenanceRecord | None = None,
) -> SourceUpdateResult:
    """Run the official DR4-like source update for one source.

    Parameters
    ----------
    raw : astropy.table.Table or pandas.DataFrame
        Published transit-level epoch astrometry in archive (snake_case) form.
        The input is deep-copied and never modified.
    source_id : int
        Identifier of the source to fit.
    provenance : ProvenanceRecord, optional
        Provenance of *raw*.

    Returns
    -------
    SourceUpdateResult
    """
    try:
        from gaiasupdate.epoch_astrometry import GaiaEpochAstrometryArchive
        from gaiasupdate.metrics import chi_squared, gaia_f2
    except ImportError as exc:  # pragma: no cover
        raise FitError(f"gaiasupdate is required for the source update: {exc}") from exc

    frame = _to_frame(raw)
    subset = frame[frame["source_id"] == int(source_id)].copy(deep=True)
    if subset.empty:
        raise FitError(f"no rows for source_id {source_id}")

    excess_noise = _excess_noise(subset)

    try:
        # A fresh object per call: gaiasupdate filters and reassigns epoch_data
        # in place, so an instance cannot be reused.
        result = GaiaEpochAstrometryArchive.supdate(subset, int(source_id))
    except Exception as exc:
        raise FitError(f"gaiasupdate failed for source_id {source_id}: {exc}") from exc

    stat = result["solution_statistic"]
    residuals = np.asarray(result["residuals"], dtype="float64")
    n_dof = int(getattr(stat, "n_degrees_of_freedom", len(residuals) - len(result["parameters"])))
    total_variance = _aligned_total_variance(result)
    chi2_total = float(chi_squared(residuals, total_variance))

    params = np.asarray(result["parameters"], dtype="float64")
    if params.size != len(PARAMETER_NAMES):
        raise FitError(
            f"expected {len(PARAMETER_NAMES)} parameters for model {DR4_LIKE_MODEL}, "
            f"got {params.size}; the upstream design matrix has changed and the "
            "parameter naming in this module is no longer valid"
        )

    record = None
    if provenance is not None:
        record = provenance.derive(
            "dr4-like-source-update",
            "GaiaEpochAstrometryArchive.supdate on a deep copy of the published rows",
            f"model={DR4_LIKE_MODEL}, solver=agis, compute_excess_noise=False",
            "excess noise taken from agis_source_excess_noise, not fitted",
            "chi2/F2 reported twice: against measurement variance (as upstream) "
            "and against total variance including excess noise",
            source_id=int(source_id),
        )
        record.software["gaiasupdate"] = gaiasupdate_version()

    return SourceUpdateResult(
        model=str(result.get("model", DR4_LIKE_MODEL)),
        parameter_names=PARAMETER_NAMES,
        parameter_units=PARAMETER_UNITS,
        parameters=params,
        parameter_errors=np.asarray(result["parameters_formal_uncertainty"], dtype="float64"),
        covariance=np.asarray(result["parameter_covariance_matrix_formal"], dtype="float64"),
        residuals=residuals,
        n_measurements=int(result["n_measurements"]),
        n_outliers=int(result["n_outliers"]),
        n_degrees_of_freedom=n_dof,
        chi2_measurement_variance=float(stat.chi2),
        f2_measurement_variance=float(stat.f2),
        chi2_total_variance=chi2_total,
        f2_total_variance=float(gaia_f2(chi2_total, n_dof)),
        excess_noise_input_mas=excess_noise,
        provenance=record,
        raw_result=result,
    )


def _to_frame(raw: Table | pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, Table):
        return raw.to_pandas()
    raise TypeError(f"expected an astropy Table or pandas DataFrame, got {type(raw).__name__}")


def _excess_noise(subset: pd.DataFrame) -> float:
    """AGIS source excess noise, checking it really is constant for the source."""
    if "agis_source_excess_noise" not in subset.columns:
        return float("nan")
    values = np.unique(np.asarray(subset["agis_source_excess_noise"], dtype="float64"))
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size > 1:
        # gaiasupdate silently takes the first row's value; if they ever differ
        # the caller needs to know rather than get a quietly arbitrary choice.
        raise FitError(
            "agis_source_excess_noise is not constant for this source: "
            f"{values.tolist()}; gaiasupdate would silently use the first row"
        )
    return float(values[0])


def _aligned_total_variance(result: dict[str, Any]) -> np.ndarray:
    """Total variance aligned with the retained residuals.

    ``gaiasupdate`` stores ``measurement_variance`` already indexed by
    ``index_keep`` but stores ``total_variance`` un-indexed, so the two can have
    different lengths when observations are dropped.
    """
    total = np.asarray(result["total_variance"], dtype="float64")
    measurement = np.asarray(result["measurement_variance"], dtype="float64")
    if total.size == measurement.size:
        return total
    index_keep = np.asarray(result["index_keep"], dtype=int)
    if total.size >= index_keep.size:
        return total[index_keep]
    raise FitError(
        f"cannot align total_variance (size {total.size}) with "
        f"{index_keep.size} retained observations"
    )
