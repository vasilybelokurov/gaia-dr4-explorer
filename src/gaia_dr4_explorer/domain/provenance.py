"""Provenance records.

Every derived dataset carries one.  A number displayed in the application must
be traceable through these records back to a named Gaia field or a documented
transformation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ProvenanceRecord:
    """Where a dataset came from and what was done to it.

    Parameters
    ----------
    kind : str
        Short label, e.g. ``"prerelease-archive"`` or ``"ccd-normalized"``.
    release : str
        Gaia release string exactly as the data declares it.  For the June-2026
        prerelease this is ``"Gaia DR4_RC3"`` -- a release *candidate*.
    source_url : str or None
        Remote location the raw data came from, if any.
    sha256 : str or None
        Checksum of the raw artefact, if any.
    retrieved_at : str
        UTC ISO-8601 timestamp of retrieval or derivation.
    solution_id : int or None
        Gaia solution identifier carried by the data.
    source_id : int or None
        Gaia source identifier, meaningful only together with *release*.
    derived_from : tuple of ProvenanceRecord
        Provenance of the inputs, for derived products.
    steps : tuple of str
        Ordered, human-readable description of the transformations applied.
    software : dict
        Package name to version for anything that touched the data.
    notes : dict
        Free-form extra metadata.
    """

    kind: str
    release: str
    source_url: str | None = None
    sha256: str | None = None
    retrieved_at: str = field(default_factory=_utcnow)
    solution_id: int | None = None
    source_id: int | None = None
    derived_from: tuple[ProvenanceRecord, ...] = ()
    steps: tuple[str, ...] = ()
    software: dict[str, str] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)

    def derive(self, kind: str, *steps: str, **notes: Any) -> ProvenanceRecord:
        """Return a child record describing a transformation of this dataset."""
        return ProvenanceRecord(
            kind=kind,
            release=self.release,
            solution_id=self.solution_id,
            source_id=self.source_id,
            derived_from=(self,),
            steps=steps,
            software=dict(self.software),
            notes=notes,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str, **kwargs)
