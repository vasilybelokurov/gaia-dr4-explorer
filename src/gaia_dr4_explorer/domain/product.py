"""Descriptors and payloads for the products attached to a source."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from gaia_dr4_explorer.domain.provenance import ProvenanceRecord


class ProductState(StrEnum):
    """Availability of a product for the current source.

    A missing product is a normal outcome, not an error.  ``UNKNOWN`` means we
    have not looked yet -- distinguishing it from ``UNAVAILABLE`` matters,
    because checking may require a network call we have chosen not to make.
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"
    LOADING = "loading"
    ERROR = "error"


@dataclass(frozen=True)
class ProductDescriptor:
    """What is known about a product without loading it."""

    key: str
    title: str
    state: ProductState = ProductState.UNKNOWN
    release: str | None = None
    detail: str | None = None
    n_rows: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def is_loadable(self) -> bool:
        return self.state in (ProductState.AVAILABLE, ProductState.UNKNOWN)


@dataclass
class ProductPayload:
    """A loaded product: the untouched raw form plus derived representations.

    ``raw`` is never modified.  Everything in ``tables`` is derived and may be
    regenerated from ``raw`` at any time.
    """

    key: str
    raw: Any
    tables: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceRecord | None = None

    def table(self, name: str) -> Any:
        try:
            return self.tables[name]
        except KeyError:
            raise KeyError(
                f"product {self.key!r} has no derived table {name!r}; "
                f"available: {sorted(self.tables)}"
            ) from None
