"""Release-agnostic domain objects.  These must never hold GUI objects."""

from gaia_dr4_explorer.domain.product import ProductDescriptor, ProductPayload, ProductState
from gaia_dr4_explorer.domain.provenance import ProvenanceRecord
from gaia_dr4_explorer.domain.source import SourceContext, SourceKey

__all__ = [
    "ProductDescriptor",
    "ProductPayload",
    "ProductState",
    "ProvenanceRecord",
    "SourceContext",
    "SourceKey",
]
