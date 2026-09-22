"""The product-plugin contract.

A *product* is one dataset attached to a source: epoch astrometry, a light
curve, a spectrum, or a record from an external service.  Plugins are the only
place that knows how to discover, load, summarise, render and export one.

Two rules the interface exists to enforce:

* nothing is loaded until the user selects it;
* a missing product is a normal outcome, reported through
  :class:`~gaia_dr4_explorer.domain.product.ProductState`, never an exception.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from gaia_dr4_explorer.domain import (
    ProductDescriptor,
    ProductPayload,
    ProductState,
    SourceContext,
)


class ProductError(RuntimeError):
    """A product failed to load for a reason the user should see."""


class ProductPlugin(ABC):
    """One inspectable dataset type.

    Subclasses must be cheap to construct and must not perform I/O in
    ``__init__``.
    """

    #: Stable identifier, used in URLs and cache paths.
    key: str = ""
    #: Human-readable tab title.
    title: str = ""
    #: Gaia release this plugin draws from.  Displayed next to every panel so a
    #: DR3 quantity is never mistaken for a DR4 one.
    release: str = "unknown"
    #: False when the plugin needs a live network service, and so cannot run in
    #: a static (Pyodide) build.
    static_safe: bool = True

    @abstractmethod
    def discover(self, context: SourceContext) -> ProductDescriptor:
        """Report availability without loading the product.

        Must be cheap.  If answering would require a network call, return
        :attr:`ProductState.UNKNOWN` rather than making one.
        """

    @abstractmethod
    def load(self, context: SourceContext) -> ProductPayload:
        """Load the product.  Called only after the user selects it."""

    def summarize(self, payload: ProductPayload) -> dict[str, Any]:
        """Scalar summary for the overview page."""
        return dict(payload.summary)

    @abstractmethod
    def build_view(self, context: SourceContext, payload: ProductPayload) -> Any:
        """Return a renderable view.  Must not perform I/O."""

    def export(self, payload: ProductPayload, fmt: str) -> Any:
        """Export the payload.  Override to support more than the default set."""
        raise ProductError(f"{self.key} does not support export format {fmt!r}")

    # ----------------------------------------------------------- convenience

    def unavailable(self, detail: str) -> ProductDescriptor:
        return ProductDescriptor(
            key=self.key,
            title=self.title,
            state=ProductState.UNAVAILABLE,
            release=self.release,
            detail=detail,
        )

    def available(self, detail: str | None = None, **notes: Any) -> ProductDescriptor:
        return ProductDescriptor(
            key=self.key,
            title=self.title,
            state=ProductState.AVAILABLE,
            release=self.release,
            detail=detail,
            notes=notes,
        )
