"""Epoch-astrometry product plugin."""

from __future__ import annotations

from typing import Any

from gaia_dr4_explorer.domain import ProductPayload, SourceContext
from gaia_dr4_explorer.domain.product import ProductDescriptor
from gaia_dr4_explorer.products.astrometry.exporters import export_provenance, export_table
from gaia_dr4_explorer.products.astrometry.normalize import normalize_epoch_astrometry
from gaia_dr4_explorer.products.base import ProductError, ProductPlugin


class EpochAstrometryPlugin(ProductPlugin):
    """Transit- and CCD-level astrometry from the DR4 prerelease."""

    key = "epoch_astrometry"
    title = "Epoch astrometry"
    release = "Gaia DR4_RC3"
    static_safe = True

    def __init__(self, provider=None) -> None:
        self._provider = provider

    def bind(self, provider) -> EpochAstrometryPlugin:
        """Attach a data provider.  Kept out of ``__init__`` so the plugin can
        be registered before any provider exists."""
        self._provider = provider
        return self

    def discover(self, context: SourceContext) -> ProductDescriptor:
        if self._provider is None:
            return self.unavailable("no data provider is bound")
        try:
            ids = self._provider.source_ids()
        except Exception as exc:  # provider failures are a normal state here
            return ProductDescriptor(
                key=self.key, title=self.title, state="error",
                release=self.release, detail=str(exc),
            )
        if context.source_id not in ids:
            return self.unavailable(
                f"{context.source_id} is not in the {self.release} prerelease"
            )
        return self.available("epoch astrometry published for this source")

    def load(self, context: SourceContext) -> ProductPayload:
        if self._provider is None:
            raise ProductError("no data provider is bound")
        raw = self._provider.raw_table_for(context.source_id)
        normalized = normalize_epoch_astrometry(raw, provenance=self._provider.provenance)
        return ProductPayload(
            key=self.key,
            raw=raw,
            tables={
                "transits": normalized.transits,
                "ccd": normalized.ccd,
            },
            summary=normalized.summary() | {"warnings": normalized.warnings},
            provenance=normalized.provenance,
        )

    def summarize(self, payload: ProductPayload) -> dict[str, Any]:
        return {k: v for k, v in payload.summary.items() if k != "warnings"}

    def build_view(self, context: SourceContext, payload: ProductPayload) -> Any:
        from gaia_dr4_explorer.ui.astrometry import AstrometryView

        return AstrometryView(context=context, payload=payload)

    def export(self, payload: ProductPayload, fmt: str) -> bytes:
        if fmt == "provenance":
            if payload.provenance is None:
                raise ProductError("no provenance record attached")
            return export_provenance(payload.provenance)
        if fmt == "raw":
            return export_table(payload.raw, "votable")
        table = payload.table("ccd")
        return export_table(table, fmt)
