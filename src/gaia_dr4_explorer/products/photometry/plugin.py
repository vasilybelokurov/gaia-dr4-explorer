"""Epoch-photometry product plugin.

Draws on **Gaia DR3**, because DR4 epoch photometry is not published until the
archive opens on 2026-12-02 and is absent from the June-2026 prerelease. Every
panel is therefore labelled DR3, and a DR3 number must never share an
unlabelled axis with a DR4 one.
"""

from __future__ import annotations

from typing import Any

from gaia_dr4_explorer.data.archive import (
    EPOCH_PHOTOMETRY,
    ArchiveError,
    ArchiveUnavailable,
)
from gaia_dr4_explorer.data.dr3_availability import CHECKED_ON, for_source
from gaia_dr4_explorer.domain import ProductPayload, SourceContext
from gaia_dr4_explorer.domain.product import ProductDescriptor, ProductState
from gaia_dr4_explorer.products.astrometry.exporters import export_provenance, export_table
from gaia_dr4_explorer.products.base import ProductError, ProductPlugin
from gaia_dr4_explorer.products.photometry.normalize import normalize_epoch_photometry


class EpochPhotometryPlugin(ProductPlugin):
    """G, BP and RP light curves from Gaia DR3 DataLink."""

    key = "epoch_photometry"
    title = "Photometry"
    release = "Gaia DR3"
    static_safe = False   # needs a live archive request

    def __init__(self, archive=None) -> None:
        self._archive = archive

    def bind(self, archive) -> EpochPhotometryPlugin:
        self._archive = archive
        return self

    def discover(self, context: SourceContext) -> ProductDescriptor:
        """Answer from shipped metadata; never make a network call here."""
        flags = for_source(context.source_id)
        if flags is None:
            return ProductDescriptor(
                key=self.key, title=self.title, state=ProductState.UNKNOWN,
                release=self.release,
                detail="DR3 availability has not been checked for this source",
            )
        if not flags.has_epoch_photometry:
            return self.unavailable(
                "Gaia DR3 published no epoch photometry for this source "
                f"(has_epoch_photometry = false, checked {CHECKED_ON}). "
                "DR4 will publish it for every source from 2026-12-02."
            )
        return self.available(
            f"Gaia DR3 epoch photometry available; phot_variable_flag = "
            f"{flags.phot_variable_flag}",
            variable_flag=flags.phot_variable_flag,
        )

    def load(self, context: SourceContext) -> ProductPayload:
        if self._archive is None:
            raise ProductError("no archive provider is bound")
        try:
            raw, provenance = self._archive.get_datalink_product(
                context.source_id, EPOCH_PHOTOMETRY, release=self.release
            )
        except ArchiveUnavailable as exc:
            raise ProductError(str(exc)) from exc
        except ArchiveError as exc:
            raise ProductError(f"could not load DR3 epoch photometry: {exc}") from exc

        normalized = normalize_epoch_photometry(raw, provenance=provenance)
        return ProductPayload(
            key=self.key,
            raw=raw,
            tables={"epochs": normalized.epochs},
            summary=normalized.summary() | {
                "warnings": normalized.warnings,
                "bands": normalized.bands_present,
                "release": self.release,
            },
            provenance=normalized.provenance,
        )

    def summarize(self, payload: ProductPayload) -> dict[str, Any]:
        return {k: v for k, v in payload.summary.items() if k != "warnings"}

    def build_view(self, context: SourceContext, payload: ProductPayload) -> Any:
        from gaia_dr4_explorer.ui.photometry import PhotometryView

        return PhotometryView(context=context, payload=payload)

    def export(self, payload: ProductPayload, fmt: str) -> bytes:
        if fmt == "provenance":
            if payload.provenance is None:
                raise ProductError("no provenance record attached")
            return export_provenance(payload.provenance)
        if fmt == "raw":
            return export_table(payload.raw, "votable")
        return export_table(payload.table("epochs"), fmt)
