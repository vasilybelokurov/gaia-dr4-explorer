"""BP/RP spectrum plugin (Gaia DR3 XP_SAMPLED)."""

from __future__ import annotations

from typing import Any

from gaia_dr4_explorer.data.archive import XP_SAMPLED, ArchiveError, ArchiveUnavailable
from gaia_dr4_explorer.data.dr3_availability import CHECKED_ON, for_source
from gaia_dr4_explorer.domain import ProductPayload, SourceContext
from gaia_dr4_explorer.domain.product import ProductDescriptor, ProductState
from gaia_dr4_explorer.products.astrometry.exporters import export_provenance, export_table
from gaia_dr4_explorer.products.base import ProductError, ProductPlugin


class XpSpectrumPlugin(ProductPlugin):
    """Mean BP/RP spectrum, sampled on a common wavelength grid."""

    key = "xp_spectrum"
    title = "Spectra"
    release = "Gaia DR3"
    static_safe = True   # served from the bundle in a browser build

    def __init__(self, provider=None) -> None:
        self._provider = provider

    def bind(self, provider) -> XpSpectrumPlugin:
        self._provider = provider
        return self

    def discover(self, context: SourceContext) -> ProductDescriptor:
        flags = for_source(context.source_id)
        if flags is None:
            return ProductDescriptor(
                key=self.key, title=self.title, state=ProductState.UNKNOWN,
                release=self.release,
                detail="DR3 availability has not been checked for this source",
            )
        if not flags.has_xp_sampled:
            return self.unavailable(
                "Gaia DR3 published no BP/RP spectrum for this source "
                f"(has_xp_sampled = false, checked {CHECKED_ON}). DR4 will publish "
                "mean and epoch spectra from 2026-12-02."
            )
        return self.available(
            f"Gaia DR3 BP/RP mean spectrum available; BP−RP = {flags.bp_rp:.3f} mag",
            bp_rp=flags.bp_rp,
        )

    def load(self, context: SourceContext) -> ProductPayload:
        if self._provider is None:
            raise ProductError("no product provider is bound")
        try:
            raw, provenance = self._provider.get_datalink_product(
                context.source_id, XP_SAMPLED, release=self.release
            )
        except ArchiveUnavailable as exc:
            raise ProductError(str(exc)) from exc
        except ArchiveError as exc:
            raise ProductError(f"could not load the BP/RP spectrum: {exc}") from exc

        import numpy as np

        wl = np.asarray(np.ma.filled(raw["wavelength"], np.nan), dtype="float64")
        flux = np.asarray(np.ma.filled(raw["flux"], np.nan), dtype="float64")
        finite = np.isfinite(flux)
        return ProductPayload(
            key=self.key, raw=raw, tables={"spectrum": raw},
            summary={
                "release": self.release,
                "n_samples": len(raw),
                "wavelength_min_nm": float(np.nanmin(wl)),
                "wavelength_max_nm": float(np.nanmax(wl)),
                "peak_wavelength_nm": (
                    float(wl[finite][np.argmax(flux[finite])]) if finite.any() else float("nan")
                ),
            },
            provenance=provenance,
        )

    def summarize(self, payload: ProductPayload) -> dict[str, Any]:
        return dict(payload.summary)

    def build_view(self, context: SourceContext, payload: ProductPayload) -> Any:
        from gaia_dr4_explorer.ui.spectra import SpectrumView

        return SpectrumView(context=context, payload=payload)

    def export(self, payload: ProductPayload, fmt: str) -> bytes:
        if fmt == "provenance":
            if payload.provenance is None:
                raise ProductError("no provenance record attached")
            return export_provenance(payload.provenance)
        if fmt == "raw":
            return export_table(payload.raw, "votable")
        return export_table(payload.table("spectrum"), fmt)
