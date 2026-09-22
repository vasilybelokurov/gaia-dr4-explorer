"""SIMBAD and ADS context.

Both are third-party services. SIMBAD sends ``Access-Control-Allow-Origin: *``
so it could be queried live from a browser; ADS does not, and its API needs a
token that must never be embedded in a public page. Both are therefore served
from the records bundled with the package, fetched once at build time.

Only public bibliographic results are stored. No credential is ever written.
"""

from __future__ import annotations

from typing import Any

from gaia_dr4_explorer.domain import ProductPayload, SourceContext
from gaia_dr4_explorer.domain.product import ProductDescriptor, ProductState
from gaia_dr4_explorer.products.base import ProductError, ProductPlugin

SIMBAD_OBJECT_URL = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident="
ADS_ABS_URL = "https://ui.adsabs.harvard.edu/abs/"


class ContextPlugin(ProductPlugin):
    """Where this object appears in SIMBAD and in the literature."""

    key = "context"
    title = "Context"
    release = "SIMBAD / ADS"
    static_safe = True   # served from the bundle

    def __init__(self, provider=None) -> None:
        self._provider = provider

    def bind(self, provider) -> ContextPlugin:
        self._provider = provider
        return self

    def _records(self, source_id: int) -> tuple[dict | None, dict | None]:
        if self._provider is None:
            return None, None
        simbad = getattr(self._provider, "simbad", dict)() or {}
        ads = getattr(self._provider, "ads", dict)() or {}
        return simbad.get(int(source_id)), ads.get(int(source_id))

    def discover(self, context: SourceContext) -> ProductDescriptor:
        simbad, ads = self._records(context.source_id)
        if simbad is None and ads is None:
            return self.unavailable(
                "This source is not in SIMBAD, by Gaia DR3 identifier or within 5 "
                "arcsec of its position, and has no ADS entry. Six of the twelve "
                "prerelease sources are anonymous field objects."
            )
        bits = []
        if simbad:
            bits.append(f"SIMBAD: {simbad.get('main_id')} ({simbad.get('otype')})")
        if ads and ads.get("num_found"):
            bits.append(f"ADS: {ads['num_found']} papers")
        return self.available("; ".join(bits) or "context available")

    def load(self, context: SourceContext) -> ProductPayload:
        simbad, ads = self._records(context.source_id)
        if simbad is None and ads is None:
            raise ProductError("no external context for this source")
        return ProductPayload(
            key=self.key, raw={"simbad": simbad, "ads": ads},
            tables={},
            summary={
                "main_id": (simbad or {}).get("main_id"),
                "otype": (simbad or {}).get("otype"),
                "nbref": (simbad or {}).get("nbref"),
                "n_cross_ids": len((simbad or {}).get("cross_ids", [])),
                "ads_num_found": (ads or {}).get("num_found", 0),
            },
        )

    def summarize(self, payload: ProductPayload) -> dict[str, Any]:
        return dict(payload.summary)

    def build_view(self, context: SourceContext, payload: ProductPayload) -> Any:
        from gaia_dr4_explorer.ui.context import ContextView

        return ContextView(context=context, payload=payload)

    def export(self, payload: ProductPayload, fmt: str) -> bytes:
        import json

        if fmt in ("json", "raw", "provenance"):
            return json.dumps(payload.raw, indent=2, default=str).encode("utf-8")
        raise ProductError(f"context supports json export only, not {fmt!r}")


def state_label(descriptor: ProductDescriptor) -> str:
    return "available" if descriptor.state is ProductState.AVAILABLE else "unavailable"
