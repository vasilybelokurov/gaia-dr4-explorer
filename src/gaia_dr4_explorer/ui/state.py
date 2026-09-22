"""Application state.

A ``param`` object, deliberately free of Panel widgets, so that state
transitions can be tested without rendering anything.
"""

from __future__ import annotations

import param

from gaia_dr4_explorer.data.catalog import catalog
from gaia_dr4_explorer.domain import SourceContext, SourceKey


class AppState(param.Parameterized):
    """What the user has selected and what has been loaded for it."""

    source_id = param.Integer(default=0, doc="Currently selected Gaia source identifier")
    release = param.String(default="Gaia DR4_RC3")
    product_key = param.String(default="epoch_astrometry")

    show_used = param.Boolean(default=True, doc="Show observations AGIS used")
    show_rejected = param.Boolean(default=True, doc="Show observations AGIS rejected")
    ccd_selection = param.ListSelector(default=[], objects=[])
    max_sigma_al = param.Number(default=0.0, bounds=(0.0, None), doc="0 disables the cut")
    matrix_quantity = param.String(default="used_by_agis_al")

    status = param.String(default="")
    error = param.String(default="")

    def __init__(self, provider=None, **params):
        super().__init__(**params)
        self._provider = provider
        self._payloads: dict[tuple[str, int, str], object] = {}
        self._fits: dict[tuple[str, int], object] = {}

    # ------------------------------------------------------------- selection

    @property
    def provider(self):
        return self._provider

    def available_source_ids(self) -> list[int]:
        if self._provider is None:
            return sorted(catalog())
        try:
            return self._provider.source_ids()
        except Exception as exc:
            self.error = str(exc)
            return []

    @property
    def key(self) -> SourceKey:
        return SourceKey(release=self.release, source_id=self.source_id)

    def context(self) -> SourceContext:
        """Domain context for the current selection."""
        entry = catalog().get(self.source_id)
        return SourceContext(
            key=self.key,
            common_name=(entry.common_name or None) if entry else None,
            sample_category=entry.sample_category if entry else None,
            page_metadata=(
                {"g_mag": entry.page_g_mag, "parallax_mas": entry.page_parallax_mas}
                if entry
                else {}
            ),
        )

    # --------------------------------------------------------------- loading

    def payload(self, plugin):
        """Load and memoise the payload for the current source.

        Loading happens here and only when asked -- never on selection change.
        """
        cache_key = (self.release, self.source_id, plugin.key)
        if cache_key not in self._payloads:
            self.status = f"loading {plugin.title}..."
            try:
                self._payloads[cache_key] = plugin.load(self.context())
                self.error = ""
            except Exception as exc:
                self.error = f"{plugin.title}: {exc}"
                raise
            finally:
                self.status = ""
        return self._payloads[cache_key]

    def fit(self, *, force: bool = False):
        """Run (or return the memoised) DR4-like source update."""
        cache_key = (self.release, self.source_id)
        if force:
            self._fits.pop(cache_key, None)
        if cache_key not in self._fits:
            from gaia_dr4_explorer.products.astrometry.fitting import (
                fit_dr4_like_single_source,
            )

            raw = self._provider.raw_table_for(self.source_id)
            self._fits[cache_key] = fit_dr4_like_single_source(
                raw, self.source_id, provenance=self._provider.provenance
            )
        return self._fits[cache_key]

    def has_fit(self) -> bool:
        return (self.release, self.source_id) in self._fits

    def reset_source(self) -> None:
        """Drop cached products, e.g. after an explicit reload."""
        self._payloads.clear()
        self._fits.clear()
        self.error = ""
