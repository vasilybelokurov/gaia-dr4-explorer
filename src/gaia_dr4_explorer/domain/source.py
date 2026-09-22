"""Identity of the object under inspection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gaia_dr4_explorer.domain.provenance import ProvenanceRecord


@dataclass(frozen=True, order=True)
class SourceKey:
    """A Gaia source identifier together with the release that defines it.

    A ``source_id`` is release-specific.  A DR3 identifier must never be assumed
    to denote the same object as the numerically equal DR4 identifier, so the
    two are never interchangeable and a cache entry is never keyed on the
    integer alone.
    """

    release: str
    source_id: int

    def __post_init__(self) -> None:
        if not str(self.release).strip():
            raise ValueError("release must be a non-empty string")
        if int(self.source_id) <= 0:
            raise ValueError(f"source_id must be a positive integer, got {self.source_id!r}")

    def __str__(self) -> str:
        return f"{self.release} {self.source_id}"

    @property
    def cache_token(self) -> str:
        """Stable token combining release and identifier, safe in a filename."""
        from gaia_dr4_explorer.config import _slug

        return f"{_slug(self.release)}__{self.source_id}"


@dataclass
class SourceContext:
    """The currently selected object and what is known about it.

    Holds no GUI objects.  ``page_metadata`` carries values quoted by the ESA
    prerelease page (which are rounded, and are *not* measurements we made);
    ``measured`` carries quantities computed from the VOTable.  They are kept
    apart so the interface can always say which is which.
    """

    key: SourceKey
    common_name: str | None = None
    sample_category: str | None = None
    page_metadata: dict[str, Any] = field(default_factory=dict)
    measured: dict[str, Any] = field(default_factory=dict)
    provenance: ProvenanceRecord | None = None

    @property
    def release(self) -> str:
        return self.key.release

    @property
    def source_id(self) -> int:
        return self.key.source_id

    @property
    def label(self) -> str:
        """Short human label.

        Most prerelease sources are not named on the release page, so fall back
        to the sample category rather than repeating the identifier, which the
        interface shows separately.
        """
        if self.common_name:
            return self.common_name
        if self.sample_category:
            return f"Unnamed {self.sample_category} example"
        return f"Source {self.source_id}"

    @property
    def is_release_candidate(self) -> bool:
        """True when the release string marks a pre-publication candidate."""
        return "RC" in self.key.release.upper()
