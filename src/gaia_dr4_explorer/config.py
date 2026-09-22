"""Application configuration.

All paths and remote locations live here so that tests can redirect them without
monkey-patching the modules that use them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import user_cache_dir

#: Official ESA prerelease archive, announced 2026-06-26.
PRERELEASE_URL = (
    "https://anonftp.cosmos.esa.int/pub/GAIA_PUBLIC_DATA/Gaia_DR4/dr4-prerelease/"
    "gaia-dr4-prerelease-epoch-astrometry_2026-06-26.zip"
)

#: SHA-256 of the archive as published.  Verified 2026-09-22.
PRERELEASE_SHA256 = "07f0e8d9ac97a29ea376a0c7242de3124d2a08ad72aba0958d6575d94d35fa0b"

#: Member of the archive holding the epoch astrometry VOTable.
PRERELEASE_MEMBER = "GAIA_DR4_PRERELEASE_EPOCH_ASTROMETRY_RAW.xml"

#: Number of sources the prerelease is documented to contain.
PRERELEASE_N_SOURCES = 12

_ENV_CACHE = "GAIA_DR4_EXPLORER_CACHE"
_ENV_LOCAL_ZIP = "GAIA_DR4_EXPLORER_PRERELEASE_ZIP"


@dataclass(frozen=True)
class AppConfig:
    """Resolved runtime configuration.

    Parameters
    ----------
    cache_dir : Path
        Root of the on-disk cache.
    prerelease_url : str
        Where to fetch the prerelease archive from.
    prerelease_sha256 : str
        Expected checksum of the archive.
    local_prerelease_zip : Path or None
        If set, this file is used instead of downloading.  Tests set it to a
        fixture so that the default suite never touches the network.
    allow_network : bool
        When False, any attempt to download raises instead of reaching out.
    """

    cache_dir: Path
    prerelease_url: str = PRERELEASE_URL
    prerelease_sha256: str = PRERELEASE_SHA256
    local_prerelease_zip: Path | None = None
    allow_network: bool = True

    @classmethod
    def from_env(cls, **overrides: object) -> AppConfig:
        """Build a config from environment variables, then apply *overrides*."""
        cache = os.environ.get(_ENV_CACHE)
        local = os.environ.get(_ENV_LOCAL_ZIP)
        base = {
            "cache_dir": Path(cache) if cache else Path(user_cache_dir("gaia-dr4-explorer")),
            "local_prerelease_zip": Path(local) if local else None,
        }
        base.update(overrides)  # type: ignore[arg-type]
        return cls(**base)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CacheLayout:
    """On-disk cache layout.

    Raw products are immutable; normalized products are derived and may be
    regenerated at any time.  A cache entry is never keyed on ``source_id``
    alone -- the release is always part of the path.
    """

    root: Path
    _made: set[Path] = field(default_factory=set, repr=False, compare=False)

    @property
    def prerelease(self) -> Path:
        return self.root / "prerelease"

    @property
    def metadata(self) -> Path:
        return self.root / "metadata"

    def raw(self, release: str, source_id: int) -> Path:
        return self.root / "raw" / _slug(release) / str(source_id)

    def normalized(self, release: str, source_id: int) -> Path:
        return self.root / "normalized" / _slug(release) / str(source_id)

    def ensure(self, path: Path) -> Path:
        """Create *path* if needed and return it."""
        path.mkdir(parents=True, exist_ok=True)
        return path


def _slug(release: str) -> str:
    """Make *release* safe for use as a single path component.

    ``"Gaia DR4_RC3"`` becomes ``"Gaia-DR4_RC3"``.
    """
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in release.strip())
